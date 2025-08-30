# -*- coding: utf-8 -*-
"""
Training and experiment orchestration for Phi-GaLore experiments.

Implements:
 - Toy SciML models (MLPField, TinyTransformerRegressor)
 - Phi-GaLore wrapper with hierarchical Count-Sketch + tiny SVD and optional physics basis
 - Naive GaLore-like baseline wrapper
 - Training loops and three experiments producing high-quality PDF plots under .research/iteration1/images

All imports from within src use relative imports.
"""
import os
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support being imported as a package (src.train) or as a script (train)
try:
    from .preprocess import (
        set_seed,
        autocast_if_cuda,
        get_device_and_mp_dtype,
        gen_darcy_like_batch,
        gen_burgers_bc_variants,
        gen_molecular_like_batch,
        build_divfree_basis_fourier,
    )
    from .evaluate import (
        physics_loss,
        relative_l2,
        divergence_and_laplacian,
        ensure_pdf_backend_and_style,
        save_plot,
    )
except Exception:  # fallback for direct script execution
    from preprocess import (  # type: ignore
        set_seed,
        autocast_if_cuda,
        get_device_and_mp_dtype,
        gen_darcy_like_batch,
        gen_burgers_bc_variants,
        gen_molecular_like_batch,
        build_divfree_basis_fourier,
    )
    from evaluate import (  # type: ignore
        physics_loss,
        relative_l2,
        divergence_and_laplacian,
        ensure_pdf_backend_and_style,
        save_plot,
    )


# ------------------------------
# Models
# ------------------------------

class MLPField(nn.Module):
    """Global MLP mapping an input grid to a full velocity field (u,v) over the grid.
    The last layer is a large Linear from hidden -> 2*Nx*Ny, enabling physics-basis projection on its rows.
    """
    def __init__(self, Nx: int, Ny: int, in_ch: int = 1, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.Nx, self.Ny = Nx, Ny
        self.in_dim = in_ch * Nx * Ny
        layers = []
        dims = [self.in_dim, hidden] + [hidden] * (depth - 1)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        self.head_vel = nn.Linear(hidden, 2 * Nx * Ny)  # row dimension matches 2*Nx*Ny

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, Nx, Ny]
        B = x.size(0)
        x_flat = x.view(B, -1)
        h = self.backbone(x_flat)
        out = self.head_vel(h)
        return out.view(B, 2, self.Nx, self.Ny)


class TinySelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp_fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, L, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = attn_scores.softmax(dim=-1)
        ctx = attn @ v  # [B, H, L, D]
        ctx = ctx.transpose(1, 2).reshape(B, L, C)
        x = x + self.dropout(self.attn_out(ctx))
        h2 = self.ln2(x)
        h2 = self.mlp_fc2(F.gelu(self.mlp_fc1(h2)))
        x = x + self.dropout(h2)
        return x


class TinyTransformerRegressor(nn.Module):
    """Transformer over token sequence; predicts scalar property (toy molecular model).
    """
    def __init__(self, seq_len: int = 32, dim: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(4, dim)  # token features: (x,y,z,atomic_num_norm)
        self.blocks = nn.ModuleList([TinySelfAttentionBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, 1)
        self.ln = nn.LayerNorm(dim)
        self.seq_len = seq_len

        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, 4]
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        pooled = h.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        return out


# ------------------------------
# Phi-GaLore primitives
# ------------------------------

class CountSketch:
    def __init__(self, d: int, r: int, device: torch.device, seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        self.idx = torch.randint(low=0, high=r, size=(d,), generator=g, device=device)
        self.sign = torch.where(
            torch.rand(d, generator=g, device=device) > 0.5,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        )
        self.r = r
        self.d = d
        self.device = device

    @torch.no_grad()
    def left(self, X: torch.Tensor) -> torch.Tensor:  # X: [m, n] -> [r, n]
        m = X.size(0)
        S = torch.zeros(self.r, X.size(1), device=X.device, dtype=X.dtype)
        S.index_add_(0, self.idx[:m], X * self.sign[:m].unsqueeze(1))
        return S

    @torch.no_grad()
    def right(self, X: torch.Tensor) -> torch.Tensor:  # X: [m, n] -> [m, r]
        n = X.size(1)
        S = torch.zeros(X.size(0), self.r, device=X.device, dtype=X.dtype)
        S.index_add_(1, self.idx[:n], X * self.sign[:n].unsqueeze(0))
        return S


@torch.no_grad()
def unsketch_rows(U_sketch: torch.Tensor, cs_left: CountSketch) -> torch.Tensor:
    # Lift from sketch-space [r, k] to row-space [m, k]
    m = cs_left.d
    r = cs_left.r
    counts = torch.bincount(cs_left.idx[:m], minlength=r).clamp_min(1).to(U_sketch.device)
    inv_sqrt = (1.0 / counts.float()).sqrt().unsqueeze(1)  # [r, 1]
    U_scaled = U_sketch.float() * inv_sqrt  # [r, k]
    P = torch.zeros(m, U_sketch.size(1), device=U_sketch.device, dtype=U_sketch.dtype)
    P.index_add_(0, cs_left.idx[:m], U_scaled * cs_left.sign[:m].unsqueeze(1))
    Q, _ = torch.linalg.qr(P.float(), mode='reduced')
    return Q.contiguous()


@torch.no_grad()
def project_grad_with_basis(G: torch.Tensor,
                            P: Optional[torch.Tensor] = None,
                            B: Optional[torch.Tensor] = None) -> torch.Tensor:
    # G: [m, n], P: learned basis [m, kp], B: physics basis [m, kb]
    if (P is None or P.numel() == 0) and (B is None or B.numel() == 0):
        return G
    if P is None or P.numel() == 0:
        Puse = B.float()
    elif B is None or B.numel() == 0:
        Puse = P.float()
    else:
        Pcat = torch.cat([P.float(), B.float()], dim=1)
        Puse, _ = torch.linalg.qr(Pcat, mode='reduced')
    return (Puse @ (Puse.T @ G.float())).to(G.dtype)


class PhiGaLoreGroupState:
    def __init__(self, m: int, n: int, r: int, ema: float, device: torch.device, mp_store_dtype: torch.dtype, seed: int = 0):
        self.cs_l = CountSketch(m, r, device=device, seed=seed)
        self.cs_r = CountSketch(n, r, device=device, seed=seed + 1)
        self.r = r
        self.ema = ema
        self.sketch_ema = torch.zeros(r, r, device=device, dtype=mp_store_dtype)
        self.step = 0
        self.U_sketch: Optional[torch.Tensor] = None  # [r, k]
        self.U_row: Optional[torch.Tensor] = None     # [m, k]

    @torch.no_grad()
    def update_sketch(self, G: torch.Tensor):
        S = self.cs_l.left(G)    # [r, n]
        Gt = self.cs_r.right(S)  # [r, r]
        self.sketch_ema = self.ema * self.sketch_ema + (1.0 - self.ema) * Gt.to(self.sketch_ema.dtype)
        self.step += 1

    @torch.no_grad()
    def refine_basis(self, k: Optional[int] = None):
        C = self.sketch_ema.float() @ self.sketch_ema.float().T
        Uc, _, _ = torch.linalg.svd(C)
        k_use = k if k is not None else self.r
        self.U_sketch = Uc[:, :k_use].contiguous()  # [r, k]
        self.U_row = None

    @torch.no_grad()
    def lift_if_needed(self) -> Optional[torch.Tensor]:
        if self.U_row is None and self.U_sketch is not None:
            self.U_row = unsketch_rows(self.U_sketch, self.cs_l)
        return self.U_row


class PhiGaLoreWrapper:
    def __init__(self,
                 model: nn.Module,
                 base_opt: torch.optim.Optimizer,
                 r: int = 32,
                 T: int = 400,
                 ema: float = 0.9,
                 share_by: Optional[Callable[[str, nn.Parameter], str]] = None,
                 symmetry_registry: Optional[Dict[str, torch.Tensor]] = None,
                 large_only_threshold: int = 0,
                 mp_store_dtype: torch.dtype = torch.float16):
        self.model = model
        self.base_opt = base_opt
        self.r = r
        self.T = T
        self.ema = ema
        self.share_by = share_by
        self.symmetry_registry = symmetry_registry or {}
        self.group_states: Dict[str, PhiGaLoreGroupState] = {}
        self.large_only_threshold = large_only_threshold
        self._proj_ms = 0.0
        self._use_cuda_timing = torch.cuda.is_available()
        self.mp_store_dtype = mp_store_dtype

        if self._use_cuda_timing:
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end = torch.cuda.Event(enable_timing=True)
        else:
            self._cpu_t0 = 0.0

    def group_key(self, name: str, param: nn.Parameter) -> str:
        if self.share_by is None:
            return name
        key = self.share_by(name, param)
        return key if key is not None else name

    @torch.no_grad()
    def pre_step_projection(self):
        if self._use_cuda_timing:
            self._ev_start.record()
        else:
            self._cpu_t0 = time.perf_counter()

        # 1) Update sketches per group once per step
        seen = set()
        for name, p in self.model.named_parameters():
            if p.grad is None or p.grad.ndim < 2:
                continue
            if p.numel() < self.large_only_threshold:
                continue
            key = self.group_key(name, p)
            if key in seen:
                continue
            seen.add(key)
            m, n = p.grad.shape
            st = self.group_states.get(key)
            if st is None:
                st = PhiGaLoreGroupState(m, n, r=self.r, ema=self.ema, device=p.grad.device, mp_store_dtype=self.mp_store_dtype)
                self.group_states[key] = st
            st.update_sketch(p.grad)

        # 2) Periodic tiny SVD per group
        for _, st in self.group_states.items():
            if st.step > 0 and st.step % self.T == 0:
                st.refine_basis(k=self.r)

        # 3) Project each grad with learned basis and optional physics basis
        for name, p in self.model.named_parameters():
            if p.grad is None or p.grad.ndim < 2:
                continue
            if p.numel() < self.large_only_threshold:
                continue
            key = self.group_key(name, p)
            st = self.group_states[key]
            U_row = st.lift_if_needed()
            B = self.symmetry_registry.get(key, None)
            p.grad.data = project_grad_with_basis(p.grad.data, P=U_row, B=B)

        if self._use_cuda_timing:
            self._ev_end.record()
            torch.cuda.synchronize()
            self._proj_ms = self._ev_start.elapsed_time(self._ev_end)
        else:
            self._proj_ms = (time.perf_counter() - self._cpu_t0) * 1000.0

    def step(self):
        self.pre_step_projection()
        self.base_opt.step()

    def zero_grad(self, set_to_none: bool = True):
        self.base_opt.zero_grad(set_to_none=set_to_none)

    def projection_overhead_ms(self) -> float:
        return float(self._proj_ms)


class GaLoreWrapper:
    """Naive GaLore-like wrapper: periodic full SVD on gradients for each large 2D parameter.
    Stores last computed row-basis U_k and reuses for intermediate steps.
    """
    def __init__(self,
                 model: nn.Module,
                 base_opt: torch.optim.Optimizer,
                 r: int = 64,
                 T: int = 200,
                 large_only_threshold: int = 0):
        self.model = model
        self.base_opt = base_opt
        self.r = r
        self.T = T
        self.large_only_threshold = large_only_threshold
        self.step_count = 0
        self.bases: Dict[str, torch.Tensor] = {}
        self._proj_ms = 0.0
        self._use_cuda_timing = torch.cuda.is_available()
        if self._use_cuda_timing:
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end = torch.cuda.Event(enable_timing=True)
        else:
            self._cpu_t0 = 0.0

    @torch.no_grad()
    def pre_step_projection(self):
        self.step_count += 1
        if self._use_cuda_timing:
            self._ev_start.record()
        else:
            self._cpu_t0 = time.perf_counter()

        for name, p in self.model.named_parameters():
            if p.grad is None or p.grad.ndim < 2:
                continue
            if p.numel() < self.large_only_threshold:
                continue
            # periodic SVD to refresh basis
            if (self.step_count - 1) % self.T == 0:
                G = p.grad.detach().float()
                try:
                    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
                except RuntimeError:
                    U, S, Vh = torch.linalg.svd(G.cpu(), full_matrices=False)
                    U = U.to(G.device)
                self.bases[name] = U[:, : min(self.r, U.size(1))].contiguous()
            Urow = self.bases.get(name, None)
            if Urow is not None and Urow.numel() > 0:
                p.grad.data = (Urow @ (Urow.T @ p.grad.float())).to(p.grad.dtype)

        if self._use_cuda_timing:
            self._ev_end.record()
            torch.cuda.synchronize()
            self._proj_ms = self._ev_start.elapsed_time(self._ev_end)
        else:
            self._proj_ms = (time.perf_counter() - self._cpu_t0) * 1000.0

    def step(self):
        self.pre_step_projection()
        self.base_opt.step()

    def zero_grad(self, set_to_none: bool = True):
        self.base_opt.zero_grad(set_to_none=set_to_none)

    def projection_overhead_ms(self) -> float:
        return float(self._proj_ms)


# ------------------------------
# Diagnostics utilities
# ------------------------------

@torch.no_grad()
def stable_rank(G: torch.Tensor) -> float:
    Gf = G.float()
    fro2 = torch.sum(Gf * Gf)
    try:
        top1 = torch.linalg.norm(Gf, 2)
    except RuntimeError:
        top1 = torch.linalg.norm(Gf.cpu(), 2).to(Gf.device)
    sr = (fro2 / (top1 * top1 + 1e-12)).item()
    return float(sr)

def cos_sim_full_vs_projected(model: nn.Module,
                              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                              batch: Tuple[torch.Tensor, torch.Tensor],
                              optimizer_wrapper,
                              target_param_names: List[str],
                              mp_dtype: torch.dtype) -> Dict[str, float]:
    x, y = batch
    # Full gradient
    model.zero_grad(set_to_none=True)
    with autocast_if_cuda(mp_dtype):
        out = model(x)
        loss = loss_fn(out, y)
    loss.backward()
    grads_full = {n: p.grad.detach().clone() for n, p in model.named_parameters() if n in target_param_names}

    # Projected gradient
    model.zero_grad(set_to_none=True)
    with autocast_if_cuda(mp_dtype):
        out2 = model(x)
        loss2 = loss_fn(out2, y)
    loss2.backward()
    optimizer_wrapper.pre_step_projection()
    grads_proj = {n: p.grad.detach().clone() for n, p in model.named_parameters() if n in target_param_names}

    cos_sims = {}
    for n in target_param_names:
        g = grads_full[n].flatten().float()
        gp = grads_proj[n].flatten().float()
        denom = (g.norm() * gp.norm() + 1e-12)
        cos_sims[n] = (torch.dot(g, gp) / denom).item()
    model.zero_grad(set_to_none=True)
    return cos_sims


# ------------------------------
# Training helpers
# ------------------------------

@dataclass
class TrainResult:
    losses: List[float]
    sranks: List[float]
    overhead_ms: List[float]


def pick_largest_linear_param_names(model: nn.Module, top_k: int = 2) -> List[str]:
    mats = []
    for n, p in model.named_parameters():
        if p.ndim == 2:
            mats.append((n, p.numel()))
    mats.sort(key=lambda x: -x[1])
    return [n for n, _ in mats[:top_k]]


def train_one(model: nn.Module,
              optimizer_wrapper,
              data_iter: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
              loss_fn: Callable,
              mp_dtype: torch.dtype,
              steps: int = 200,
              log_every: int = 20,
              srank_param_name: Optional[str] = None) -> TrainResult:
    model.train()
    losses, sranks, overhead_ms = [], [], []
    for step in range(steps):
        x, y = data_iter()
        optimizer_wrapper.zero_grad(set_to_none=True)
        with autocast_if_cuda(mp_dtype):
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward()
        optimizer_wrapper.step()
        l = float(loss.detach().cpu().item())
        losses.append(l)
        # record stable rank on chosen param
        if srank_param_name is not None:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n == srank_param_name and p.grad is not None and p.grad.ndim == 2:
                        sr = stable_rank(p.grad)
                        sranks.append(sr)
                        break
        # handle both wrapped optimizers and plain torch optimizers
        overhead = 0.0
        if hasattr(optimizer_wrapper, 'projection_overhead_ms'):
            try:
                overhead = float(optimizer_wrapper.projection_overhead_ms())
            except Exception:
                overhead = 0.0
        overhead_ms.append(overhead)
        if (step + 1) % log_every == 0 or step == 0:
            print(f"Step {step+1}/{steps} | loss={l:.6f} | proj_overhead_ms={overhead_ms[-1]:.3f}")
    return TrainResult(losses=losses, sranks=sranks, overhead_ms=overhead_ms)


# ------------------------------
# Experiments
# ------------------------------

def experiment_1_micro_bench(images_dir: str,
                             steps: int,
                             Nx: int,
                             Ny: int,
                             device: torch.device,
                             mp_dtype: torch.dtype):
    print("\n=== Experiment 1: Gradient Low-Rankness and Projection Efficiency (Micro-bench) ===")
    set_seed(42)
    ensure_pdf_backend_and_style()

    # Model A: Grid-to-velocity MLP (Darcy-like)
    modelA = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)
    base_optA_full = torch.optim.AdamW(modelA.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)

    modelA_phi = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)
    modelA_gal = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)
    modelA_phi.load_state_dict(modelA.state_dict())
    modelA_gal.load_state_dict(modelA.state_dict())

    base_optA_phi = torch.optim.AdamW(modelA_phi.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    base_optA_gal = torch.optim.AdamW(modelA_gal.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)

    phiA = PhiGaLoreWrapper(modelA_phi, base_optA_phi, r=32, T=50, ema=0.9, large_only_threshold=0, mp_store_dtype=mp_dtype)
    galA = GaLoreWrapper(modelA_gal, base_optA_gal, r=32, T=20, large_only_threshold=0)

    def dataA():
        return gen_darcy_like_batch(B=8, Nx=Nx, Ny=Ny, device=device, noise=0.05)

    loss_fnA = lambda out, y: F.mse_loss(out, y)

    print("Training Model A (MLPField) Full-rank AdamW...")
    resA_full = train_one(modelA, base_optA_full, dataA, loss_fnA, mp_dtype, steps=steps, log_every=max(1, steps//5),
                          srank_param_name='head_vel.weight')

    print("Training Model A (MLPField) Phi-GaLore...")
    resA_phi = train_one(modelA_phi, phiA, dataA, loss_fnA, mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name='head_vel.weight')

    print("Training Model A (MLPField) GaLore (naive)...")
    resA_gal = train_one(modelA_gal, galA, dataA, loss_fnA, mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name='head_vel.weight')

    # Alignment check for Phi-GaLore (on largest param)
    print("Computing cosine alignment on Model A (Phi-GaLore) vs Full gradient...")
    target_paramsA = ['head_vel.weight']
    batchA = dataA()
    cosA = cos_sim_full_vs_projected(modelA_phi, loss_fnA, batchA, phiA, target_paramsA, mp_dtype)
    for k, v in cosA.items():
        print(f"  Phi-GaLore alignment {k}: {v:.4f}")

    # Model B: Tiny transformer regressor (molecular-like)
    L = 32
    modelB_full = TinyTransformerRegressor(seq_len=L, dim=128, depth=4, heads=4).to(device)
    modelB_phi = TinyTransformerRegressor(seq_len=L, dim=128, depth=4, heads=4).to(device)
    modelB_gal = TinyTransformerRegressor(seq_len=L, dim=128, depth=4, heads=4).to(device)
    modelB_phi.load_state_dict(modelB_full.state_dict())
    modelB_gal.load_state_dict(modelB_full.state_dict())

    base_optB_full = torch.optim.AdamW(modelB_full.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    base_optB_phi = torch.optim.AdamW(modelB_phi.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    base_optB_gal = torch.optim.AdamW(modelB_gal.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    def share_by_B(name: str, p: nn.Parameter) -> Optional[str]:
        if 'blocks' in name and ('.attn_out.weight' in name or '.mlp_fc2.weight' in name):
            parts = name.split('.')
            for tok in parts:
                if tok.isdigit():
                    depth = int(tok)
                    bucket = depth // 2
                    return f'group.blocks.bucket{bucket}'
        return name

    phiB = PhiGaLoreWrapper(modelB_phi, base_optB_phi, r=32, T=50, ema=0.9, share_by=share_by_B, large_only_threshold=0, mp_store_dtype=mp_dtype)
    galB = GaLoreWrapper(modelB_gal, base_optB_gal, r=32, T=20, large_only_threshold=0)

    def dataB():
        return gen_molecular_like_batch(B=32, L=L, device=device, rot_noise=0.0, pos_noise=0.01)

    loss_fnB = lambda out, y: F.l1_loss(out, y)

    print("Training Model B (Transformer) Full-rank AdamW...")
    largestB_full = pick_largest_linear_param_names(modelB_full, top_k=1)[0]
    resB_full = train_one(modelB_full, base_optB_full, dataB, loss_fnB, mp_dtype, steps=steps, log_every=max(1, steps//5),
                          srank_param_name=largestB_full)

    print("Training Model B (Transformer) Phi-GaLore...")
    largestB_phi = pick_largest_linear_param_names(modelB_phi, top_k=1)[0]
    resB_phi = train_one(modelB_phi, phiB, dataB, loss_fnB, mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name=largestB_phi)

    print("Training Model B (Transformer) GaLore (naive)...")
    largestB_gal = pick_largest_linear_param_names(modelB_gal, top_k=1)[0]
    resB_gal = train_one(modelB_gal, galB, dataB, loss_fnB, mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name=largestB_gal)

    # Alignment on B
    batchB = dataB()
    cosB = cos_sim_full_vs_projected(modelB_phi, loss_fnB, batchB, phiB, [largestB_phi], mp_dtype)
    for k, v in cosB.items():
        print(f"  Phi-GaLore alignment (Transformer) {k}: {v:.4f}")

    # Plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_pdf_backend_and_style()

    plt.figure(figsize=(6, 4))
    plt.plot(resA_full.losses, label='MLP-AdamW')
    plt.plot(resA_phi.losses, label='MLP-PhiGaLore')
    plt.plot(resA_gal.losses, label='MLP-GaLore')
    plt.plot(resB_full.losses, label='Trans-AdamW', linestyle='--')
    plt.plot(resB_phi.losses, label='Trans-PhiGaLore', linestyle='--')
    plt.plot(resB_gal.losses, label='Trans-GaLore', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss (Micro-bench)')
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'training_loss_microbench.pdf'))
    plt.close()

    plt.figure(figsize=(6, 4))
    if len(resA_phi.sranks) > 0:
        plt.plot(resA_phi.sranks, label='MLP-PhiGaLore srank')
    if len(resA_gal.sranks) > 0:
        plt.plot(resA_gal.sranks, label='MLP-GaLore srank')
    if len(resB_phi.sranks) > 0:
        plt.plot(resB_phi.sranks, label='Trans-PhiGaLore srank')
    if len(resB_gal.sranks) > 0:
        plt.plot(resB_gal.sranks, label='Trans-GaLore srank')
    plt.xlabel('Recorded step index')
    plt.ylabel('Stable rank')
    plt.legend()
    plt.title('Gradient Stable Rank')
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'gradient_srank_microbench.pdf'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(resA_phi.overhead_ms, label='MLP-PhiGaLore')
    plt.plot(resA_gal.overhead_ms, label='MLP-GaLore')
    plt.plot(resB_phi.overhead_ms, label='Trans-PhiGaLore', linestyle='--')
    plt.plot(resB_gal.overhead_ms, label='Trans-GaLore', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Projection overhead (ms)')
    plt.legend()
    plt.title('Projection Overhead')
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'projection_overhead_microbench.pdf'))
    plt.close()

    names, vals = [], []
    for k, v in cosA.items():
        names.append(f"MLP:{k}")
        vals.append(v)
    for k, v in cosB.items():
        names.append(f"Trans:{k}")
        vals.append(v)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=names, y=vals)
    plt.ylim(0, 1.0)
    plt.ylabel('Cosine similarity')
    plt.title('Phi-GaLore vs Full Gradient Alignment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'alignment_phi-galore_microbench.pdf'))
    plt.close()

    print("Experiment 1 complete. Plots saved:")
    print("  - training_loss_microbench.pdf")
    print("  - gradient_srank_microbench.pdf")
    print("  - projection_overhead_microbench.pdf")
    print("  - alignment_phi-galore_microbench.pdf")


def experiment_2_physics_aware(images_dir: str,
                                Nx: int,
                                Ny: int,
                                steps: int,
                                device: torch.device,
                                mp_dtype: torch.dtype):
    print("\n=== Experiment 2: Physics-Aware Subspace Improves PDE Fidelity (Toy Navier–Stokes) ===")
    set_seed(123)
    ensure_pdf_backend_and_style()

    BATCH = 8
    def data_clean():
        return gen_burgers_bc_variants(B=BATCH, Nx=Nx, Ny=Ny, device=device, bc_noise=0.0, jitter=0.0)
    def data_bc_noise():
        return gen_burgers_bc_variants(B=BATCH, Nx=Nx, Ny=Ny, device=device, bc_noise=0.05, jitter=0.0)
    def data_jitter():
        return gen_burgers_bc_variants(B=BATCH, Nx=Nx, Ny=Ny, device=device, bc_noise=0.0, jitter=0.05)

    # Model: global MLP field with large final layer
    model_full = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)
    model_phi_noB = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)
    model_phi_B = MLPField(Nx=Nx, Ny=Ny, in_ch=1, hidden=128, depth=3).to(device)

    model_phi_noB.load_state_dict(model_full.state_dict())
    model_phi_B.load_state_dict(model_full.state_dict())

    opt_full = torch.optim.AdamW(model_full.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    opt_phi_noB = torch.optim.AdamW(model_phi_noB.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    opt_phi_B = torch.optim.AdamW(model_phi_B.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)

    # Symmetry basis for head_vel rows (2*Nx*Ny)
    B_out = build_divfree_basis_fourier(Nx, Ny, K=16, device=device)
    symreg = { 'head_vel.weight': B_out }

    phi_noB = PhiGaLoreWrapper(model_phi_noB, opt_phi_noB, r=32, T=50, ema=0.9, symmetry_registry={}, large_only_threshold=0, mp_store_dtype=mp_dtype)
    phi_B = PhiGaLoreWrapper(model_phi_B, opt_phi_B, r=32, T=50, ema=0.9, symmetry_registry=symreg, large_only_threshold=0, mp_store_dtype=mp_dtype)

    print("Training Full-rank on clean data ...")
    res_full = train_one(model_full, opt_full, data_clean,
                         lambda out, y: physics_loss(out, y), mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name='head_vel.weight')

    print("Training Phi-GaLore (no B) on clean data ...")
    res_phi_noB = train_one(model_phi_noB, phi_noB, data_clean,
                            lambda out, y: physics_loss(out, y), mp_dtype, steps=steps, log_every=max(1, steps//5),
                            srank_param_name='head_vel.weight')

    print("Training Phi-GaLore (+B) on clean data ...")
    res_phi_B = train_one(model_phi_B, phi_B, data_clean,
                          lambda out, y: physics_loss(out, y), mp_dtype, steps=steps, log_every=max(1, steps//5),
                          srank_param_name='head_vel.weight')

    # Robustness eval: rL2 and divergence norms on variants
    def evaluate_metrics(model: nn.Module, data_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]], n_batches: int = 4) -> Tuple[float, float]:
        model.eval()
        rL2s, divs = [], []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = data_fn()
                out = model(x)
                rL2s.append(relative_l2(out, y))
                div, _ = divergence_and_laplacian(out)
                divs.append(div.abs().mean().item())
        return float(np.mean(rL2s)), float(np.mean(divs))

    print("Evaluating robustness (rL2, divergence norm) on variants ...")
    metrics = {}
    for tag, mdl in [("Full", model_full), ("Phi", model_phi_noB), ("Phi+B", model_phi_B)]:
        m_clean = evaluate_metrics(mdl, data_clean)
        m_noise = evaluate_metrics(mdl, data_bc_noise)
        m_jitt  = evaluate_metrics(mdl, data_jitter)
        metrics[tag] = { 'clean': m_clean, 'bc_noise': m_noise, 'jitter': m_jitt }
        print(f"  {tag}: clean rL2={m_clean[0]:.4f}, div={m_clean[1]:.4e} | bc_noise rL2={m_noise[0]:.4f} | jitter rL2={m_jitt[0]:.4f}")

    # Plots
    import matplotlib.pyplot as plt
    ensure_pdf_backend_and_style()

    plt.figure(figsize=(6, 4))
    plt.plot(res_full.losses, label='Full-rank')
    plt.plot(res_phi_noB.losses, label='Phi-GaLore (no B)')
    plt.plot(res_phi_B.losses, label='Phi-GaLore (+B)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Navier–Stokes toy)')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'training_loss_navierstokes.pdf'))
    plt.close()

    conds = ['clean', 'bc_noise', 'jitter']
    methods = ['Full', 'Phi', 'Phi+B']
    vals = [[metrics[m][c][0] for c in conds] for m in methods]
    x = np.arange(len(conds))
    width = 0.25
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.bar(x + i*width, vals[i], width=width, label=m)
    plt.xticks(x + width, conds)
    plt.ylabel('rL2 (lower is better)')
    plt.title('Robustness across BC noise and jitter')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'rL2_navierstokes.pdf'))
    plt.close()

    dvals = [[metrics[m][c][1] for c in conds] for m in methods]
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods):
        plt.bar(x + i*width, dvals[i], width=width, label=m)
    plt.xticks(x + width, conds)
    plt.ylabel('Mean |div(u)|')
    plt.title('Divergence norm across conditions')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'divergence_norm_navierstokes.pdf'))
    plt.close()

    print("Experiment 2 complete. Plots saved:")
    print("  - training_loss_navierstokes.pdf")
    print("  - rL2_navierstokes.pdf")
    print("  - divergence_norm_navierstokes.pdf")


def experiment_3_hierarchical_sharing(images_dir: str,
                                      steps: int,
                                      L: int,
                                      device: torch.device,
                                      mp_dtype: torch.dtype):
    print("\n=== Experiment 3: Memory-Constrained Training with Hierarchical Sharing (Toy Molecular) ===")
    set_seed(7)
    ensure_pdf_backend_and_style()

    model_full = TinyTransformerRegressor(seq_len=L, dim=128, depth=6, heads=4).to(device)
    model_phi_share = TinyTransformerRegressor(seq_len=L, dim=128, depth=6, heads=4).to(device)
    model_phi_noshare = TinyTransformerRegressor(seq_len=L, dim=128, depth=6, heads=4).to(device)
    model_gal = TinyTransformerRegressor(seq_len=L, dim=128, depth=6, heads=4).to(device)

    model_phi_share.load_state_dict(model_full.state_dict())
    model_phi_noshare.load_state_dict(model_full.state_dict())
    model_gal.load_state_dict(model_full.state_dict())

    opt_full = torch.optim.AdamW(model_full.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    opt_phi_share = torch.optim.AdamW(model_phi_share.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    opt_phi_noshare = torch.optim.AdamW(model_phi_noshare.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    opt_gal = torch.optim.AdamW(model_gal.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    def share_by(name: str, p: nn.Parameter) -> Optional[str]:
        if 'blocks' in name and ('.attn_out.weight' in name or '.mlp_fc2.weight' in name):
            parts = name.split('.')
            # Expect pattern blocks.N.* so find integer token
            idx = None
            for t in parts:
                if t.isdigit():
                    idx = int(t)
                    break
            if idx is not None:
                bucket = idx // 3
                return f'group.blocks.bucket{bucket}'
        return name

    phi_share = PhiGaLoreWrapper(model_phi_share, opt_phi_share, r=32, T=50, ema=0.9, share_by=share_by, large_only_threshold=0, mp_store_dtype=mp_dtype)
    phi_noshare = PhiGaLoreWrapper(model_phi_noshare, opt_phi_noshare, r=32, T=50, ema=0.9, share_by=None, large_only_threshold=0, mp_store_dtype=mp_dtype)
    gal = GaLoreWrapper(model_gal, opt_gal, r=32, T=20, large_only_threshold=0)

    def data_train():
        return gen_molecular_like_batch(B=32, L=L, device=device, rot_noise=0.2, pos_noise=0.02)

    def data_eval_rot():
        return gen_molecular_like_batch(B=64, L=L, device=device, rot_noise=1.0, pos_noise=0.0)

    def data_eval_noise():
        return gen_molecular_like_batch(B=64, L=L, device=device, rot_noise=0.0, pos_noise=0.05)

    loss_fn = lambda out, y: F.l1_loss(out, y)

    print("Training Full-rank ...")
    largest_full = pick_largest_linear_param_names(model_full, top_k=1)[0]
    res_full = train_one(model_full, opt_full, data_train, loss_fn, mp_dtype, steps=steps, log_every=max(1, steps//5),
                         srank_param_name=largest_full)

    print("Training Phi-GaLore (sharing ON) ...")
    largest_share = pick_largest_linear_param_names(model_phi_share, top_k=1)[0]
    res_phi_share = train_one(model_phi_share, phi_share, data_train, loss_fn, mp_dtype, steps=steps, log_every=max(1, steps//5),
                              srank_param_name=largest_share)

    print("Training Phi-GaLore (sharing OFF) ...")
    largest_noshare = pick_largest_linear_param_names(model_phi_noshare, top_k=1)[0]
    res_phi_noshare = train_one(model_phi_noshare, phi_noshare, data_train, loss_fn, mp_dtype, steps=steps, log_every=max(1, steps//5),
                                srank_param_name=largest_noshare)

    print("Training GaLore ...")
    largest_gal = pick_largest_linear_param_names(model_gal, top_k=1)[0]
    res_gal = train_one(model_gal, gal, data_train, loss_fn, mp_dtype, steps=steps, log_every=max(1, steps//5),
                        srank_param_name=largest_gal)

    # Evaluation on robustness (MAE under rotations/noise)
    def eval_mae(model: nn.Module, data_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]], n_batches: int = 5) -> float:
        model.eval()
        maes = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = data_fn()
                pred = model(x)
                maes.append(F.l1_loss(pred, y).item())
        return float(np.mean(maes))

    mae_rot = {
        'Full': eval_mae(model_full, data_eval_rot),
        'Phi-share': eval_mae(model_phi_share, data_eval_rot),
        'Phi-noshare': eval_mae(model_phi_noshare, data_eval_rot),
        'GaLore': eval_mae(model_gal, data_eval_rot),
    }
    mae_noise = {
        'Full': eval_mae(model_full, data_eval_noise),
        'Phi-share': eval_mae(model_phi_share, data_eval_noise),
        'Phi-noshare': eval_mae(model_phi_noshare, data_eval_noise),
        'GaLore': eval_mae(model_gal, data_eval_noise),
    }
    print("MAE under rotations:")
    for k, v in mae_rot.items():
        print(f"  {k}: {v:.4f}")
    print("MAE under coord noise:")
    for k, v in mae_noise.items():
        print(f"  {k}: {v:.4f}")

    # Plots
    import matplotlib.pyplot as plt
    ensure_pdf_backend_and_style()

    plt.figure(figsize=(6, 4))
    plt.plot(res_full.losses, label='Full-rank')
    plt.plot(res_phi_share.losses, label='Phi-share')
    plt.plot(res_phi_noshare.losses, label='Phi-noshare')
    plt.plot(res_gal.losses, label='GaLore')
    plt.xlabel('Step')
    plt.ylabel('Loss (L1)')
    plt.title('Training Loss (Toy Molecular)')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'training_loss_molecular.pdf'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(res_phi_share.overhead_ms, label='Phi-share')
    plt.plot(res_phi_noshare.overhead_ms, label='Phi-noshare')
    plt.plot(res_gal.overhead_ms, label='GaLore')
    plt.xlabel('Step')
    plt.ylabel('Projection overhead (ms)')
    plt.title('Projection Overhead (Toy Molecular)')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'projection_overhead_molecular.pdf'))
    plt.close()

    names = list(mae_rot.keys())
    rvals = [mae_rot[n] for n in names]
    nvals = [mae_noise[n] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, rvals, width=width, label='Rotations')
    plt.bar(x + width/2, nvals, width=width, label='Coord noise')
    plt.xticks(x, names)
    plt.ylabel('MAE (lower is better)')
    plt.title('Robustness (Toy Molecular)')
    plt.legend()
    plt.tight_layout()
    save_plot(os.path.join(images_dir, 'mae_molecular.pdf'))
    plt.close()

    print("Experiment 3 complete. Plots saved:")
    print("  - training_loss_molecular.pdf")
    print("  - projection_overhead_molecular.pdf")
    print("  - mae_molecular.pdf")
