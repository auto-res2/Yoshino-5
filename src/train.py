import os
import math
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: All plotting and PDF saving utilities live in evaluate.py to keep roles clean.

# ------------------------------ Quantization emulation utils ------------------------------

def quantize_dequant_4bit_dense(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """
    Emulate group-wise 4-bit quantization via linear quantizer (per-group min/max) and return dequantized tensor.
    Returns a tensor in the SAME dtype as input x (important for CPU compatibility).
    """
    orig = x.shape
    L = orig[-1]
    num_groups = math.ceil(L / group_size)
    pads = num_groups * group_size - L
    if pads > 0:
        xpad = F.pad(x, (0, pads))
    else:
        xpad = x
    xview = xpad.view(-1, num_groups, group_size)
    xmin = xview.amin(dim=-1, keepdim=True)
    xmax = xview.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin) / 15.0 + 1e-8
    zero = xmin
    q = torch.clamp(torch.round((xview - zero) / scale), 0, 15)
    xhat = q * scale + zero
    xhat = xhat.view(*orig[:-1], num_groups * group_size)
    if pads > 0:
        xhat = xhat[..., :L]
    return xhat.to(x.dtype)


def quantize_int8_groupwise_dense(x: torch.Tensor, group_size: int = 64) -> Tuple[torch.IntTensor, torch.Tensor]:
    """
    Per-group symmetric int8 quantization returning:
    - q: int8 tensor same shape as x
    - scale_map: per-element expanded scale tensor (float32)
    """
    orig = x.shape
    L = orig[-1]
    num_groups = math.ceil(L / group_size)
    pads = num_groups * group_size - L
    if pads > 0:
        xpad = F.pad(x, (0, pads))
    else:
        xpad = x
    xview = xpad.view(-1, num_groups, group_size)
    scale = xview.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-8
    q = torch.round(xview / scale).clamp_(-127, 127).to(torch.int8)
    q = q.view(*orig[:-1], num_groups * group_size)
    scale_map = scale.view(*orig[:-1], num_groups * group_size)
    if pads > 0:
        q = q[..., :L]
        scale_map = scale_map[..., :L]
    return q.contiguous(), scale_map.contiguous()


# ------------------------------ Gate registry and context manager ------------------------------

class _GateRegistry:
    def __init__(self):
        self.active = False
        self.gates = None  # List[Dict[str, BoolTensor[chunks]]]

_gate_registry = _GateRegistry()


class hiqua_gating:
    """
    Context manager to activate per-layer gates. Gates is a list of dicts (one per layer), mapping sub_name -> BoolTensor[1, chunks].
    """
    def __init__(self, gates: List[Dict[str, torch.Tensor]]):
        self.gates = gates
    def __enter__(self):
        _gate_registry.active = True
        _gate_registry.gates = self.gates
    def __exit__(self, exc_type, exc_val, exc_tb):
        _gate_registry.active = False
        _gate_registry.gates = None


# ------------------------------ Residual-injecting Linear wrapper ------------------------------

class ResidualInjectLinear(nn.Module):
    """
    Emulates HiQuA dual-space storage:
    - Base: per-group 4-bit dequantized weight W4 (dense, same dtype as original weight)
    - Residual: per-group int8 residual with per-element scale map (dense)
    - Gating: per-chunk promotion along output features to inject residual into selected chunks

    Expects weight_inout as shape [in_features, out_features]. Forward expects x shape [B, T, in_features].
    """
    def __init__(self, weight_inout: torch.Tensor, bias: Optional[torch.Tensor], layer_idx: int, sub_name: str, group_size: int = 64, chunks: int = 8):
        super().__init__()
        in_f, out_f = int(weight_inout.shape[0]), int(weight_inout.shape[1])
        self.in_f, self.out_f = in_f, out_f
        self.layer_idx = layer_idx
        self.sub_name = sub_name
        self.group_size = group_size
        self.chunks = chunks
        base_dtype = weight_inout.dtype
        with torch.no_grad():
            # Base 4-bit emulation: quantize-dequant on W^T along input dim, keep same dtype as input for CPU compatibility
            W4T = quantize_dequant_4bit_dense(weight_inout.t().contiguous(), group_size=group_size)  # [out, in]
            self.register_buffer('W4', W4T.t().contiguous().to(base_dtype))  # [in, out]
            # Residual int8 per-group with scale map, stored as [out, in] for easier chunking by out
            residual = (weight_inout - self.W4.to(weight_inout.dtype)).t().contiguous()  # [out, in]
            q8, s_map = quantize_int8_groupwise_dense(residual, group_size=group_size)
            self.register_buffer('R8_q', q8)
            self.register_buffer('R8_smap', s_map)
        self.bias = nn.Parameter(bias.to(base_dtype)) if bias is not None else None
        # Chunking along out_features (columns)
        rows_per_chunk = out_f // chunks
        self.chunk_slices = []
        start = 0
        for i in range(chunks):
            end = out_f if i == chunks - 1 else start + rows_per_chunk
            self.chunk_slices.append(slice(start, end))
            start = end
        self.enable_residual = True

    def _current_gate(self, device: torch.device) -> Optional[torch.Tensor]:
        if not _gate_registry.active or _gate_registry.gates is None:
            return None
        gdict = _gate_registry.gates[self.layer_idx]
        g = gdict.get(self.sub_name, None)
        if g is None:
            return None
        if g.dim() > 1:
            reduce_dims = tuple(range(g.dim() - 1))
            g = g.any(dim=reduce_dims)
        return g.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,in]
        B, T, Din = x.shape
        x2 = x.reshape(B * T, Din)
        # Cast x to W4 dtype if necessary (CUDA kernels support fp16/bf16; CPU may be fp32)
        if x2.dtype != self.W4.dtype:
            x2_mm = x2.to(self.W4.dtype)
        else:
            x2_mm = x2
        base = x2_mm @ self.W4  # [B*T, out]
        gate = self._current_gate(x.device)
        if self.enable_residual and (gate is not None):
            for i, sl in enumerate(self.chunk_slices):
                if bool(gate[i].item()):
                    # Recompose W_rec[:, sl] = W4[:, sl] + R8[:, :,]*scale --> Note: R8 stored as [out, in]
                    R8_chunk = (self.R8_q[sl, :].float() * self.R8_smap[sl, :].float())  # [len(sl), in]
                    Wrec = (self.W4[:, sl].to(torch.float32) + R8_chunk.t().contiguous()).to(self.W4.dtype)  # [in, len(sl)]
                    base[:, sl] = x2_mm @ Wrec
        out = base.view(B, T, -1)
        if self.bias is not None:
            out = out + self.bias
        # Match original x dtype on output to avoid dtype drift in rest of the model
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out


# ------------------------------ Model patcher for Llama/Mistral-like models ------------------------------

@dataclass
class WrapSpec:
    attr_path: str  # dotted path to the module
    sub_name: str   # {'q','k','v','o','up','gate','down'}

DEFAULT_WRAP = [
    WrapSpec('self_attn.q_proj', 'q'),
    WrapSpec('self_attn.k_proj', 'k'),
    WrapSpec('self_attn.v_proj', 'v'),
    WrapSpec('self_attn.o_proj', 'o'),
    WrapSpec('mlp.up_proj', 'up'),
    WrapSpec('mlp.gate_proj', 'gate'),
    WrapSpec('mlp.down_proj', 'down'),
]


def _get_by_path(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for p in path.split('.'):
        cur = getattr(cur, p)
    return cur


def _set_by_path(root: nn.Module, path: str, value: nn.Module):
    parts = path.split('.')
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def patch_model_with_hiqua(hf_model: nn.Module, chunks: int = 8, group_size: int = 64, wrap_specs: List[WrapSpec] = DEFAULT_WRAP) -> nn.Module:
    """
    Replace selected Linear layers with ResidualInjectLinear. Assumes Llama/Mistral-like .model.layers structure.
    """
    layers = hf_model.model.layers
    for li, layer in enumerate(layers):
        for spec in wrap_specs:
            mod: nn.Linear = _get_by_path(layer, spec.attr_path)
            # PyTorch Linear weight is [out, in] — convert to [in, out]
            W_inout = mod.weight.data.t().contiguous()
            b = mod.bias.data if mod.bias is not None else None
            wrapped = ResidualInjectLinear(W_inout, b, layer_idx=li, sub_name=spec.sub_name, group_size=group_size, chunks=chunks)
            _set_by_path(layer, spec.attr_path, wrapped)
    return hf_model


def set_residual_enabled(model: nn.Module, enabled: bool):
    for _, m in model.named_modules():
        if isinstance(m, ResidualInjectLinear):
            m.enable_residual = enabled


# ------------------------------ Tiny predictor and budget bandit ------------------------------

class HiQuAPredictor(nn.Module):
    def __init__(self, hidden_size: int, chunks: int = 8, depth: int = 2, d_model: int = 128, k_ctx: int = 8):
        super().__init__()
        self.k_ctx = k_ctx
        self.chunks = chunks
        self.proj = nn.Linear(hidden_size, d_model)
        blocks = []
        for _ in range(depth):
            blocks += [nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)]
        self.encoder = nn.Sequential(*blocks)
        self.scorer = nn.Linear(d_model, 7 * chunks)
        self.tau = nn.Parameter(torch.tensor(0.0))
    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        x = last_hidden[:, -self.k_ctx:, :]
        x = self.proj(x)
        x = self.encoder(x).mean(dim=1)
        return self.scorer(x)


class BudgetBandit:
    def __init__(self, target_ratio: float = 0.15, lr: float = 0.2):
        self.target = target_ratio
        self.lr = lr
        self.ema_ratio = 0.0
    def update(self, current_ratio: float, predictor: HiQuAPredictor):
        self.ema_ratio = 0.9 * self.ema_ratio + 0.1 * current_ratio
        delta = (self.ema_ratio - self.target)
        with torch.no_grad():
            predictor.tau.add_(-self.lr * delta)


def scores_to_gates(scores: torch.Tensor, predictor: HiQuAPredictor) -> Tuple[Dict[str, torch.Tensor], float]:
    tau = predictor.tau
    gates = (scores > tau).bool()
    ratio = float(gates.float().mean().item())
    names = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
    chunks = predictor.chunks
    gsplit = {name: gates[:, i * chunks:(i + 1) * chunks] for i, name in enumerate(names)}
    return gsplit, ratio


# ------------------------------ Warm-up and stepwise decoding ------------------------------

@torch.no_grad()
def warmup_train_predictor(model, predictor, data_loader, steps: int = 200, lr: float = 5e-4, device: str = 'cuda') -> List[float]:
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr)
    predictor.train()
    losses = []
    names = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
    L = len(model.model.layers)
    for step_i, batch in enumerate(data_loader):
        if step_i >= steps:
            break
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        # Teacher: all promoted
        all_true = {k: torch.ones(1, predictor.chunks, dtype=torch.bool, device=device) for k in names}
        with hiqua_gating([all_true for _ in range(L)]):
            teacher = model(input_ids=input_ids, attention_mask=attn_mask)
        # Student: predictor-gated
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        scores = predictor(last_hidden)
        gates_dict, _ = scores_to_gates(scores, predictor)
        with hiqua_gating([gates_dict for _ in range(L)]):
            student = model(input_ids=input_ids, attention_mask=attn_mask)
        pt = F.log_softmax(teacher.logits[:, -1, :].detach(), dim=-1)
        ps = F.log_softmax(student.logits[:, -1, :], dim=-1)
        loss = F.kl_div(ps, pt, log_target=True, reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (step_i + 1) % 20 == 0:
            print(f"[warmup] step={step_i+1} kl_loss={np.mean(losses[-20:]):.4f}")
    predictor.eval()
    return losses


@torch.no_grad()
def step_forward(model, predictor, input_ids, attention_mask, past_key_values=None, bandit: Optional[BudgetBandit] = None):
    # Base pass to obtain hidden states for gating
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
    last_hidden = outputs.hidden_states[-1]
    scores = predictor(last_hidden)
    gates_dict, ratio = scores_to_gates(scores, predictor)
    if bandit is not None:
        bandit.update(ratio, predictor)
    # Gated pass
    L = len(model.model.layers)
    gates = [gates_dict for _ in range(L)]
    with hiqua_gating(gates):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
    return out, ratio
