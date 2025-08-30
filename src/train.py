import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .preprocess import SynthVideoDataset, MockCLIP


# -----------------------------
# Config
# -----------------------------

@dataclass
class ABLoRAConfig:
    r: int = 4
    K_max: int = 16
    mdl_penalty: float = 1.0
    hidden_dim: int = 256
    freeze_basis_encoder_stage3: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# EM with MDL (diagonal GMM) over latent features
# -----------------------------

@torch.no_grad()
def em_learn_bases(Ftd: torch.Tensor, K_max: int, mdl_penalty: float = 1.0, iters: int = 5,
                   seed: int = 0) -> Tuple[torch.Tensor, int, Dict]:
    """
    EM in latent space (diagonal GMM) to compute responsibilities gamma and select K via MDL.
    Ftd: [T, D]
    Returns: gamma [T, K_sel], K_sel, aux
    """
    torch.manual_seed(seed)
    T, D = Ftd.shape
    device = Ftd.device
    # Initialize K heuristically based on std dev
    K = min(K_max, max(1, int((Ftd.std(dim=0).mean() * 10).item()) + 1))
    idxs = torch.linspace(0, T - 1, steps=K).long()
    means = Ftd[idxs].clone()
    cov = torch.ones(K, D, device=device) * Ftd.var(dim=0).mean().clamp_min(1e-3)
    pis = torch.ones(K, device=device) / K

    def e_step(Ftd, means, cov, pis):
        diff = Ftd.unsqueeze(1) - means.unsqueeze(0)  # [T,K,D]
        log_det = cov.log().sum(dim=-1)              # [K]
        maha = (diff.pow(2) / cov.unsqueeze(0)).sum(dim=-1)  # [T,K]
        log_prob = -0.5 * (maha + log_det.unsqueeze(0) + D * math.log(2 * math.pi))
        log_prob = log_prob + pis.log().unsqueeze(0)
        gamma = torch.softmax(log_prob, dim=1)
        return gamma

    def m_step(Ftd, gamma):
        Nk = gamma.sum(dim=0) + 1e-6
        means = (gamma.transpose(0, 1) @ Ftd) / Nk.unsqueeze(-1)
        diff = Ftd.unsqueeze(1) - means.unsqueeze(0)
        cov = (gamma.unsqueeze(-1) * diff.pow(2)).sum(dim=0) / Nk.unsqueeze(-1)
        cov = cov.clamp_min(1e-4)
        pis = Nk / Nk.sum()
        return means, cov, pis

    for _ in range(iters):
        gamma = e_step(Ftd, means, cov, pis)
        means, cov, pis = m_step(Ftd, gamma)

    def neg_log_likelihood(Ftd, means, cov, pis):
        diff = Ftd.unsqueeze(1) - means.unsqueeze(0)
        log_det = cov.log().sum(dim=-1)
        maha = (diff.pow(2) / cov.unsqueeze(0)).sum(dim=-1)
        log_prob = -0.5 * (maha + log_det.unsqueeze(0) + D * math.log(2 * math.pi))
        log_prob = log_prob + pis.log().unsqueeze(0)
        ll = torch.logsumexp(log_prob, dim=1).sum()
        return -ll

    # Greedy MDL/BIC-like pruning
    best = {
        'K': K, 'gamma': gamma, 'means': means, 'cov': cov, 'pis': pis,
        'nll': neg_log_likelihood(Ftd, means, cov, pis).item()
    }
    while K > 1:
        Nk = gamma.sum(dim=0)
        remove_idx = torch.argmin(Nk)
        keep = torch.ones(K, dtype=torch.bool, device=device)
        keep[remove_idx] = False
        means2, cov2, pis2 = means[keep], cov[keep], (pis[keep] / pis[keep].sum())
        gamma2 = e_step(Ftd, means2, cov2, pis2)
        nll2 = neg_log_likelihood(Ftd, means2, cov2, pis2).item()
        p_per_comp = D + D              # mean + diag cov
        bic2 = nll2 + mdl_penalty * 0.5 * (p_per_comp * means2.shape[0]) * math.log(T)
        bic_best = best['nll'] + mdl_penalty * 0.5 * (p_per_comp * best['K']) * math.log(T)
        if bic2 < bic_best:
            K = means2.shape[0]
            means, cov, pis, gamma = means2, cov2, pis2, gamma2
            best = {'K': K, 'gamma': gamma, 'means': means, 'cov': cov, 'pis': pis, 'nll': nll2}
        else:
            break

    gamma = best['gamma'][:, :best['K']]
    aux = {k: best[k] for k in ['means', 'cov', 'pis']}
    return gamma, best['K'], aux


# -----------------------------
# Low-rank directions and efficient AB-LoRA mixture
# -----------------------------

class LowRankDirectionParam(nn.Module):
    def __init__(self, d_model: int, r: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(r))

    def factors(self):
        U = F.normalize(self.U, dim=0)
        V = F.normalize(self.V, dim=0)
        S = self.alpha.tanh()  # [r]
        return U, V, S


class ABLoRAEff(nn.Module):
    """
    Efficient AB-LoRA that applies a mixture of low-rank updates without materializing DxD matrices.
    y = x + sum_k w_tk * (x @ V_k @ diag(S_k) @ U_k^T), with w_tk = gamma_tk * m_k
    """
    def __init__(self, d_model: int, r: int, K_max: int):
        super().__init__()
        self.K_max = K_max
        self.d_model = d_model
        self.r = r
        self.magnitudes = nn.Parameter(torch.zeros(K_max))
        self.directions = nn.ModuleList([LowRankDirectionParam(d_model, r) for _ in range(K_max)])
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        # x: [B,T,D], gamma: [T, K_sel]
        B, T, D = x.shape
        temp = torch.exp(self.log_temp).clamp_min(0.1)
        g = F.softmax(gamma / temp, dim=-1)  # [T,K_sel]
        if g.shape[1] < self.K_max:
            pad = torch.zeros(T, self.K_max - g.shape[1], device=x.device, dtype=x.dtype)
            g = torch.cat([g, pad], dim=1)  # [T,K_max]
        m = self.magnitudes.tanh()  # [Kmax]
        w = g * m.unsqueeze(0)       # [T,Kmax]
        # Stack factors
        U = torch.stack([d.factors()[0] for d in self.directions], dim=0)  # [K,D,r]
        V = torch.stack([d.factors()[1] for d in self.directions], dim=0)  # [K,D,r]
        S = torch.stack([d.factors()[2] for d in self.directions], dim=0)  # [K,r]
        # Compute x @ V_k for all k: Z = einsum('btd,kdr->btkr')
        Z = torch.einsum('btd,kdr->btkr', x, V)  # [B,T,K,r]
        # scale by S and weights w_tk
        Z = Z * S.unsqueeze(0).unsqueeze(0)      # [B,T,K,r]
        Z = Z * w.unsqueeze(0).unsqueeze(-1)     # [B,T,K,r]
        # Project via U: y_delta = sum_k Z_k @ U_k^T => einsum('btkr,krd->btd', sum over k,r)
        y_delta = torch.einsum('btkr,krd->btd', Z, U.transpose(1, 2))
        return x + y_delta


# -----------------------------
# Baseline adapters (LoRA, DoRA, SimDA-equivalent)
# -----------------------------

class LoRAAdapter(nn.Module):
    def __init__(self, d_model: int, r: int = 8):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.alpha = nn.Parameter(torch.ones(r))

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        U = F.normalize(self.U, dim=0)
        V = F.normalize(self.V, dim=0)
        S = self.alpha
        Z = torch.einsum('btd,dr->btr', x, V)      # [B,T,r]
        Z = Z * S
        y_delta = torch.einsum('btr,rd->btd', Z, U.t())
        return x + y_delta


class DoRAAdapter(nn.Module):
    def __init__(self, d_model: int, r: int = 64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.magnitude = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.ones(r))

    def forward(self, x: torch.Tensor):
        U = F.normalize(self.U, dim=0)
        V = F.normalize(self.V, dim=0)
        S = self.alpha
        mag = self.magnitude.tanh()
        Z = torch.einsum('btd,dr->btr', x, V)  # [B,T,r]
        Z = Z * S * mag
        y_delta = torch.einsum('btr,rd->btd', Z, U.t())
        return x + y_delta


class SimDAAdapter(nn.Module):
    """Equivalent to AB-LoRA with K=1 and r=4."""
    def __init__(self, d_model: int, r: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, r) * 0.02)

    def forward(self, x: torch.Tensor):
        U = F.normalize(self.U, dim=0)
        V = F.normalize(self.V, dim=0)
        Z = torch.einsum('btd,dr->btr', x, V)
        y_delta = torch.einsum('btr,rd->btd', Z, U.t())
        return x + y_delta


# -----------------------------
# Temporal Basis Encoder (predicts per-timestep basis logits)
# -----------------------------

class TemporalBasisEncoder(nn.Module):
    def __init__(self, d: int, K_max: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d // 2, 3, padding=1)
        self.conv2 = nn.Conv1d(d // 2, d // 4, 3, padding=1)
        self.proj = nn.Conv1d(d // 4, K_max, 1)
        self.log_decay = nn.Parameter(torch.tensor(-1.0))
        self.K_max = K_max

    def forward(self, Ftd: torch.Tensor):
        # Ftd: [T, D]
        x = Ftd.unsqueeze(0).transpose(1, 2)  # [1,D,T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.proj(x).squeeze(0).transpose(0, 1)  # [T,K_max]
        # EMA smoothing
        decay = torch.sigmoid(self.log_decay)
        ema = torch.zeros_like(logits)
        for t in range(logits.shape[0]):
            if t == 0:
                ema[t] = logits[t]
            else:
                ema[t] = decay * ema[t - 1] + (1 - decay) * logits[t]
        return ema


# -----------------------------
# Mock attention block (identity + adapter injection)
# -----------------------------

class MockAttentionBlock(nn.Module):
    def __init__(self, d_model: int, adapter: Optional[nn.Module] = None):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.adapter = adapter

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None):
        # x: [B,T,D]
        out = self.proj_out(F.relu(self.proj_in(x)))
        if isinstance(self.adapter, ABLoRAEff):
            assert gamma is not None, "AB-LoRA requires gamma responsibilities"
            out = self.adapter(out, gamma)
        elif self.adapter is not None:
            out = self.adapter(out)
        return out


# -----------------------------
# Heads for QA (classification) and temporal grounding (regression over time)
# -----------------------------

class QAHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, y: torch.Tensor):
        # y: [B,T,D]; mean pool over time
        pooled = y.mean(dim=1)
        return self.mlp(pooled)  # logits [B,C]


class GroundingHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, y: torch.Tensor):
        # y: [B,T,D]
        scores = self.scorer(y).squeeze(-1)  # [B,T]
        return scores


# -----------------------------
# Training routines
# -----------------------------

def stage2_basis_pretrain(basis_enc: TemporalBasisEncoder, clip_model: MockCLIP,
                          loader: DataLoader, cfg: ABLoRAConfig, steps: int = 100) -> List[float]:
    basis_enc.train()
    opt = torch.optim.AdamW(basis_enc.parameters(), lr=1e-3)
    losses = []
    it = 0
    for batch in loader:
        frames = batch['frames'][0].numpy()
        imgs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float().to(cfg.device) / 255.0
        with torch.no_grad():
            Ftd = clip_model(imgs).detach()  # [T,D]
        # Teacher responsibilities via EM + MDL
        gamma, K_sel, _ = em_learn_bases(Ftd, cfg.K_max, cfg.mdl_penalty)
        logits = basis_enc(Ftd)
        pred = F.log_softmax(logits[:, :K_sel], dim=-1)
        target = F.softmax(gamma, dim=-1)
        loss = -(target * pred).sum(dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        it += 1
        if it >= steps:
            break
    print(f"Stage-2 basis pretraining completed: {len(losses)} steps; final loss={losses[-1]:.4f}")
    return losses


def stage3_train_adapter(attn_block: MockAttentionBlock, basis_enc: TemporalBasisEncoder,
                         clip_model: MockCLIP, qa_head: QAHead, grd_head: GroundingHead,
                         loader: DataLoader, cfg: ABLoRAConfig, steps: int = 200,
                         use_mdl_gamma: bool = True) -> Tuple[List[float], List[float]]:
    # Freeze basis encoder if requested
    for p in basis_enc.parameters():
        p.requires_grad = not cfg.freeze_basis_encoder_stage3
    params = list(attn_block.parameters()) + list(qa_head.parameters()) + list(grd_head.parameters())
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=5e-4)
    ce_losses, total_losses = [], []
    step = 0
    attn_block.train(); qa_head.train(); grd_head.train()
    for batch in loader:
        frames = batch['frames'][0].numpy()
        qa_cls = int(batch['qa_cls'][0].item()) if isinstance(batch['qa_cls'], torch.Tensor) else int(batch['qa_cls'][0])
        first_t = int(batch['first_change_t'][0].item()) if isinstance(batch['first_change_t'], torch.Tensor) else int(batch['first_change_t'][0])
        T = int(batch['length'][0].item()) if isinstance(batch['length'], torch.Tensor) else int(batch['length'][0])
        imgs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float().to(cfg.device) / 255.0
        with torch.no_grad():
            Ftd = clip_model(imgs).detach()  # [T,D]
        gamma = None
        if isinstance(attn_block.adapter, ABLoRAEff):
            if use_mdl_gamma:
                gamma, K_sel, _ = em_learn_bases(Ftd, cfg.K_max, cfg.mdl_penalty)
            else:
                K_fixed = min(8, cfg.K_max)
                gamma = torch.ones(T, K_fixed, device=cfg.device) / K_fixed
        x = Ftd.unsqueeze(0)  # [1,T,D]
        y = attn_block(x, gamma)  # [1,T,D]
        # Heads
        qa_logits = qa_head(y)
        grd_scores = grd_head(y)
        # Losses: CE for QA; soft target for grounding (use KL over time)
        qa_target = torch.tensor([qa_cls], device=cfg.device)
        ce_loss = F.cross_entropy(qa_logits, qa_target)
        grd_target = torch.zeros(1, T, device=cfg.device)
        grd_target[0, min(first_t, T - 1)] = 1.0
        grd_ce = F.kl_div(F.log_softmax(grd_scores, dim=-1), grd_target, reduction='batchmean')
        # Smoothness regularizer
        smooth = (y[:, 1:] - y[:, :-1]).pow(2).mean()
        loss = ce_loss + 0.5 * grd_ce + 0.1 * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        ce_losses.append(float(ce_loss.item()))
        total_losses.append(float(loss.item()))
        step += 1
        if step >= steps:
            break
    print(f"Stage-3 training completed: {len(total_losses)} steps; final loss={total_losses[-1]:.4f}")
    return ce_losses, total_losses


# -----------------------------
# Baseline training helper
# -----------------------------

def train_baseline(attn_block: MockAttentionBlock, qa_head: QAHead, grd_head: GroundingHead,
                   clip_model: MockCLIP, loader: DataLoader, device: str, steps: int = 120) -> List[float]:
    params = list(attn_block.parameters()) + list(qa_head.parameters()) + list(grd_head.parameters())
    opt = torch.optim.AdamW(params, lr=5e-4)
    losses = []
    it = 0
    attn_block.train(); qa_head.train(); grd_head.train()
    for batch in loader:
        frames = batch['frames'][0].numpy()
        qa_cls = int(batch['qa_cls'][0].item()) if isinstance(batch['qa_cls'], torch.Tensor) else int(batch['qa_cls'][0])
        first_t = int(batch['first_change_t'][0].item()) if isinstance(batch['first_change_t'], torch.Tensor) else int(batch['first_change_t'][0])
        T = int(batch['length'][0].item()) if isinstance(batch['length'], torch.Tensor) else int(batch['length'][0])
        imgs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float().to(device) / 255.0
        with torch.no_grad():
            Ftd = clip_model(imgs).detach()
        y = attn_block(Ftd.unsqueeze(0))
        qa_logits = qa_head(y)
        grd_scores = grd_head(y)
        qa_target = torch.tensor([qa_cls], device=device)
        ce = F.cross_entropy(qa_logits, qa_target)
        grd_target = torch.zeros(1, T, device=device); grd_target[0, min(first_t, T - 1)] = 1.0
        grd_ce = F.kl_div(F.log_softmax(grd_scores, dim=-1), grd_target, reduction='batchmean')
        smooth = (y[:, 1:] - y[:, :-1]).pow(2).mean()
        loss = ce + 0.5 * grd_ce + 0.1 * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        it += 1
        if it >= steps:
            break
    return losses


# -----------------------------
# Utility helpers
# -----------------------------

def pad_or_trim(arr: List[float], L: int) -> List[float]:
    a = list(arr)
    if len(a) >= L:
        return a[:L]
    else:
        return a + [a[-1] if a else 0.0] * (L - len(a))


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        return 0.0


# -----------------------------
# Builder for all models used in experiments
# -----------------------------

def build_models(cfg: ABLoRAConfig):
    clip_model = MockCLIP(d=cfg.hidden_dim).to(cfg.device).eval()
    basis_enc = TemporalBasisEncoder(d=cfg.hidden_dim, K_max=cfg.K_max).to(cfg.device)

    # AB-LoRA
    ab_adapter = ABLoRAEff(d_model=cfg.hidden_dim, r=cfg.r, K_max=cfg.K_max).to(cfg.device)
    attn_ab = MockAttentionBlock(d_model=cfg.hidden_dim, adapter=ab_adapter).to(cfg.device)
    qa_head_ab = QAHead(cfg.hidden_dim, n_classes=4).to(cfg.device)
    grd_head_ab = GroundingHead(cfg.hidden_dim).to(cfg.device)

    # Baselines
    lora_adapter = LoRAAdapter(d_model=cfg.hidden_dim, r=8).to(cfg.device)
    attn_lora = MockAttentionBlock(d_model=cfg.hidden_dim, adapter=lora_adapter).to(cfg.device)
    qa_head_lora = QAHead(cfg.hidden_dim, n_classes=4).to(cfg.device)
    grd_head_lora = GroundingHead(cfg.hidden_dim).to(cfg.device)

    dora_adapter = DoRAAdapter(d_model=cfg.hidden_dim, r=64).to(cfg.device)
    attn_dora = MockAttentionBlock(d_model=cfg.hidden_dim, adapter=dora_adapter).to(cfg.device)
    qa_head_dora = QAHead(cfg.hidden_dim, n_classes=4).to(cfg.device)
    grd_head_dora = GroundingHead(cfg.hidden_dim).to(cfg.device)

    simda_adapter = SimDAAdapter(d_model=cfg.hidden_dim, r=cfg.r).to(cfg.device)
    attn_simda = MockAttentionBlock(d_model=cfg.hidden_dim, adapter=simda_adapter).to(cfg.device)
    qa_head_simda = QAHead(cfg.hidden_dim, n_classes=4).to(cfg.device)
    grd_head_simda = GroundingHead(cfg.hidden_dim).to(cfg.device)

    return {
        'clip': clip_model,
        'basis_enc': basis_enc,
        'attn_ab': attn_ab, 'qa_ab': qa_head_ab, 'grd_ab': grd_head_ab,
        'attn_lora': attn_lora, 'qa_lora': qa_head_lora, 'grd_lora': grd_head_lora,
        'attn_dora': attn_dora, 'qa_dora': qa_head_dora, 'grd_dora': grd_head_dora,
        'attn_simda': attn_simda, 'qa_simda': qa_head_simda, 'grd_simda': grd_head_simda,
    }
