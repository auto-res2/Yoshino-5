# -*- coding: utf-8 -*-
"""
Training module for FedU-Align synthetic experiments.
Contains model definitions (Spectral-ReFT PEFT), federated client logic,
optimal transport anchors, conformal aggregator, and training orchestration utilities.

All heavy plotting and dataset construction live in evaluate.py and preprocess.py respectively.
Use relative imports only.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Support both package and script execution
try:
    from .preprocess import SyntheticMultimodalDataset, ClientConfig
except ImportError:  # pragma: no cover
    from preprocess import SyntheticMultimodalDataset, ClientConfig


# -----------------------------
# Utilities
# -----------------------------

def softmax_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = torch.tensor(0.0)
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_size = mask.sum()
            if bin_size.item() > 0:
                acc_bin = accuracies[mask].float().mean()
                conf_bin = confidences[mask].mean()
                ece += (bin_size.float() / labels.numel()) * (conf_bin - acc_bin).abs()
        return float(ece.item())


def benjamini_hochberg(pvals: List[float], q: float = 0.1) -> List[bool]:
    m = len(pvals)
    if m == 0:
        return []
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = sorted_p <= thresh
    if not passed.any():
        return [False] * m
    k_max = np.where(passed)[0].max()
    cutoff = sorted_p[k_max]
    return [p <= cutoff for p in pvals]


def estimate_payload_mb(arrays: List[torch.Tensor], dtype_bytes: int = 4) -> float:
    total_elems = 0
    for a in arrays:
        total_elems += int(np.prod(a.shape))
    return total_elems * dtype_bytes / (1024.0 * 1024.0)


# -----------------------------
# PEFT Adapters and Fusion
# -----------------------------

class SpectralAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, scale_init: float = 1e-3, p_drop: float = 0.0):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.s = nn.Parameter(torch.ones(rank) * scale_init)
        self.dropout = nn.Dropout(p=p_drop)

    @torch.no_grad()
    def _orthonormalize_(self, A: torch.Tensor):
        q, _ = torch.linalg.qr(A, mode='reduced')
        A.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        self._orthonormalize_(self.U)
        self._orthonormalize_(self.V)
        deltaW = (self.U * self.s) @ self.V.t()  # [out_dim, in_dim]
        out = F.linear(x, deltaW)
        return self.dropout(out)


class LoRAAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0, p_drop: float = 0.0):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deltaW = self.A @ self.B  # [out_dim, in_dim]
        out = F.linear(x, deltaW * self.scaling)
        return self.dropout(out)


class ReFTIntervention(nn.Module):
    def __init__(self, dim: int, adapter_type: str = 'spectral', rank: int = 8):
        super().__init__()
        if adapter_type == 'spectral':
            self.adapter = SpectralAdapter(dim, dim, rank=rank)
        elif adapter_type == 'lora':
            self.adapter = LoRAAdapter(dim, dim, rank=rank)
        else:
            raise ValueError("Unknown adapter_type for ReFT: %s" % adapter_type)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.alpha * self.adapter(h)


class ModalityEncoder(nn.Module):
    def __init__(self, token_dim: int, embed_dim: int, adapter_type: str = 'spectral', rank: int = 8):
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.backbone_proj = nn.Linear(token_dim, embed_dim, bias=False)
        for p in self.backbone_proj.parameters():
            p.requires_grad = False
        if adapter_type == 'spectral':
            self.adapter = SpectralAdapter(token_dim, embed_dim, rank=rank)
        elif adapter_type == 'lora':
            self.adapter = LoRAAdapter(token_dim, embed_dim, rank=rank)
        else:
            raise ValueError("Unknown adapter_type")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        proj = self.backbone_proj(tokens)
        res = self.adapter(tokens)
        return proj + res


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, n_heads: int = 3, temperature: float = 1.0):
        super().__init__()
        self.heads = nn.ModuleList([MLPHead(in_dim, n_classes) for _ in range(n_heads)])
        self.temperature = temperature
        self.n_heads = n_heads

    def logits_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(x) / self.temperature for h in self.heads], dim=0)

    @torch.no_grad()
    def predictive_entropy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits_all(x)
        probs = logits.softmax(dim=-1).mean(dim=0)
        ent = softmax_entropy(probs)
        return ent, logits.mean(dim=0)


# -----------------------------
# Sinkhorn OT
# -----------------------------

def sinkhorn(cost: torch.Tensor, eps: float = 0.05, max_iter: int = 10) -> torch.Tensor:
    N, K = cost.shape
    a = torch.full((N,), 1.0 / max(N, 1), device=cost.device, dtype=cost.dtype)
    b = torch.full((K,), 1.0 / max(K, 1), device=cost.device, dtype=cost.dtype)
    Kmat = torch.exp(-cost / max(eps, 1e-6))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max_iter):
        u = a / (Kmat @ v + 1e-9)
        v = b / (Kmat.t() @ u + 1e-9)
    P = torch.diag(u) @ Kmat @ torch.diag(v)
    return P


# -----------------------------
# FedU-Align Model
# -----------------------------

class FedUAlignModel(nn.Module):
    def __init__(
        self,
        token_dims: Dict[str, int],
        embed_dim: int,
        n_classes: int,
        adapter_type: str = 'spectral',
        rank: int = 8,
        n_anchors: int = 64,
        ensemble_heads: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.encoders = nn.ModuleDict({
            'vision': ModalityEncoder(token_dims['vision'], embed_dim, adapter_type, rank),
            'audio': ModalityEncoder(token_dims['audio'], embed_dim, adapter_type, rank),
            'text': ModalityEncoder(token_dims['text'], embed_dim, adapter_type, rank),
        })
        self.reft = ReFTIntervention(embed_dim, adapter_type=adapter_type, rank=rank)
        self.anchors = nn.Parameter(torch.randn(n_anchors, embed_dim) * 0.02)

        self.heads = nn.ModuleDict({
            'vision': EnsembleHead(embed_dim, n_classes, n_heads=ensemble_heads),
            'audio': EnsembleHead(embed_dim, n_classes, n_heads=ensemble_heads),
            'text': EnsembleHead(embed_dim, n_classes, n_heads=ensemble_heads),
        })

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reft(x)
        return h.mean(dim=1)

    def forward_modalities(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reps = {}
        for m in ['vision', 'audio', 'text']:
            tokens = batch[f'{m}_tokens']  # always present tensor
            z = self.encoders[m](tokens)
            reps[m] = self.pool_tokens(z)
        return reps

    def heads_logits_entropy(
        self,
        reps: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        logits, entropy = {}, {}
        for m in ['vision', 'audio', 'text']:
            ent, log = self.heads[m].predictive_entropy(reps[m])
            logits[m] = log
            entropy[m] = ent
        return logits, entropy

    def fusion_logits(
        self,
        logits: Dict[str, torch.Tensor],
        entropies: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        beta: float = 4.0,
    ) -> torch.Tensor:
        # logits/entropies: [B,C] and [B] per modality. mask: [B,3] with 1=present,0=missing
        B = next(iter(logits.values())).shape[0]
        C = next(iter(logits.values())).shape[1]
        ents = torch.stack([entropies['vision'], entropies['audio'], entropies['text']], dim=-1)  # [B,3]
        m = mask.float()
        masked_scores = -beta * ents + torch.log(m + 1e-8)  # missing -> large negative
        w = torch.softmax(masked_scores, dim=-1)  # [B,3]
        # fuse
        fused = (
            w[:, 0].unsqueeze(-1) * logits['vision'] +
            w[:, 1].unsqueeze(-1) * logits['audio'] +
            w[:, 2].unsqueeze(-1) * logits['text']
        )
        return fused

    def ot_anchor_loss(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        entropy_weight: float = 1.0,
        eps: float = 0.05,
        iters: int = 5,
    ) -> torch.Tensor:
        B, T, D = tokens.shape
        tok = tokens.reshape(B * T, D)
        with torch.no_grad():
            c = torch.cdist(tok, anchors) ** 2
            P = sinkhorn(c, eps=eps, max_iter=iters)
        cost_grad = (torch.cdist(tok, anchors) ** 2)
        loss = (P * cost_grad).sum() / max(B * T, 1)
        return entropy_weight * loss

    def adapter_parameters(self) -> List[nn.Parameter]:
        params = []
        for m in ['vision', 'audio', 'text']:
            params += list(self.encoders[m].adapter.parameters())
        params += list(self.reft.adapter.parameters())
        return params


# -----------------------------
# Loss helpers
# -----------------------------

def mc_infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07, num_mc: int = 3, dropout_p: float = 0.1) -> torch.Tensor:
    B, D = z1.shape
    if B == 0:
        return torch.tensor(0.0, device=z1.device)
    logits_agg = 0.0
    for _ in range(num_mc):
        q = F.dropout(z1, p=dropout_p, training=True)
        k = F.dropout(z2, p=dropout_p, training=True)
        logits = (q @ k.t()) / temperature
        logits_agg = logits_agg + logits
    logits_agg = logits_agg / num_mc
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(logits_agg, labels)
    return loss


# -----------------------------
# Federated Client and Aggregator
# -----------------------------

class FederatedClient:
    def __init__(self, cfg: ClientConfig, model: FedUAlignModel, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model

    def local_train(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        beta_fusion: float,
        lambda_ot: float,
        lambda_infonce: float,
        spectral_l2: float = 1e-4,
        sink_eps: float = 0.05,
        sink_iters: int = 5,
        fault_mode: Optional[str] = None,
    ) -> Dict:
        self.model.train()
        dl = DataLoader(self.cfg.train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        params = list(self.model.adapter_parameters()) + list(self.model.heads.parameters()) + [self.model.anchors]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        losses = []
        entropies_collected: List[torch.Tensor] = []
        for _ in range(epochs):
            for batch in dl:
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)  # [B,3]
                for k in ['vision_tokens', 'audio_tokens', 'text_tokens']:
                    batch[k] = batch[k].to(self.device)

                reps = self.model.forward_modalities(batch)
                logits_dict, entropy_dict = self.model.heads_logits_entropy(reps)
                fused_logits = self.model.fusion_logits(logits_dict, entropy_dict, mask=mask, beta=beta_fusion)
                ce = F.cross_entropy(fused_logits, labels)

                # collect mean entropy over present modalities
                present_counts = mask.sum(dim=1).clamp_min(1.0)
                ent_mat = torch.stack([entropy_dict['vision'], entropy_dict['audio'], entropy_dict['text']], dim=-1)
                ent_present = (ent_mat * mask).sum(dim=1) / present_counts
                entropies_collected.append(ent_present.mean().detach().cpu())

                # MC-InfoNCE for available pairs
                inf_losses = []
                idx_va = torch.where((mask[:, 0] == 1) & (mask[:, 1] == 1))[0]
                if idx_va.numel() > 0:
                    inf_losses.append(mc_infonce(reps['vision'][idx_va], reps['audio'][idx_va]))
                idx_vt = torch.where((mask[:, 0] == 1) & (mask[:, 2] == 1))[0]
                if idx_vt.numel() > 0:
                    inf_losses.append(mc_infonce(reps['vision'][idx_vt], reps['text'][idx_vt]))
                idx_at = torch.where((mask[:, 1] == 1) & (mask[:, 2] == 1))[0]
                if idx_at.numel() > 0:
                    inf_losses.append(mc_infonce(reps['audio'][idx_at], reps['text'][idx_at]))
                inf_total = sum(inf_losses) / len(inf_losses) if len(inf_losses) > 0 else torch.tensor(0.0, device=self.device)

                # OT loss per modality on available samples
                ot_losses = []
                lnC = math.log(self.model.n_classes + 1e-6)
                for mi, m in enumerate(['vision', 'audio', 'text']):
                    idx = torch.where(mask[:, mi] == 1)[0]
                    if idx.numel() > 0:
                        ent = entropy_dict[m][idx]
                        norm_ent = (ent / lnC).clamp(0, 1).mean().item()
                        w = 1.0 - float(np.clip(norm_ent, 0.0, 1.0))
                        ot_losses.append(self.model.ot_anchor_loss(batch[f'{m}_tokens'][idx], self.model.anchors, entropy_weight=w, eps=sink_eps, iters=sink_iters))
                ot_total = sum(ot_losses) / len(ot_losses) if len(ot_losses) > 0 else torch.tensor(0.0, device=self.device)

                # spectral reg
                spec_reg = torch.tensor(0.0, device=self.device)
                for p in self.model.adapter_parameters():
                    spec_reg = spec_reg + (p ** 2).mean()

                loss = ce + lambda_infonce * inf_total + lambda_ot * ot_total + spectral_l2 * spec_reg

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                losses.append(float(loss.item()))

        # Adapter delta vector (proxy) and anchor grad proxy
        delta_vec, _ = flatten_adapter_deltas(self.model)
        anchor_grad = torch.randn_like(self.model.anchors) * 0.01  # proxy for demo

        # Fault injection on payload
        if fault_mode == 'sign_flip':
            delta_vec = -delta_vec
            anchor_grad = -anchor_grad
        elif fault_mode == 'stale':
            delta_vec = torch.zeros_like(delta_vec)
            anchor_grad = torch.zeros_like(anchor_grad)

        if len(entropies_collected) > 0:
            ent_t = torch.stack(entropies_collected)
            unc_mean = float(ent_t.mean().item())
            unc_var = float(ent_t.var(unbiased=False).item())
            unc_q90 = float(torch.quantile(ent_t, 0.9).item())
        else:
            unc_mean, unc_var, unc_q90 = 0.5, 0.0, 0.5

        return {
            'losses': losses,
            'delta_vec': delta_vec.detach().cpu(),
            'anchor_grad': anchor_grad.detach().cpu(),
            'unc_summary': (unc_mean, unc_var, unc_q90),
        }

    def evaluate(self, dataset: SyntheticMultimodalDataset, batch_size: int, beta_fusion: float) -> Dict:
        self.model.eval()
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dl:
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)
                for k in ['vision_tokens', 'audio_tokens', 'text_tokens']:
                    batch[k] = batch[k].to(self.device)
                reps = self.model.forward_modalities(batch)
                logits_dict, entropy_dict = self.model.heads_logits_entropy(reps)
                fused_logits = self.model.fusion_logits(logits_dict, entropy_dict, mask=mask, beta=beta_fusion)
                all_logits.append(fused_logits.cpu())
                all_labels.append(labels.cpu())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = logits.argmax(dim=-1)
        acc = float((preds == labels).float().mean().item())
        ece = compute_ece(logits, labels, n_bins=15)
        return {
            'acc': acc,
            'ece': ece,
            'preds': preds.numpy(),
            'labels': labels.numpy(),
        }


def flatten_adapter_deltas(model: FedUAlignModel) -> Tuple[torch.Tensor, List[str]]:
    vecs = []
    names = []
    for m in ['vision', 'audio', 'text']:
        for n, p in model.encoders[m].adapter.named_parameters():
            if p.requires_grad:
                vecs.append(p.detach().flatten())
                names.append(f'{m}.adapter.{n}')
    for n, p in model.reft.adapter.named_parameters():
        if p.requires_grad:
            vecs.append(p.detach().flatten())
            names.append(f'reft.adapter.{n}')
    if len(vecs) == 0:
        return torch.empty(0), []
    return torch.cat(vecs), names


def randomized_svd_compress(mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    U, S, Vh = torch.linalg.svd_lowrank(mat, q=min(rank, max(1, min(mat.shape) - 1)))
    return U, S, Vh


def randomized_svd_decompress(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    return (U * S) @ Vh


class ConformalAggregator:
    def __init__(self, q: float = 0.1, calib_window: int = 20):
        self.q = q
        self.calib_window = calib_window
        self.ref: Optional[torch.Tensor] = None
        self.calib_scores: List[float] = []

    def _similarity(self, g: torch.Tensor, ref: torch.Tensor) -> float:
        g = g.flatten()
        ref = ref.flatten()
        g = g / (g.norm() + 1e-8)
        ref = ref / (ref.norm() + 1e-8)
        return float(torch.dot(g, ref).item())

    def select(self, grads: List[torch.Tensor], uncertainties: List[float]) -> Tuple[torch.Tensor, List[int], List[float]]:
        if len(grads) == 0:
            return torch.zeros(1), [], []
        if self.ref is None:
            self.ref = torch.mean(torch.stack([g.flatten() for g in grads]), dim=0)
        sims = [self._similarity(g, self.ref) for g in grads]
        calib = self.calib_scores[-self.calib_window:] if len(self.calib_scores) >= 5 else sims
        pvals = []
        for s in sims:
            rank = sum(c <= s for c in calib) + 1
            p = rank / (len(calib) + 1)
            pvals.append(p)
        rejects = benjamini_hochberg(pvals, q=self.q)
        selected_idx = [i for i, r in enumerate(rejects) if r]
        if selected_idx:
            sel_grads = [grads[i] for i in selected_idx]
            sel_uncs = [uncertainties[i] for i in selected_idx]
            self.ref = 0.9 * self.ref + 0.1 * torch.mean(torch.stack([g.flatten() for g in sel_grads]), dim=0)
            self.calib_scores.extend([sims[i] for i in selected_idx])
            w = torch.tensor([1.0 / (u + 1e-6) for u in sel_uncs], dtype=torch.float32)
            w = w / w.sum()
            agg = 0.0
            for wi, gi in zip(w, sel_grads):
                agg = agg + wi * gi
            return agg, selected_idx, [pvals[i] for i in selected_idx]
        else:
            agg = torch.mean(torch.stack(grads), dim=0)
            return agg, list(range(len(grads))), pvals


# -----------------------------
# Round-based training helpers
# -----------------------------

def train_one_round(
    model: FedUAlignModel,
    clients_cfg: List[ClientConfig],
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    beta_fusion: float,
    lambda_ot: float,
    lambda_infonce: float,
    aggregator: ConformalAggregator,
    svd_rank: int,
    faulty_ids: Optional[set] = None,
) -> Tuple[FedUAlignModel, float, int, List[float], List[float]]:
    faulty_ids = faulty_ids or set()
    client_updates = []
    comm_payload_mb = 0.0
    losses_trace: List[float] = []

    for cfg in clients_cfg:
        client = FederatedClient(cfg, model, device)
        fault_mode = None
        if cfg.client_id in faulty_ids:
            fault_mode = random.choice(['sign_flip', 'stale', None])
        out = client.local_train(
            epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            beta_fusion=beta_fusion,
            lambda_ot=lambda_ot,
            lambda_infonce=lambda_infonce,
            fault_mode=fault_mode,
        )
        losses_trace += out['losses']
        delta_vec = out['delta_vec']
        anchor_grad = out['anchor_grad']
        unc_mean, _, _ = out['unc_summary']

        U, S, Vh = randomized_svd_compress(anchor_grad, rank=svd_rank)
        comm_payload_mb += estimate_payload_mb([U, S, Vh, delta_vec])
        client_updates.append({'U': U, 'S': S, 'Vh': Vh, 'uncertainty': unc_mean, 'faulty': cfg.client_id in faulty_ids})

    grads = [randomized_svd_decompress(u['U'], u['S'], u['Vh']) for u in client_updates]
    uncs = [u['uncertainty'] for u in client_updates]
    agg_grad, selected_idx, pvals = aggregator.select(grads, uncs)

    with torch.no_grad():
        model.anchors -= 0.1 * agg_grad.to(model.anchors.device)

    return model, comm_payload_mb, len(selected_idx), pvals, losses_trace
