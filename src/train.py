import os
import time
import math
import json
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Matplotlib is only used in evaluate.py for plotting


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------
# Lightweight LoRA modules
# -------------------------

class LoraLinear(nn.Module):
    """A minimal LoRA wrapper for nn.Linear.
    y = x @ W^T + scale * x @ A @ B^T, with A:[in,r], B:[out,r].
    Only A and B are trainable; W is frozen. Bias is trainable by default.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scale = lora_alpha / max(1, r)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        # Frozen base
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base_linear.bias is not None:
            # allow bias fine-tune
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=True)
        # LoRA adapters
        if r > 0:
            self.A = nn.Parameter(torch.zeros(self.in_features, r))
            self.B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            l = self.dropout(x) @ self.A  # [B, r]
            l = l @ self.B.t()            # [B, out]
            base = base + self.scale * l
        return base


def replace_linear_with_lora(module: nn.Module, target_names: Optional[List[str]] = None,
                             r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
    """Recursively replace nn.Linear layers with LoraLinear if name matches target_names; if target_names is None, replace all Linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (target_names is None or any(t in name for t in target_names)):
            setattr(module, name, LoraLinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        else:
            replace_linear_with_lora(child, target_names, r, lora_alpha, lora_dropout)


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# -------------------------
# Tiny models for fast tests
# -------------------------

class TinyCNN(nn.Module):
    """A small CNN with a clearly defined 'first block' for embedding extraction. LoRA applied to head linears.
    Input: 3x32x32, Output: num_classes logits.
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 10, lora_r: int = 16):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.second_block = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
        replace_linear_with_lora(self.head, target_names=None, r=lora_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.first_block(x)
        h2 = self.second_block(h1)
        logits = self.head(h2)
        return logits

    @torch.no_grad()
    def embed_first_block(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first_block(x)
        h = h.mean(dim=[2, 3])  # GAP [B, C]
        return h


class TinyTextTransformer(nn.Module):
    """A minimal text classifier with a tiny Transformer encoder. Includes LoRA on encoder and head linears."""
    def __init__(self, vocab_size: int = 5000, d_model: int = 128, nhead: int = 4, num_layers: int = 1,
                 num_classes: int = 2, max_len: int = 128, lora_r: int = 16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))
        replace_linear_with_lora(self.encoder, target_names=None, r=lora_r)
        replace_linear_with_lora(self.cls_head, target_names=None, r=lora_r)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        pos = self.pos_emb[:, :T, :]
        h = x + pos
        h = self.encoder(h)
        h_cls = h.mean(dim=1)
        logits = self.cls_head(h_cls)
        return logits

    @torch.no_grad()
    def embed_first_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        pos = self.pos_emb[:, :T, :]
        h0 = x + pos
        first_layer = self.encoder.layers[0]
        h1 = first_layer(h0)
        pooled = h1.mean(dim=1)
        return pooled


# -------------------------
# DyCLOS signals
# -------------------------

@torch.no_grad()
def task_fingerprint_embeddings_vision(model: TinyCNN, dataloader: DataLoader, max_samples: int = 512,
                                       device: Optional[str] = None) -> torch.Tensor:
    device = device or get_device()
    embs = []
    seen = 0
    for x, _ in dataloader:
        x = x.to(device)
        h = model.embed_first_block(x)
        embs.append(h.cpu())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return torch.cat(embs, dim=0)[:max_samples]


@torch.no_grad()
def task_fingerprint_embeddings_text(model: TinyTextTransformer, dataloader: DataLoader, max_samples: int = 512,
                                     device: Optional[str] = None) -> torch.Tensor:
    device = device or get_device()
    embs = []
    seen = 0
    for x, _ in dataloader:
        x = x.to(device)
        h = model.embed_first_block(x)
        embs.append(h.cpu())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return torch.cat(embs, dim=0)[:max_samples]


def gaussian_w2(emb1: torch.Tensor, emb2: torch.Tensor, eps: float = 1e-5) -> float:
    """Gaussian 2-Wasserstein distance between empirical Gaussians fit to embeddings."""
    m1, m2 = emb1.mean(0), emb2.mean(0)
    X1, X2 = emb1 - m1, emb2 - m2
    n1 = max(1, emb1.shape[0] - 1)
    n2 = max(1, emb2.shape[0] - 1)
    C1 = (X1.t() @ X1) / n1 + eps * torch.eye(emb1.shape[1])
    C2 = (X2.t() @ X2) / n2 + eps * torch.eye(emb2.shape[1])
    U, S, _ = torch.linalg.svd(C1)
    C1s = (U * S.clamp_min(0).sqrt()) @ U.t()
    M = C1s @ C2 @ C1s
    U2, S2, _ = torch.linalg.svd(M)
    tr_term = torch.trace(C1 + C2 - 2 * (U2 * S2.clamp_min(0).sqrt()) @ U2.t())
    mean_term = torch.sum((m1 - m2) ** 2)
    val = (mean_term + tr_term).clamp_min(0).sqrt().item()
    return float(val)


def flatten_grads(params: List[nn.Parameter]) -> torch.Tensor:
    parts = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def random_projection_matrix(D: int, d_proj: int = 1024, device: Optional[str] = None) -> torch.Tensor:
    device = device or get_device()
    R = torch.randn(D, d_proj, device=device)
    R /= math.sqrt(D)
    return R


def average_grad_vector(model: nn.Module, params: List[nn.Parameter], dataloader: DataLoader, loss_fn,
                        max_batches: int = 8, device: Optional[str] = None) -> torch.Tensor:
    """Accumulate gradients over up to max_batches and return flattened average gradient vector."""
    device = device or get_device()
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    model.train()
    n_used = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            # support dict-style batches if ever used
            x, y = batch['input_ids'], batch['labels']
        x, y = x.to(device), y.to(device)
        # Do NOT zero grads inside the loop to accumulate
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        n_used += 1
    g = flatten_grads(params)
    if n_used > 0:
        g = g / float(n_used)
    # Clear grads to avoid contaminating subsequent training
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    return g


def cosine_interference(model: nn.Module, dl_i: DataLoader, dl_j: DataLoader, loss_fn,
                        proj_dim: int = 1024, max_batches: int = 8, device: Optional[str] = None) -> float:
    params = get_trainable_params(model)
    gi = average_grad_vector(model, params, dl_i, loss_fn, max_batches, device)
    gj = average_grad_vector(model, params, dl_j, loss_fn, max_batches, device)
    D = gi.numel()
    if proj_dim > 0 and D > proj_dim:
        R = random_projection_matrix(D, proj_dim, gi.device)
        gi = gi @ R
        gj = gj @ R
    if gi.numel() == 0 or gj.numel() == 0:
        return 0.0
    cos = F.cosine_similarity(gi, gj, dim=0).item()
    return float(cos)


# -------------------------
# DyCLOS scheduler (windowed + beam search)
# -------------------------

class DyCLOSScheduler:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, window_k: int = 4, beam_size: int = 6):
        self.alpha = alpha
        self.beta = beta
        self.window_k = window_k
        self.beam = beam_size
        self.Sdata: Dict[int, Dict[int, float]] = {}
        self.Sint: Dict[int, Dict[int, float]] = {}
        self.trained_prefix: List[int] = []
        self.untrained_queue: List[int] = []

    def _ensure_dicts(self, tid: int):
        if tid not in self.Sdata:
            self.Sdata[tid] = {}
        if tid not in self.Sint:
            self.Sint[tid] = {}

    def update_signal_pair(self, i: int, j: int, sdata: float, sint: float):
        self._ensure_dicts(i)
        self._ensure_dicts(j)
        self.Sdata[i][j] = sdata
        self.Sdata[j][i] = sdata
        self.Sint[i][j] = sint
        self.Sint[j][i] = sint

    def edge_cost(self, i: int, j: int) -> float:
        sdata = self.Sdata.get(i, {}).get(j, 0.0)
        sint = self.Sint.get(i, {}).get(j, 0.0)
        return self.alpha * sdata + self.beta * max(0.0, -sint)

    def path_cost(self, path: List[int]) -> float:
        return sum(self.edge_cost(path[t], path[t + 1]) for t in range(len(path) - 1))

    def insert_task(self, task_id: int) -> List[int]:
        self.untrained_queue.append(task_id)
        # If enough tasks are queued, solve local window and commit the first
        while len(self.untrained_queue) >= self.window_k:
            window = self.untrained_queue[:self.window_k]
            candidates = [[t] for t in window]
            for _ in range(1, len(window)):
                new_cands = []
                for path in candidates:
                    rem = [t for t in window if t not in path]
                    for t in rem:
                        new_cands.append(path + [t])
                def score(p):
                    cost = 0.0
                    if self.trained_prefix:
                        cost += self.edge_cost(self.trained_prefix[-1], p[0])
                    cost += self.path_cost(p)
                    return cost
                candidates = sorted(new_cands, key=score)[:self.beam]
            def final_score(p):
                c = 0.0
                if self.trained_prefix:
                    c += self.edge_cost(self.trained_prefix[-1], p[0])
                c += self.path_cost(p)
                return c
            best = min(candidates, key=final_score)
            chosen = best[0]
            self.trained_prefix.append(chosen)
            self.untrained_queue.remove(chosen)
        return list(self.trained_prefix) + list(self.untrained_queue)


# -------------------------
# Helper: compute signals for a new task against peers
# -------------------------

def compute_signals_for_new_task_vision(model: TinyCNN, new_task_id: int, peers: List[int], task_dls: Dict[int, Dict[str, Any]],
                                         scheduler: DyCLOSScheduler, max_probe: int = 512, max_batches_grad: int = 8,
                                         device: Optional[str] = None):
    device = device or get_device()
    _ = task_fingerprint_embeddings_vision(model, task_dls[new_task_id]['probe'], max_samples=max_probe, device=device)
    loss_fn = nn.CrossEntropyLoss()
    for t in peers:
        emb_new = task_fingerprint_embeddings_vision(model, task_dls[new_task_id]['probe'], max_samples=max_probe, device=device)
        emb_t = task_fingerprint_embeddings_vision(model, task_dls[t]['probe'], max_samples=max_probe, device=device)
        sdata = gaussian_w2(emb_new, emb_t)
        sint = cosine_interference(model, task_dls[new_task_id]['probe'], task_dls[t]['probe'], loss_fn,
                                   proj_dim=512, max_batches=max_batches_grad, device=device)
        scheduler.update_signal_pair(new_task_id, t, sdata, sint)


def compute_signals_for_new_task_text(model: TinyTextTransformer, new_task_id: int, peers: List[int], task_dls: Dict[int, Dict[str, Any]],
                                       scheduler: DyCLOSScheduler, max_probe: int = 512, max_batches_grad: int = 8,
                                       device: Optional[str] = None):
    device = device or get_device()
    _ = task_fingerprint_embeddings_text(model, task_dls[new_task_id]['probe'], max_samples=max_probe, device=device)
    loss_fn = nn.CrossEntropyLoss()
    for t in peers:
        emb_new = task_fingerprint_embeddings_text(model, task_dls[new_task_id]['probe'], max_samples=max_probe, device=device)
        emb_t = task_fingerprint_embeddings_text(model, task_dls[t]['probe'], max_samples=max_probe, device=device)
        sdata = gaussian_w2(emb_new, emb_t)
        sint = cosine_interference(model, task_dls[new_task_id]['probe'], task_dls[t]['probe'], loss_fn,
                                   proj_dim=512, max_batches=max_batches_grad, device=device)
        scheduler.update_signal_pair(new_task_id, t, sdata, sint)


# -------------------------
# Trainer with stability-aware inner loop and metrics
# -------------------------

class ContinualTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn,
                 task_dls: Dict[int, Dict[str, Any]], scheduler: Optional[DyCLOSScheduler] = None,
                 device: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task_dls = task_dls
        self.scheduler = scheduler
        self.device = device or get_device()
        self.model.to(self.device)
        self.acc_hist: Dict[int, List[float]] = {}
        self.best_seen: Dict[int, float] = {}
        self.seen_tasks: List[int] = []
        self.per_epoch_train_loss: Dict[int, List[float]] = {}

    def eval_accuracy(self, task_id: int, split: str = 'test') -> float:
        self.model.eval()
        dl = self.task_dls[task_id][split]
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dl:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x, y = batch['input_ids'], batch['labels']
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        return (correct / max(1, total))

    def mean_val_over_seen(self, extra_task: Optional[int] = None) -> float:
        tasks = set(self.seen_tasks)
        if extra_task is not None:
            tasks.add(extra_task)
        if not tasks:
            return 0.0
        accs = [self.eval_accuracy(t, split='val') for t in tasks]
        return float(np.mean(accs))

    def train_task(self, task_id: int, epochs: int = 3, base_lr: float = 1e-3):
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)
        for g in self.optimizer.param_groups:
            g['lr'] = base_lr
        train_dl = self.task_dls[task_id]['train']
        base_mean = self.mean_val_over_seen()
        lowered = False
        self.per_epoch_train_loss.setdefault(task_id, [])
        for _ep in range(epochs):
            self.model.train()
            losses = []
            for batch in train_dl:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x, y = batch['input_ids'], batch['labels']
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            mean_loss = float(np.mean(losses)) if losses else 0.0
            self.per_epoch_train_loss[task_id].append(mean_loss)
            cur = self.mean_val_over_seen(extra_task=task_id)
            if not lowered and base_mean - cur > 0.05 * max(base_mean, 1e-6):
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.5
                lowered = True
        # Evaluate all seen tasks after training this one
        for t in self.seen_tasks:
            acc = self.eval_accuracy(t, split='test')
            self.acc_hist.setdefault(t, []).append(acc)
            self.best_seen[t] = max(self.best_seen.get(t, 0.0), acc)

    def compute_metrics(self) -> Tuple[float, float]:
        if not self.seen_tasks:
            return 0.0, 0.0
        last_accs = [self.acc_hist[t][-1] for t in self.seen_tasks if self.acc_hist.get(t)]
        omega_acc = float(np.mean(last_accs)) if last_accs else 0.0
        forg = []
        for t in self.seen_tasks:
            if not self.acc_hist.get(t):
                continue
            last = self.acc_hist[t][-1]
            best = max(self.acc_hist[t])
            forg.append(max(0.0, best - last))
        omega_forg = float(np.mean(forg)) if forg else 0.0
        return omega_acc, omega_forg


# -------------------------
# Baseline helper
# -------------------------

def run_baseline_fixed_order(model: nn.Module, task_ids: List[int], task_dls: Dict[int, Dict[str, Any]],
                             epochs_per_task: int = 2, lr: float = 1e-3) -> Tuple[ContinualTrainer, float, float, float]:
    optimizer = torch.optim.Adam(get_trainable_params(model), lr=lr, weight_decay=0.01)
    trainer = ContinualTrainer(model, optimizer, nn.CrossEntropyLoss(), task_dls, scheduler=None)
    t0 = time.perf_counter()
    for t in task_ids:
        trainer.train_task(t, epochs=epochs_per_task, base_lr=lr)
    t1 = time.perf_counter()
    omega_acc, omega_forg = trainer.compute_metrics()
    wall = t1 - t0
    return trainer, omega_acc, omega_forg, wall
