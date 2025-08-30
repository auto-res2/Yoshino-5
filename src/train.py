import os
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional deps (listed in requirements.txt)
try:
    import timm  # noqa: F401
except Exception:
    timm = None  # not used in quick test

try:
    from sklearn.cluster import KMeans  # noqa: F401
except Exception:
    KMeans = None


# ------------------------
# Models & PEFT (simple LoRA for linear layers)
# ------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(256, emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        z = self.features(x)
        z = self.flatten(z)
        e = self.proj(z)
        h = F.relu(e)
        out = self.head(h)
        if return_embed:
            return out, e
        return out


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, scale: float = 1.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.scale = scale
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())


class PEFTWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, lora_rank: int = 8):
        super().__init__()
        self.backbone = backbone
        # Try to wrap last linear layer with LoRA
        target = None
        if hasattr(self.backbone, 'head') and isinstance(self.backbone.head, nn.Linear):
            target = ('head', self.backbone.head)
        elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Linear):
            target = ('classifier', self.backbone.classifier)
        else:
            last_name, last_mod = None, None
            for n, m in self.backbone.named_modules():
                if isinstance(m, nn.Linear):
                    last_name, last_mod = n, m
            if last_mod is not None:
                target = (last_name, last_mod)
        if target is not None:
            name, mod = target
            parent = self.backbone
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(mod, rank=lora_rank))
        # Freeze backbone, train only LoRA params
        for n, p in self.backbone.named_parameters():
            if ('.A' in n) or ('.B' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        if return_embed:
            try:
                return self.backbone(x, return_embed=True)
            except TypeError:
                out = self.backbone(x)
                # Fallback: use logits as embeddings if backbone lacks return_embed
                return out, out
        return self.backbone(x)


# ------------------------
# Utilities for CL training
# ------------------------

def build_backbone(dataset_name: str, num_classes: int):
    name = dataset_name.lower()
    if name in ["mnist", "permuted_mnist", "rotated_mnist"]:
        return SmallCNN(in_ch=3, num_classes=num_classes, emb_dim=128)
    elif name in ["cifar100", "imagenet100", "imagenet"]:
        return SmallCNN(in_ch=3, num_classes=num_classes, emb_dim=256)
    else:
        return SmallCNN(in_ch=3, num_classes=num_classes, emb_dim=128)


@torch.no_grad()
def compute_embeddings(model: nn.Module, loader: DataLoader, device: str, max_samples: int = 32):
    model.eval()
    embs, ys = [], []
    n_collected = 0
    for x, y in loader:
        if n_collected >= max_samples:
            break
        x = x.to(device)
        try:
            _, e = model(x, return_embed=True)
        except TypeError:
            # Fallback if wrapper didn't handle: use logits as embedding
            e = model(x)
        e = e.detach().cpu()
        # Cap to max_samples even if batch is larger
        remaining = max_samples - n_collected
        if e.size(0) > remaining:
            e = e[:remaining]
            y = y[:remaining]
        embs.append(e)
        ys.append(y)
        n_collected += e.size(0)
    if len(embs) == 0:
        return torch.zeros(1, 16), torch.zeros(1, dtype=torch.long)
    return torch.cat(embs, 0), torch.cat(ys, 0)


@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X is None or Y is None or X.numel() == 0 or Y.numel() == 0:
        return 0.0
    # Ensure same number of samples
    n = min(X.shape[0], Y.shape[0])
    if n == 0:
        return 0.0
    if X.shape[0] != n:
        X = X[:n]
    if Y.shape[0] != n:
        Y = Y[:n]
    # Center features
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    HSIC = (X.t() @ Y).pow(2).sum()
    var1 = (X.t() @ X).pow(2).sum().clamp_min(1e-8)
    var2 = (Y.t() @ Y).pow(2).sum().clamp_min(1e-8)
    return float((HSIC / (var1.sqrt() * var2.sqrt())).item())


def grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) == 0:
        return torch.tensor([])
    return torch.cat(grads)


@torch.no_grad()
def cosine(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if u.numel() == 0 or v.numel() == 0:
        return torch.tensor(0.0)
    return F.cosine_similarity(u, v, dim=0).clamp(-1, 1)


def compute_grad_conflict(model: nn.Module, device: str, cand_loader: DataLoader, anchor_loader: DataLoader, criterion: nn.Module) -> float:
    model.train()
    try:
        x_c, y_c = next(iter(cand_loader))
    except StopIteration:
        return 0.0
    x_c, y_c = x_c.to(device), y_c.to(device)
    model.zero_grad(set_to_none=True)
    out_c = model(x_c)
    loss_c = criterion(out_c, y_c)
    loss_c.backward()
    g_c = grad_vector(model).detach().cpu()

    try:
        x_a, y_a = next(iter(anchor_loader))
    except StopIteration:
        return 0.0
    x_a, y_a = x_a.to(device), y_a.to(device)
    model.zero_grad(set_to_none=True)
    out_a = model(x_a)
    loss_a = criterion(out_a, y_a)
    loss_a.backward()
    g_a = grad_vector(model).detach().cpu()
    return float(cosine(g_c, g_a).item())


def compute_lmc_loss(model: nn.Module, device: str, cand_loader: DataLoader, anchor_loader: DataLoader, criterion: nn.Module, lr: float = 1e-2) -> float:
    model.train()
    try:
        x_c, y_c = next(iter(cand_loader))
    except StopIteration:
        return 0.0
    x_c, y_c = x_c.to(device), y_c.to(device)
    model.zero_grad(set_to_none=True)
    out_c = model(x_c)
    loss_c = criterion(out_c, y_c)
    loss_c.backward()
    step = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            step[n] = -lr * p.grad.detach().clone()
    with torch.no_grad():
        base = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    alphas = [0.0, 0.5, 1.0]
    losses = []
    for a in alphas:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in step:
                    p.data = base[n] + a * step[n]
        try:
            x_a, y_a = next(iter(anchor_loader))
        except StopIteration:
            l = 0.0
        else:
            x_a, y_a = x_a.to(device), y_a.to(device)
            with torch.no_grad():
                l = float(criterion(model(x_a), y_a).item())
        losses.append(l)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in base:
                p.data = base[n]
    return float(np.mean(losses))


def compute_surrogates(model: nn.Module, device: str, cand_loader: DataLoader, anchor_loader: DataLoader, rb_embeds: Optional[torch.Tensor], criterion: nn.Module) -> Tuple[float, float, float]:
    with torch.no_grad():
        E_c, _ = compute_embeddings(model, cand_loader, device)
    Fi = linear_cka(rb_embeds, E_c) if rb_embeds is not None else 0.0
    Gi = compute_grad_conflict(model, device, cand_loader, anchor_loader, criterion)
    Mi = compute_lmc_loss(model, device, cand_loader, anchor_loader, criterion)
    return float(Fi), float(Gi), float(Mi)


class ReplayBuffer:
    def __init__(self, anchor_size: int = 32):
        self.anchor: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.anchor_size = anchor_size

    def add_anchor(self, task_id: int, loader: DataLoader):
        Xs, Ys = [], []
        try:
            x, y = next(iter(loader))
            Xs.append(x)
            Ys.append(y)
        except StopIteration:
            pass
        if len(Xs) == 0:
            return
        X = torch.cat(Xs, 0)[: self.anchor_size]
        Y = torch.cat(Ys, 0)[: self.anchor_size]
        self.anchor[task_id] = (X.cpu(), Y.cpu())

    def build_anchor_loader(self, batch_size: int = 32) -> DataLoader:
        if len(self.anchor) == 0:
            X = torch.zeros(batch_size, 3, 32, 32)
            Y = torch.zeros(batch_size, dtype=torch.long)
            ds = torch.utils.data.TensorDataset(X, Y)
            return DataLoader(ds, batch_size=batch_size, shuffle=False)
        Xs, Ys = [], []
        for (X, Y) in self.anchor.values():
            Xs.append(X)
            Ys.append(Y)
        X = torch.cat(Xs, 0)
        Y = torch.cat(Ys, 0)
        ds = torch.utils.data.TensorDataset(X, Y)
        return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)


class MetaMLP(nn.Module):
    def __init__(self, d_in: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)


def train_meta_mlp(train_data: List[Tuple[np.ndarray, float]], val_data: List[Tuple[np.ndarray, float]], epochs: int = 20, lr: float = 1e-3, device: str = 'cpu') -> MetaMLP:
    model = MetaMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def to_batch(data):
        X = torch.tensor(np.stack([d[0] for d in data], axis=0), dtype=torch.float32).to(device)
        y = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)
        return X, y

    if len(train_data) == 0:
        return model
    Xtr, ytr = to_batch(train_data)
    Xva, yva = to_batch(val_data) if len(val_data) else (Xtr, ytr)
    best = 1e9
    best_state = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = F.mse_loss(pred, ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(Xva), yva).item()
        if v < best:
            best = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class PTSScheduler:
    def __init__(self, omega: MetaMLP, K: int = 5, D: Optional[int] = None, device: str = 'cpu'):
        self.omega = omega.eval().to(device)
        self.K = K
        self.D = K if D is None else D
        self.device = device

    @torch.no_grad()
    def score(self, feat: np.ndarray) -> float:
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.omega(x).item())

    @torch.no_grad()
    def select_next(self, buffer_feats: List[Tuple[int, np.ndarray]]) -> int:
        items = buffer_feats[: self.K]
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        for _ in range(self.D):
            new_beams: List[Tuple[List[int], float]] = []
            for seq, sc in beams:
                used = set(seq)
                for tid, feat in items:
                    if tid in used:
                        continue
                    s = self.score(feat)
                    new_beams.append((seq + [tid], sc + s))
            if not new_beams:
                break
            new_beams = sorted(new_beams, key=lambda x: -x[1])[: self.K]
            beams = new_beams
        return beams[0][0][0] if beams and beams[0][0] else items[0][0]


def train_task(model: nn.Module, device: str, train_loader: DataLoader, criterion: nn.Module, epochs: int = 1, lr: float = 5e-4, log_losses: Optional[List[float]] = None):
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            if log_losses is not None:
                log_losses.append(float(loss.item()))


# ------------------------
# Baseline ordering (lightweight variants)
# ------------------------

def chronological_order(T: int) -> List[int]:
    return list(range(T))


def random_order(T: int) -> List[int]:
    l = list(range(T))
    random.shuffle(l)
    return l
