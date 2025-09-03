import math
import itertools
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Relative imports for models/utilities
from .train import (
    SmallCNNBackbone,
    extract_image_features,
    BoWBackbone,
    extract_text_features,
)

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------
# STORM Controller
# ------------------------------
class StormController:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, buffer_size: int = 5):
        self.G: Dict[int, torch.Tensor] = {}
        self.alpha = alpha
        self.beta = beta
        self.buffer_size = buffer_size

    def add_signature(self, task_id: int, g: torch.Tensor):
        g = g.detach().float().cpu()
        g = torch.nn.functional.normalize(g, dim=0)
        self.G[task_id] = g

    def pair_score(self, i: int, j: int) -> float:
        gi, gj = self.G[i], self.G[j]
        cos = torch.dot(gi, gj)
        prod = gi * gj
        inter = torch.sum(torch.abs(prod[prod < 0]))
        return float(self.alpha * cos - self.beta * inter)

    def greedy_order(self, buffer_ids: List[int]) -> List[int]:
        n = len(buffer_ids)
        if n <= 1:
            return list(buffer_ids)
        # Dense score matrix
        M = torch.zeros(n, n)
        for a, i in enumerate(buffer_ids):
            for b, j in enumerate(buffer_ids):
                if i == j:
                    continue
                M[a, b] = self.pair_score(i, j)
        order_idx = [0]
        rest = list(range(1, n))

        def score(seq_idx: List[int]) -> float:
            s = 0.0
            for a in range(len(seq_idx)):
                for b in range(a + 1, len(seq_idx)):
                    s += float(M[seq_idx[a], seq_idx[b]].item())
            return s
        while rest:
            best_seq, best_sc = None, -1e30
            for r in rest:
                for k in range(len(order_idx) + 1):
                    cand = order_idx[:k] + [r] + order_idx[k:]
                    sc = score(cand)
                    if sc > best_sc:
                        best_seq, best_sc = cand, sc
            order_idx = best_seq
            rest = [x for x in rest if x not in order_idx]
        return [buffer_ids[i] for i in order_idx]


def build_M_from_signatures(G: List[torch.Tensor], alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    T = len(G)
    M = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            cos = torch.dot(G[i], G[j])
            prod = G[i] * G[j]
            inter = torch.sum(torch.abs(prod[prod < 0]))
            M[i, j] = alpha * cos - beta * inter
    return M


def greedy_order_from_M(M: torch.Tensor) -> List[int]:
    T = M.size(0)
    if T <= 1:
        return list(range(T))
    order = [0]
    rest = list(range(1, T))

    def score(seq: List[int]) -> float:
        s = 0.0
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                s += float(M[seq[a], seq[b]].item())
        return s

    while rest:
        best, best_sc = None, -1e30
        for r in rest:
            for k in range(len(order) + 1):
                cand = order[:k] + [r] + order[k:]
                sc = score(cand)
                if sc > best_sc:
                    best, best_sc = cand, sc
        order = best
        rest = [x for x in rest if x not in order]
    return order


# ------------------------------
# Signatures from features via a tiny probe
# ------------------------------

def compute_signature_from_features(phi: torch.Tensor, y: torch.Tensor, num_classes: int,
                                    epochs: int = 1, bs: int = 256, device: str = 'cpu') -> torch.Tensor:
    ds = torch.utils.data.TensorDataset(phi, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    probe = nn.Linear(phi.size(1), num_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    sig = torch.zeros(phi.size(1), device=device)
    for _ in range(epochs):
        for z, t in loader:
            z = z.to(device).requires_grad_(True)
            t = t.to(device)
            opt.zero_grad(set_to_none=True)
            logits = probe(z)
            loss = nn.CrossEntropyLoss()(logits, t)
            loss.backward()
            sig += z.grad.detach().sum(dim=0)
            opt.step()
    sig = torch.nn.functional.normalize(sig, dim=0)
    return sig.detach().cpu()


# ------------------------------
# Experiment 1 (Synthetic) data
# ------------------------------

def make_linear_tasks(T: int = 10, n_train: int = 4000, n_test: int = 1000, d: int = 64, D: int = 256,
                      angle_pattern: str = 'sweep', seed: int = 0) -> Tuple[List[Dict], torch.Tensor]:
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, d, generator=g) / math.sqrt(d)

    # Construct task normals w_t in R^d with controlled pairwise angles
    W = []
    if angle_pattern == 'sweep':
        base = torch.randn(d, generator=g); base = base / base.norm()
        for t in range(T):
            v = torch.randn(d, generator=g)
            v = v - (v @ base) * base
            v = v / (v.norm() + 1e-8)
            angle = (t / max(1, T - 1)) * math.pi
            w = math.cos(angle) * base + math.sin(angle) * v
            W.append(w)
    elif angle_pattern == 'clustered':
        centers = [torch.randn(d, generator=g) for _ in range(3)]
        centers = [c / c.norm() for c in centers]
        for t in range(T):
            c = centers[t % 3]
            noise = torch.randn(d, generator=g)
            noise = noise - (noise @ c) * c
            noise = noise / (noise.norm() + 1e-8)
            angle = np.random.default_rng(seed + t).uniform(0, math.pi / 8)
            w = math.cos(angle) * c + math.sin(angle) * noise
            W.append(w)
    else:
        for _ in range(T):
            w = torch.randn(d, generator=g); w = w / (w.norm() + 1e-8)
            W.append(w)

    tasks = []
    for w in W:
        Xtr = torch.randn(n_train, d, generator=g)
        Xte = torch.randn(n_test, d, generator=g)
        def label(X):
            y = (X @ w + 0.1 * torch.randn(X.size(0), generator=g) > 0).long()
            return y
        ytr, yte = label(Xtr), label(Xte)
        phi_tr = torch.nn.functional.gelu(Xtr @ A.t())
        phi_te = torch.nn.functional.gelu(Xte @ A.t())
        tasks.append(dict(phi_tr=phi_tr, ytr=ytr, phi_te=phi_te, yte=yte))
    return tasks, A


# ------------------------------
# Experiment 2 (Vision) tasks and signatures
# ------------------------------

def make_vision_tasks(dataset: str = 'cifar10', num_tasks: int = 5, seed: int = 0, samples_per_class: int = 200,
                      quick: bool = True):
    set_seed(seed)
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, CIFAR100, FakeData

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset.lower() == 'cifar100':
        try:
            train_ds = CIFAR100(root='./data', train=True, download=True, transform=transform)
            test_ds = CIFAR100(root='./data', train=False, download=True, transform=transform)
            total_classes = 100
        except Exception:
            total_classes = 10
            size = (3, 32, 32)
            ntr = 2000 if not quick else 400
            nte = 500 if not quick else 200
            train_ds = FakeData(size=ntr, image_size=size, num_classes=total_classes, transform=transform)
            test_ds = FakeData(size=nte, image_size=size, num_classes=total_classes, transform=transform)
    elif dataset.lower() == 'cifar10':
        try:
            train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
            total_classes = 10
            num_tasks = min(num_tasks, 5)
        except Exception:
            total_classes = 10
            size = (3, 32, 32)
            ntr = 2000 if not quick else 400
            nte = 500 if not quick else 200
            train_ds = FakeData(size=ntr, image_size=size, num_classes=total_classes, transform=transform)
            test_ds = FakeData(size=nte, image_size=size, num_classes=total_classes, transform=transform)
    else:
        total_classes = 10
        size = (3, 32, 32)
        ntr = 2000 if not quick else 400
        nte = 500 if not quick else 200
        train_ds = FakeData(size=ntr, image_size=size, num_classes=total_classes, transform=transform)
        test_ds = FakeData(size=nte, image_size=size, num_classes=total_classes, transform=transform)

    # Partition classes equally per task
    classes = list(range(total_classes))
    rng = np.random.default_rng(seed)
    rng.shuffle(classes)
    per_task = max(1, total_classes // num_tasks)
    partitions = [classes[i * per_task:(i + 1) * per_task] for i in range(num_tasks)]

    def idxs_for(ds, cls_list):
        idxs = []
        for i in range(len(ds)):
            y = ds[i][1]
            if y in cls_list:
                idxs.append(i)
        return idxs

    tasks = []
    for t, cls_list in enumerate(partitions):
        tr_idx = idxs_for(train_ds, cls_list)
        te_idx = idxs_for(test_ds, cls_list)
        # Subsample train indices for speed and balance
        if samples_per_class is not None and len(cls_list) > 0:
            per_c = min(samples_per_class, max(1, len(tr_idx) // len(cls_list)))
            by_class = {c: [] for c in cls_list}
            for i in tr_idx:
                by_class[train_ds[i][1]].append(i)
            tr_idx_bal = []
            for c in cls_list:
                rng.shuffle(by_class[c])
                tr_idx_bal.extend(by_class[c][:per_c])
            tr_idx = tr_idx_bal
        train_subset = torch.utils.data.Subset(train_ds, tr_idx)
        test_subset = torch.utils.data.Subset(test_ds, te_idx)
        tasks.append(dict(train=train_subset, test=test_subset, classes=cls_list))

    # Frozen random backbone
    feat_dim = 128
    frozen_backbone = SmallCNNBackbone(feat_dim=feat_dim)
    backbone_state = frozen_backbone.state_dict()

    # Compute signatures per task on frozen features
    device = get_device()
    controller = StormController(alpha=1.0, beta=1.0, buffer_size=3)
    signatures = {}
    for tid, task in enumerate(tasks):
        tr_loader = DataLoader(task['train'], batch_size=128, shuffle=False)
        phi_tr, ytr = extract_image_features(frozen_backbone, tr_loader, device)
        # Remap labels to local 0..k-1 for the probe
        unique = sorted(set(ytr.tolist()))
        mapping = {c: i for i, c in enumerate(unique)}
        y_local = torch.tensor([mapping[int(c)] for c in ytr], dtype=torch.long)
        g = compute_signature_from_features(phi_tr, y_local, num_classes=len(unique), epochs=1, device=device)
        controller.add_signature(tid, g)
        signatures[tid] = g

    return tasks, total_classes, backbone_state, feat_dim, controller


# ------------------------------
# Experiment 3 (NLP synthetic) tasks and signatures
# ------------------------------

def make_nlp_tasks(num_tasks: int = 5, seed: int = 0, vocab_size: int = 1000, feat_dim: int = 256,
                   n_classes: int = 4, ntr: int = 400, nte: int = 200, quick: bool = True):
    from .train import SyntheticTextDataset
    set_seed(seed)
    tasks = []
    for t in range(num_tasks):
        tr = SyntheticTextDataset(n_samples=ntr if quick else max(ntr, 800), vocab_size=vocab_size,
                                  n_classes=n_classes, task_seed=seed * 97 + t)
        te = SyntheticTextDataset(n_samples=nte, vocab_size=vocab_size,
                                  n_classes=n_classes, task_seed=seed * 197 + t)
        tasks.append(dict(train=tr, test=te))
    frozen_backbone = BoWBackbone(vocab_size=vocab_size, feat_dim=feat_dim)
    backbone_state = frozen_backbone.state_dict()

    # Compute signatures per task
    device = get_device()
    controller = StormController(alpha=1.0, beta=1.0, buffer_size=3)
    signatures = {}
    for tid, task in enumerate(tasks):
        dl = DataLoader(task['train'], batch_size=128, shuffle=False)
        phi_tr, ytr = extract_text_features(frozen_backbone, dl, device)
        g = compute_signature_from_features(phi_tr, ytr.long(), num_classes=n_classes, epochs=1, device=device)
        controller.add_signature(tid, g)
        signatures[tid] = g

    return tasks, n_classes, backbone_state, feat_dim, controller
