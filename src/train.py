import os
import math
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Matplotlib only used by evaluate.py for plotting; keep backend configurable there

# ------------------------------
# LoRA modules
# ------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(max(1, r))
        # Shapes: base: out x in; A: out x r; B: r x in
        self.A = nn.Parameter(torch.zeros(self.base.out_features, r))
        self.B = nn.Parameter(torch.zeros(r, self.base.in_features))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.B.t()) @ self.A.t()


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(max(1, r))
        in_c = base.in_channels
        out_c = base.out_channels
        # 1x1 LoRA decomposition
        self.A = nn.Parameter(torch.zeros(out_c, r, 1, 1))
        self.B = nn.Parameter(torch.zeros(r, in_c, 1, 1))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * nn.functional.conv2d(
            nn.functional.conv2d(x, self.B, bias=None), self.A, bias=None
        )


# ------------------------------
# Synthetic (Experiment 1) models and training
# ------------------------------
class MLPWithLoRA(nn.Module):
    def __init__(self, D: int, H: int = 128, C: int = 2, r: int = 8, alpha: int = 16):
        super().__init__()
        base1 = nn.Linear(D, H)
        self.lora1 = LoRALinear(base1, r=r, alpha=alpha)
        self.fc2 = nn.Linear(H, C)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.lora1(z))
        return self.fc2(h)


def eval_acc_logits(model: nn.Module, phi: torch.Tensor, y: torch.Tensor, bs: int = 256, device: str = 'cpu') -> float:
    model.eval()
    loader = DataLoader(TensorDataset(phi, y), batch_size=bs, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for z, t in loader:
            z = z.to(device)
            t = t.to(device)
            logits = model(z)
            pred = logits.argmax(dim=1)
            correct += (pred == t).sum().item()
            total += t.numel()
    return correct / max(1, total)


def train_sequence_synthetic(tasks: List[Dict], order: List[int], epochs_per_task: int = 3, bs: int = 256,
                             lr: float = 1e-3, device: str = 'cpu') -> Tuple[np.ndarray, List[float]]:
    D = tasks[0]['phi_tr'].size(1)
    C = int(max([int(tasks[i]['ytr'].max().item()) for i in range(len(tasks))]) + 1)
    model = MLPWithLoRA(D=D, H=128, C=C, r=8, alpha=16).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    acc_matrix = []
    losses_over_time = []
    for t_idx in order:
        ds = DataLoader(TensorDataset(tasks[t_idx]['phi_tr'], tasks[t_idx]['ytr']), batch_size=bs, shuffle=True)
        for _ in range(epochs_per_task):
            for z, y in ds:
                z, y = z.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(z)
                loss = nn.CrossEntropyLoss()(logits, y)
                loss.backward()
                opt.step()
                losses_over_time.append(float(loss.item()))
        accs = []
        for j in range(len(tasks)):
            accs.append(eval_acc_logits(model, tasks[j]['phi_te'], tasks[j]['yte'], device=device))
        acc_matrix.append(accs)
    return np.array(acc_matrix), losses_over_time


# ------------------------------
# Vision (Experiment 2) models and training
# ------------------------------
class SmallCNNBackbone(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.pool(x).flatten(1)
        z = self.proj(x)
        return z


class VisionCLLearner(nn.Module):
    def __init__(self, backbone_state: Dict, feat_dim: int, num_classes: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.backbone = SmallCNNBackbone(feat_dim=feat_dim)
        self.backbone.load_state_dict(backbone_state)
        # Inject LoRA into first conv and projection to keep it tiny
        self.backbone.conv1 = LoRAConv2d(self.backbone.conv1, r=r, alpha=alpha)
        self.backbone.proj = LoRALinear(self.backbone.proj, r=r, alpha=alpha)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits = self.classifier(z)
        return logits


@torch.no_grad()
def extract_image_features(frozen_backbone: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    frozen_backbone.eval().to(device)
    feats = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device)
        z = frozen_backbone(xb)
        feats.append(z.detach().cpu())
        labels.append(yb.detach().cpu())
    return torch.cat(feats), torch.cat(labels)


def train_sequence_vision(tasks: List[Dict], order: List[int], backbone_state: Dict, feat_dim: int,
                          total_classes: int, device: str = 'cpu', quick: bool = True) -> Tuple[np.ndarray, List[float]]:
    learner = VisionCLLearner(backbone_state=backbone_state, feat_dim=feat_dim, num_classes=total_classes, r=8, alpha=16).to(device)
    for p in learner.parameters():
        p.requires_grad = False
    for m in [learner.backbone.conv1, learner.backbone.proj, learner.classifier]:
        for p in m.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, learner.parameters()), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    acc_matrix = []
    loss_curve = []
    for t_idx in order:
        dl = DataLoader(tasks[t_idx]['train'], batch_size=64, shuffle=True)
        epochs = 2 if quick else 5
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = learner(xb)
                    loss = nn.CrossEntropyLoss()(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_curve.append(float(loss.item()))
        # Evaluate on all tasks
        accs = []
        learner.eval()
        with torch.no_grad():
            for j in range(len(tasks)):
                te_loader = DataLoader(tasks[j]['test'], batch_size=128, shuffle=False)
                correct = 0
                total = 0
                for xb, yb in te_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = learner(xb)
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
                accs.append(correct / max(1, total))
        learner.train()
        acc_matrix.append(accs)

    return np.array(acc_matrix), loss_curve


# ------------------------------
# NLP synthetic (Experiment 3) models and training
# ------------------------------
class SyntheticTextDataset(Dataset):
    def __init__(self, n_samples: int, vocab_size: int, n_classes: int, task_seed: int, length_range=(5, 20)):
        super().__init__()
        rng = np.random.default_rng(task_seed)
        self.samples = []
        self.labels = []
        word_bias = rng.random((n_classes, vocab_size))
        word_bias = word_bias / word_bias.sum(axis=1, keepdims=True)
        for _ in range(n_samples):
            y = int(rng.integers(0, n_classes))
            L = int(rng.integers(length_range[0], length_range[1] + 1))
            words = rng.choice(np.arange(vocab_size), size=L, p=word_bias[y], replace=True)
            bow = np.bincount(words, minlength=vocab_size).astype(np.float32)
            self.samples.append(torch.tensor(bow))
            self.labels.append(y)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


class BoWBackbone(nn.Module):
    def __init__(self, vocab_size: int, feat_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(vocab_size, feat_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.data.normal_(0, 1.0 / math.sqrt(vocab_size))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class NLPCLLearner(nn.Module):
    def __init__(self, backbone_state: Dict, feat_dim: int, num_labels: int, r: int = 8, alpha: int = 16):
        super().__init__()
        # Infer vocab size from state dict
        vocab_size = backbone_state['proj.weight'].shape[1]
        self.backbone = BoWBackbone(vocab_size=vocab_size, feat_dim=feat_dim)
        self.backbone.load_state_dict(backbone_state)
        self.backbone.proj = LoRALinear(self.backbone.proj, r=r, alpha=alpha)
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


@torch.no_grad()
def extract_text_features(backbone: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval().to(device)
    feats = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device).float()
        z = backbone(xb)
        feats.append(z.detach().cpu())
        labels.append(torch.tensor(yb))
    return torch.cat(feats), torch.cat(labels)


def train_sequence_nlp(tasks: List[Dict], order: List[int], backbone_state: Dict, feat_dim: int,
                       num_labels: int, device: str = 'cpu', quick: bool = True) -> Tuple[np.ndarray, List[float]]:
    learner = NLPCLLearner(backbone_state=backbone_state, feat_dim=feat_dim, num_labels=num_labels, r=8, alpha=16).to(device)
    for p in learner.parameters():
        p.requires_grad = False
    for m in [learner.backbone.proj, learner.head]:
        for p in m.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, learner.parameters()), lr=1e-3)

    acc_matrix = []
    losses = []
    for t_idx in order:
        dl = DataLoader(tasks[t_idx]['train'], batch_size=128, shuffle=True)
        epochs = 2 if quick else 4
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device).float()
                yb = torch.tensor(yb).to(device)
                opt.zero_grad(set_to_none=True)
                logits = learner(xb)
                loss = nn.CrossEntropyLoss()(logits, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        # Eval all tasks
        accs = []
        learner.eval()
        with torch.no_grad():
            for j in range(len(tasks)):
                te = DataLoader(tasks[j]['test'], batch_size=256, shuffle=False)
                correct = 0
                total = 0
                for xb, yb in te:
                    xb = xb.to(device).float()
                    yb = torch.tensor(yb).to(device)
                    logits = learner(xb)
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
                accs.append(correct / max(1, total))
        learner.train()
        acc_matrix.append(accs)

    return np.array(acc_matrix), losses
