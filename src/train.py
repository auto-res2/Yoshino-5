"""src/train.py
This module implements all training-related components: models, curriculum
construction, online swapper and the per–task optimisation loop.  Nothing is
new – we strictly refactor logic that already exists in the original
single-file script.
"""
from __future__ import annotations

import itertools
import logging
import random
import time
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from peft import LoraConfig, get_peft_model

# -----------------------------------------------------------------------------
# 1.  Model & Adapters
# -----------------------------------------------------------------------------

class SpectralAdapter(nn.Module):
    """Very small 1×1 depth-wise conv + per-channel scale (used in CLIP paper)."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.dw(x) * self.scale.view(1, -1, 1, 1)


def _inject_lora(module: nn.Module, r: int) -> nn.Module:  # pragma: no cover
    """Attach LoRA adapters to all 3×3 convs inside a module using PEFT."""
    if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
        cfg = LoraConfig(r=r, lora_alpha=r, target_modules=["weight"], bias="none")
        return get_peft_model(module, cfg)
    return module


def resnet18_lora_sa(lora_r: int = 128, pretrained: bool = True) -> nn.Module:
    """Frozen ResNet-18 backbone with LoRA + SpectralAdapters, **trainable**
    parameters ≈0.8 % of backbone."""

    backbone = torchvision.models.resnet18(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # SpectralAdapter after conv2_x / conv3_x
    backbone.layer2.add_module("spectral_adapter", SpectralAdapter(128))
    backbone.layer3.add_module("spectral_adapter", SpectralAdapter(256))

    # LoRA for every 3×3 conv in layer2 & layer3
    backbone.apply(lambda m: _inject_lora(m, lora_r))

    # Classification head (actual #classes patched later by Avalanche)
    backbone.fc = nn.Linear(backbone.fc.in_features, 100)

    # NB:  print_trainable_parameters only if PEFT present – silently ignore.
    if hasattr(backbone, "print_trainable_parameters"):
        backbone.print_trainable_parameters()
    return backbone

# -----------------------------------------------------------------------------
# 2.  Curriculum Construction (Interference graph)
# -----------------------------------------------------------------------------

def projected_gradient_overlap(g1: torch.Tensor, g2: torch.Tensor) -> float:
    return (torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-12)).item()


def build_interference_graph(
    tasks: List[Any], model_fn, pct_data: float, device: torch.device
) -> np.ndarray:
    """Compute pair-wise similarity of tasks using (Fisher cosine + PGO)/2."""

    node_grads: List[torch.Tensor] = []
    for task in tasks:
        model: nn.Module = model_fn(lora_r=64, pretrained=True).to(device)
        loader = DataLoader(task.dataset, batch_size=16, shuffle=True)
        # use only a tiny fraction of the data
        num_batches = max(1, int(len(loader) * pct_data))
        grads: List[torch.Tensor] = []
        for i, (x, y, *_ignored) in enumerate(loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            grads.append(
                torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
            )
        node_grads.append(torch.mean(torch.stack(grads), dim=0).detach())
        del model
        torch.cuda.empty_cache()

    T = len(node_grads)
    graph = np.zeros((T, T), dtype=np.float32)
    for i, j in itertools.product(range(T), repeat=2):
        if i == j:
            continue
        pgo = projected_gradient_overlap(node_grads[i], node_grads[j])
        fcos = torch.cosine_similarity(node_grads[i], node_grads[j], dim=0).item()
        graph[i, j] = 0.5 * (pgo + fcos)
    return graph


def beam_search_order(graph: np.ndarray, beam: int = 8) -> List[int]:
    """TSP-like beam search that maximises cumulative positive interference."""

    T = graph.shape[0]
    beams: List[Tuple[Tuple[int, ...], float]] = [[(0,), 0]]
    for _ in range(1, T):
        new: List[Tuple[Tuple[int, ...], float]] = []
        for path, score in beams:
            for nxt in set(range(T)) - set(path):
                new.append((path + (nxt,), score + graph[path[-1], nxt]))
        new.sort(key=lambda x: x[1], reverse=True)
        beams = new[:beam]
    best_path, _best_score = beams[0]
    return list(best_path)


# -----------------------------------------------------------------------------
# 3.  Online Bubble-swap for negative interference correction
# -----------------------------------------------------------------------------

class OnlineBubbleSwap:
    """Implements the local O(T²) swap rule described in the paper."""

    def __init__(self, theta: float, window: int, K: int):
        self.theta = theta
        self.window = window
        self.K = K
        self.buffer: List[Any] = []
        self.num_swaps = 0

    # ------------------------------------------------------------------
    def update_buffer(self, batch):
        self.buffer.extend(batch)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]

    # ------------------------------------------------------------------
    def maybe_swap(
        self,
        tasks: List[Any],
        idx: int,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """If interference < theta, swap task idx and idx+1 (in-place)."""

        if idx + 1 >= len(tasks) or not self.buffer:
            return
        model.eval()
        # compute grad on buffer
        buf = random.sample(self.buffer, min(8, len(self.buffer)))
        buf_x = torch.stack([b[0] for b in buf]).to(device)
        buf_y = torch.stack([b[1] for b in buf]).to(device)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(buf_x), buf_y).backward()
        g1 = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])

        # single sample from *next* task
        nx, ny, *_ = next(iter(tasks[idx + 1].dataset))
        nx = nx.unsqueeze(0).to(device)
        ny = ny.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(nx), ny).backward()
        g2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])

        cos = (torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-12)).item()
        if cos < self.theta:
            tasks[idx], tasks[idx + 1] = tasks[idx + 1], tasks[idx]
            self.num_swaps += 1

# -----------------------------------------------------------------------------
# 4.  Training loop for a single continual task
# -----------------------------------------------------------------------------

def train_single_task(
    model: nn.Module,
    stream,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    """Standard supervised training for one task."""

    model.train()
    loader = DataLoader(
        stream.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    for _ in range(epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimiser.step()
        scheduler.step()
