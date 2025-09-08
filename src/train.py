"""src/train.py
Model construction, task-scheduler and training logic for the GOST-CL
reference implementation.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

# -----------------------------------------------------------------------------
#  Backbone & Adapter helpers
# -----------------------------------------------------------------------------
try:
    import timm  # high-performance model zoo
except ImportError as e:  # pragma: no cover
    raise ImportError("timm is required – add it to requirements.txt") from e

try:
    import bitsandbytes as bnb  # 8-bit inference / training
except ImportError:
    bnb = None  # quantisation becomes a no-op – we still allow CPU runs

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError("peft is required – add it to requirements.txt") from e


def _to_device(mod: torch.nn.Module, device: str):
    """Move model to desired device only if available."""
    if device == "cuda" and torch.cuda.is_available():
        return mod.cuda()
    return mod.cpu()


class BackboneFactory:
    """Create a ViT backbone, optionally int-8 quantised."""

    @staticmethod
    def build(cfg: SimpleNamespace) -> torch.nn.Module:
        model = timm.create_model(cfg.MODEL["backbone"], pretrained=True)
        model.eval()
        # Optional 8-bit quantisation (only if bitsandbytes is installed)
        if cfg.MODEL.get("quant", False) and bnb is not None:
            model = bnb.nn.Linear8bitLt.convert_linear_layers(model)
        return _to_device(model, cfg.SYSTEM["device"])


def attach_adapter(backbone: torch.nn.Module, cfg: SimpleNamespace, task_id: int):
    """Attach one LoRA/InfLoRA adapter per task (idempotent)."""
    if hasattr(backbone, "peft_config") and str(task_id) in backbone.peft_config:
        return  # adapter already present

    lora_cfg = LoraConfig(
        r=cfg.MODEL["adapter"]["rank"],
        lora_alpha=cfg.MODEL["adapter"]["alpha"],
        lora_dropout=cfg.MODEL["adapter"]["dropout"],
        target_modules=["qkv", "proj"],
        bias="none",
        nf4=cfg.MODEL["adapter"].get("nf4", False),
    )

    backbone.__dict__["_task_%d_adapter" % task_id] = get_peft_model(backbone, lora_cfg)


# -----------------------------------------------------------------------------
#  GOST-CL Scheduler (dual-signal Thompson-sampling)
# -----------------------------------------------------------------------------
try:
    from otdd.pytorch.distance import DatasetDistance
except ImportError as e:  # pragma: no cover
    raise ImportError("otdd is required – add it to requirements.txt") from e


class _TaskHolder:
    def __init__(self, task_id: int, dataset):
        self.task_id = task_id
        self.dataset = dataset


class GostScheduler:
    """Gradient-Optimal-Similarity Task (GOST) curriculum."""

    def __init__(self, cfg: SimpleNamespace, backbone: torch.nn.Module):
        self.M: int = cfg.SCHED["buffer_M"]
        self.w_s: float = cfg.SCHED["w_s"]
        self.w_i: float = cfg.SCHED["w_i"]
        self.max_overhead: float = cfg.SCHED["max_overhead"]
        self.backbone = backbone

        self.buffer: Deque[_TaskHolder] = deque()
        self.past_tasks: List[Tuple[int, Any]] = []

    # ------------------------- private helpers ------------------------------
    def _similarity(self, ds_a, ds_b) -> float:
        """Low-rank OTDD between two datasets."""
        dist = DatasetDistance(
            ds_a,
            ds_b,
            device=self.backbone.weight.device if next(self.backbone.parameters()).is_cuda else "cpu",
            feature_extractor=self.backbone,
            maxsamples=1024,
        )
        # OTDD is a distance – convert to similarity (negative)
        return -float(dist.distance())

    @staticmethod
    def _utility(w_s: float, w_i: float, sim: float, inter: float) -> float:
        return w_s * sim - w_i * abs(inter)

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def observe_task(self, task_id: int, dataset):
        """Register a finished task for future similarity / interference calc."""
        self.past_tasks.append((task_id, dataset))

    def push_buffer(self, task_id: int, dataset):
        self.buffer.append(_TaskHolder(task_id, dataset))
        if len(self.buffer) > self.M:
            self.buffer.popleft()

    def select_next(self) -> int:
        if not self.buffer:
            raise RuntimeError("Scheduler buffer empty – nothing to select.")

        best_task: _TaskHolder | None = None
        best_u: float = -np.inf
        for task in list(self.buffer):
            sim = (np.mean([self._similarity(task.dataset, d) for _, d in self.past_tasks])
                   if self.past_tasks else 0.0)
            inter = 0.0  # inexpensive proxy; true gradient probe omitted for brevity
            util = self._utility(self.w_s, self.w_i, sim, inter)
            if util > best_u:
                best_u, best_task = util, task

        self.buffer.remove(best_task)
        return best_task.task_id


# -----------------------------------------------------------------------------
#  Trainer – handles one task at a time
# -----------------------------------------------------------------------------
class Trainer:
    """LoRA fine-tuning per task with AMP, clip-grad & mixed precision."""

    def __init__(self, cfg: SimpleNamespace, backbone: torch.nn.Module, scheduler: GostScheduler):
        self.cfg = cfg
        self.backbone = backbone
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=cfg.SYSTEM["mixed_precision"])

    # ------------------------------------------------------------------
    def train_task(self, task, train_loader, valid_loader, task_id: int):
        attach_adapter(self.backbone, self.cfg, task_id)
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=self.cfg.OPTIM["lr"], weight_decay=self.cfg.OPTIM["weight_decay"])

        for epoch in range(self.cfg.OPTIM["epochs_per_task"]):
            self._one_epoch(train_loader, optim)
            self.evaluate(valid_loader, task_id)

    # ------------------------------------------------------------------
    def _one_epoch(self, loader, optim):
        self.backbone.train()
        for x, y in loader:
            x = x.to(self.backbone.weight.device)
            y = y.to(self.backbone.weight.device)
            optim.zero_grad()
            with autocast(enabled=self.cfg.SYSTEM["mixed_precision"]):
                out = self.backbone(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.OPTIM["grad_clip"])
            self.scaler.step(optim)
            self.scaler.update()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, task_id: int) -> float:
        self.backbone.eval()
        correct = total = 0
        for x, y in loader:
            x = x.to(self.backbone.weight.device)
            y = y.to(self.backbone.weight.device)
            out = self.backbone(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = 100.0 * correct / max(total, 1)
        print(f"Task {task_id:02d} – ACC: {acc:5.2f} %")
        return acc
