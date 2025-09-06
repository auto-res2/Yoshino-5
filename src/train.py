"""src/train.py
Contains all training–related code: model/optimizer preparation, the CURIOUS
scheduler and the VisionTrainer.  The trainer is completely parameterised by a
GlobalConfig object that is created in src/main and handed over so no global
state is required here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
import timm
from peft import get_peft_model, LoraConfig
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from codecarbon import OfflineEmissionsTracker

from .evaluate import ContinualMetrics
from .preprocess import ImagenetteTasks

# -----------------------------------------------------------------------------
#                               Helper
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    """Reproducibility helper for python / numpy / pytorch."""
    import random as _random
    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
#                           CURIOUS  Scheduler
# -----------------------------------------------------------------------------

class CuriousScheduler:
    """Simplified CURIOUS scheduler – computes utilities from cheap surrogates
    and draws the next task using a soft Thompson sampling variant.  For the
    reference implementation we use uniform random surrogates; plug-ins for
    OTDD etc. can directly modify  _utility()."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        flops_budget_ratio: float = 0.03,
        beam_width: int = 8,
        temperature: float = 1.0,
    ):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.flops_budget_ratio = flops_budget_ratio
        self.beam_width = beam_width
        self.temperature = temperature
        self._seen_tasks: List[int] = []
        self._stability: Dict[int, float] = {}
        self._grad_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def order_stream(self, tasks: List[dict]):
        remaining = list(range(len(tasks)))
        while remaining:
            if not self._seen_tasks:
                nxt = remaining.pop(0)
            else:
                utilities = [self._utility(cid) for cid in remaining]
                probs = torch.softmax(torch.tensor(utilities) / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                nxt = remaining.pop(idx)
            self._seen_tasks.append(nxt)
            yield nxt

    # ------------------------------------------------------------------
    def _utility(self, task_id: int):
        s_d = random.random()
        s_g = random.random()
        s_s = self._stability.get(task_id, 0.0)
        return self.alpha * s_d + self.beta * s_g + self.gamma * s_s

    # ------------------------------------------------------------------
    def update_after_task(self, task_id: int, forget_score: float, grad_vec: torch.Tensor):
        self._stability[task_id] = forget_score
        self._grad_cache[task_id] = grad_vec.detach().cpu()

# -----------------------------------------------------------------------------
#                               Vision Trainer
# -----------------------------------------------------------------------------

@dataclass
class GlobalConfig:  # local copy so train.py is self-contained w.r.t typing
    work_dir: Path
    device: str
    # optimiser
    lr_vision: float
    betas: tuple
    weight_decay: float
    eps: float
    # lora
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    # curious
    curious_alpha: float
    curious_beta: float
    curious_gamma: float
    flops_budget_ratio: float
    # training
    epochs_per_task: int
    batch_size_vision: int
    precision: str

class VisionTrainer:
    """End-to-end continual learning for Imagenette pseudo-stream."""

    def __init__(self, tasks: ImagenetteTasks, cfg: GlobalConfig, seed: int):
        set_seed(seed)
        self.cfg = cfg
        self.tasks = tasks
        self.device = cfg.device
        # ----------------------- Model + LoRA ---------------------------------
        base = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=10
        ).to(self.device)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["qkv", "proj"],
        )
        self.model = get_peft_model(base, lora_cfg).train()
        # ----------------------- Optimiser  &  Scheduler -----------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr_vision,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        approx_steps = (
            len(tasks.tasks)
            * cfg.epochs_per_task
            * (len(tasks.tasks[0]["train"]) // cfg.batch_size_vision)
        )
        self.lr_sched = get_cosine_schedule_with_warmup(
            self.optimizer, int(0.2 * approx_steps), approx_steps
        )
        self.scaler = torch.cuda.amp.GradScaler()
        # ----------------------- CURIOUS Scheduler -----------------------------
        self.scheduler = CuriousScheduler(
            cfg.curious_alpha,
            cfg.curious_beta,
            cfg.curious_gamma,
            cfg.flops_budget_ratio,
        )
        # ----------------------- Energy tracker --------------------------------
        self.energy_tracker = OfflineEmissionsTracker(
            project_name="vision", output_dir=str(cfg.work_dir)
        )

    # ---------------------------------------------------------------------
    def train_stream(self):
        n_tasks = len(self.tasks.tasks)
        metrics = ContinualMetrics(n_tasks=n_tasks, n_classes=10, device=self.device)
        self.energy_tracker.start()
        for t_idx in self.scheduler.order_stream(self.tasks.tasks):
            task = self.tasks.tasks[t_idx]
            dl_train = self.tasks.dataloader_for(task["train"])
            self._train_task(dl_train)
            # evaluate after task
            for ev_idx in range(t_idx + 1):
                ev_dl = self.tasks.dataloader_for(self.tasks.tasks[ev_idx]["test"])
                acc = self._eval(ev_dl)
                metrics.acc_matrix[t_idx, ev_idx] = acc
            # simplistic update
            self.scheduler.update_after_task(t_idx, forget_score=0.0, grad_vec=torch.zeros(1))
        self.energy_tracker.stop()
        return metrics.final_results()

    # ------------------------------------------------------------------
    def _train_task(self, dataloader: DataLoader):
        dtype = torch.float16 if self.cfg.precision == "fp16" else torch.bfloat16
        for _ in range(self.cfg.epochs_per_task):
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=dtype):
                    logits = self.model(imgs)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_sched.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, dataloader: DataLoader):
        self.model.eval()
        correct, total = 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = self.model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
        self.model.train()
        return correct / total
