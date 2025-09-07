"""src/train.py
Utility, model- and training-related code for the IATG + CC-LoRA experiments.
All heavy lifting (backbone loading, LoRA helpers, scheduler, gradient
capture, etc.) lives in this file so that the other modules can stay tiny.
"""
from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path
from typing import List, Mapping

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------------------
#  Constants & helpers
# ----------------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_FP16 = torch.float16 if torch.cuda.is_available() else torch.float32


# ----------------------------------------------------------------------------------
#  Reproducibility / misc. utils
# ----------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def elapsed(msg: str):
    t0 = time.perf_counter()
    yield
    dur = time.perf_counter() - t0
    print(f"[TIME] {msg}: {dur:6.2f} s")


# ----------------------------------------------------------------------------------
#  Backbone / PEFT helpers
# ----------------------------------------------------------------------------------

def load_backbone(name: str, *, quant: bool = False):
    """Load an HF model either in full precision or 8-bit (bitsandbytes)."""
    if quant and torch.cuda.is_available():
        qc = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", quantization_config=qc)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
        model.to(DEVICE)

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    return model, tok


def attach_cls_head(backbone: nn.Module, n_classes: int = 3) -> nn.Module:
    """Freeze backbone parameters and add a tiny classification head."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = nn.Linear(backbone.config.hidden_size, n_classes).to(backbone.device)  # type: ignore[attr-defined]
    backbone.classifier_head = head  # type: ignore[attr-defined]
    return backbone


# ----------------------------------------------------------------------------------
#  Scheduler (IATG) – same implementation as paper, minor refactor only
# ----------------------------------------------------------------------------------

class IATGScheduler:
    """Gradient/feature-aware beam-search task-ordering solver."""

    def __init__(self, beam: int = 5, alpha: float = 1.0, beta: float = 1.0):
        self.beam, self.alpha, self.beta = beam, alpha, beta
        self.G: nx.DiGraph = nx.DiGraph()
        self.grad: dict[str, torch.Tensor] = {}
        self.feat: dict[str, torch.Tensor] = {}
        self.curr: list[str] = []

    # ----- helpers -----
    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.float(), b.float(), dim=0, eps=1e-8).item()

    @staticmethod
    def _proj_norm(gj: torch.Tensor, gi: torch.Tensor) -> float:
        proj = (gi @ gj) / (gj @ gj + 1e-10) * gj
        return (proj.norm() / (gi.norm() + 1e-10)).item()

    def _cost(self, i: str, j: str) -> float:
        gi, gj = self.grad[i], self.grad[j]
        fi, fj = self.feat[i], self.feat[j]
        return (
            self.alpha * self._proj_norm(gj, gi)
            - self._cos(gi, gj)
            - self.beta * torch.linalg.norm(fi - fj).item()
        )

    # ----- public API -----

    def add_task(self, tid: str, g: torch.Tensor, f: torch.Tensor) -> List[str]:
        g, f = g.detach().cpu().float(), f.detach().cpu().float()
        self.grad[tid] = g
        self.feat[tid] = f
        self.G.add_node(tid)
        for other in self.G.nodes:
            if other == tid:
                continue
            self.G.add_edge(other, tid, weight=self._cost(other, tid))
        self.curr = self._beam() if len(self.G) else []
        return self.curr

    def _beam(self) -> List[str]:
        tasks = list(self.G.nodes)
        beam: list[tuple[float, list[str]]] = [(0.0, [t]) for t in tasks]
        for _ in range(len(tasks) - 1):
            new: list[tuple[float, list[str]]] = []
            for cost, path in beam:
                for cand in tasks:
                    if cand in path:
                        continue
                    new.append((cost + self.G[path[-1]][cand]["weight"], path + [cand]))
            beam = sorted(new, key=lambda x: x[0])[: self.beam]
        return beam[0][1]


# ----------------------------------------------------------------------------------
#  Gradient helpers & small utilities used during probing / PEFT training
# ----------------------------------------------------------------------------------

class GradTracker:
    """Flatten gradients from all *trainable* parameters (used by IATG)."""

    @staticmethod
    def capture(model: nn.Module) -> torch.Tensor:
        vecs: list[torch.Tensor] = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                vecs.append(p.grad.detach().float().flatten())
        if not vecs:
            raise RuntimeError("No gradients captured – check requires_grad flags.")
        return torch.cat(vecs)


# separate tiny helper so unit tests can import it directly

def forward_back_loss(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch)
    out.loss.backward()
    return out.loss.detach()


# ----------------------------------------------------------------------------------
#  PEFT task-level trainer (used by EXP-2, but generally useful)
# ----------------------------------------------------------------------------------

def train_lora_task(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    *,
    epochs: int = 1,
    grad_accum: int = 1,
):
    """LoRA fine-tuning with optional gradient accumulation & AMP."""
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            global_step += 1
            if global_step % grad_accum == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(True)
