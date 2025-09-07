
"""
train.py – model, scheduler and training utilities
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import networkx as nx

# -----------------------------------------------------------------------------
# 0.  Device helper
# -----------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1.  LoRA helpers & model loader
# -----------------------------------------------------------------------------

def get_lora_cfg(rank: int, alpha: int, dropout: float) -> LoraConfig:  # noqa: N802
    """Return a PEFT LoRA configuration object."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )


def load_model(
    name_or_path: str,
    quant_cfg: Dict[str, Any] | None,
    lora_cfg: LoraConfig,
):
    """Create a causal-LM model with optional 8-bit quantisation and LoRA."""
    kwargs: Dict[str, Any] = {}
    if quant_cfg and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(**quant_cfg)
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model = get_peft_model(model, lora_cfg)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# -----------------------------------------------------------------------------
# 2.  IATG scheduler
# -----------------------------------------------------------------------------

class IATGScheduler:  # noqa: D101 – documentation in paper
    def __init__(self, beam: int = 5, alpha: float = 1.0, beta: float = 1.0):
        self.beam, self.alpha, self.beta = beam, alpha, beta
        self.G = nx.DiGraph()
        self.grads: Dict[str, torch.Tensor] = {}
        self.features: Dict[str, torch.Tensor] = {}
        self.current_order: List[str] = []

    # ------------------------- internal helpers -------------------------
    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    @staticmethod
    def _proj(g_j: torch.Tensor, g_i: torch.Tensor) -> float:
        """Scalar length of the projection of g_i onto g_j (normalised)."""
        proj = (torch.dot(g_i, g_j) / (torch.dot(g_j, g_j) + 1e-12)) * g_j
        return proj.norm().item() / (g_i.norm().item() + 1e-12)

    def _edge_cost(self, a: str, b: str) -> float:
        if self.G.has_edge(a, b):
            return self.G[a][b]["weight"]
        return 1e3  # disconnected penalty

    def _beam_search(self) -> List[str]:
        tasks = list(self.G.nodes)
        if not tasks:
            return []
        beam: List[Tuple[float, List[str]]] = [(0.0, [t]) for t in tasks]
        for _ in range(len(tasks) - 1):
            new_beam: List[Tuple[float, List[str]]] = []
            for cost, path in beam:
                for cand in tasks:
                    if cand in path:
                        continue
                    extra = self._edge_cost(path[-1], cand)
                    new_beam.append((cost + extra, path + [cand]))
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[: self.beam]
        best = beam[0][1]
        self.current_order = best
        return best

    # ------------------------- public API -------------------------

    def add_task(self, tid: str, g: torch.Tensor, f: torch.Tensor) -> List[str]:
        """Register a new task and recompute the recommended curriculum order.

        The incoming gradient (g) may reside on CUDA when the caller performed
        backward() on a GPU-resident model.  We explicitly move it to CPU so
        that all stored tensors live on the same device – this avoids the
        cross-device dot-product error that surfaced in CI.
        """
        g_cpu = g.detach().cpu()
        f_cpu = f.detach().cpu()

        self.grads[tid] = g_cpu
        self.features[tid] = f_cpu
        self.G.add_node(tid)
        for j in self.grads:
            if j == tid:
                continue
            cost = (
                self.alpha * self._proj(self.grads[j], g_cpu)
                - self._cos(g_cpu, self.grads[j])
                - self.beta * torch.linalg.norm(f_cpu - self.features[j]).item()
            )
            self.G.add_edge(j, tid, weight=cost)
        return self._beam_search()

# -----------------------------------------------------------------------------
# 3.  Training loop utilities
# -----------------------------------------------------------------------------

def train_single_epoch(model, loader, optimiser, scaler, cfg):
    """One epoch with gradient accumulation & AMP."""
    model.train()
    accum = 0
    for batch in loader:
        batch = {k: v.to(_DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type=_DEVICE.type, enabled=cfg.get("fp16", False)):
            loss = model(**batch).loss / cfg["grad_accum"]
        scaler.scale(loss).backward()
        accum += 1
        if accum == cfg["grad_accum"]:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(True)
            accum = 0
