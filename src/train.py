"""src/train.py
-------------------------------------------------------------------
All training-related logic: model loading, LoRA preparation, task
scheduler, ordering baselines, continual-learning loop.
"""
from __future__ import annotations

import os
import math
import time
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import transformers
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb  # noqa: F401 – required for 8-bit loading

# ------------------------------------------------------------------
# Safety: make sure script does not crash on non-GPU machines.
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories -------------------------------------------------------
RESULT_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def set_global_seed(seed: int):
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------
# Model loading helpers (LoRA + optional 8-bit)
# ------------------------------------------------------------------

def _prepare_lora(model: torch.nn.Module, target_modules: List[str], cfg: Dict):
    l_cfg = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg["lora_dropout"],
        bias="none",
    )
    return get_peft_model(model, l_cfg)


def load_llm(cfg: Dict):
    """Load causal-LM backbone with optional 8-bit weights + LoRA."""
    print("[INFO] Loading LLM:", cfg["llm_ckpt"])
    model_kwargs = {}
    if cfg.get("load_in_8bit", True):
        model_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["llm_ckpt"], **model_kwargs
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["llm_ckpt"])
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token exists
    # LoRA
    model = _prepare_lora(model, ["q_proj", "v_proj"], cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def load_siglip(cfg: Dict):
    """Vision encoder with LoRA adapters."""
    model = transformers.SiglipModel.from_pretrained(
        cfg["vision_ckpt"],
        variant="base",
        torch_dtype=torch.float16 if cfg.get("fp16", True) else torch.float32,
    )
    processor = transformers.SiglipProcessor.from_pretrained(cfg["vision_ckpt"])
    model = _prepare_lora(model, ["query", "value"], cfg)
    return model, processor

# ------------------------------------------------------------------
# IATG  – Interference-Aware Task Graph Scheduler
# ------------------------------------------------------------------
class IATGScheduler:
    """Online task-ordering scheduler that minimises gradient interference."""

    def __init__(self, beam: int = 5, alpha: float = 1.0, beta: float = 1.0):
        import networkx as nx  # heavy import deferred

        self.beam = beam
        self.alpha = alpha
        self.beta = beta
        self.G = nx.DiGraph()
        self.grad_store: Dict[str, torch.Tensor] = {}
        self.feat_store: Dict[str, torch.Tensor] = {}
        self.current_order: List[str] = []

    # ------------- internal helpers --------------------------------
    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    @staticmethod
    def _proj_norm(g_j: torch.Tensor, g_i: torch.Tensor) -> float:
        proj = (torch.dot(g_i, g_j) / (torch.dot(g_j, g_j) + 1e-12)) * g_j
        return proj.norm().item() / (g_i.norm().item() + 1e-12)

    # ------------- public API --------------------------------------
    def add_task(self, tid: str, grad: torch.Tensor, feat_vec: torch.Tensor) -> List[str]:
        """Insert new task and recompute best ordering."""
        self.grad_store[tid] = grad.detach().cpu()
        self.feat_store[tid] = feat_vec.detach().cpu()

        self.G.add_node(tid)
        for j in self.grad_store:
            if j == tid:
                continue
            g_j = self.grad_store[j]
            sim_grad = self._cos(grad, g_j)
            interf = self._proj_norm(g_j, grad)
            sim_feat = torch.linalg.norm(feat_vec - self.feat_store[j]).item()
            cost = self.alpha * interf - self.beta * sim_feat - sim_grad
            self.G.add_edge(j, tid, weight=cost)

        self.current_order = self._beam_search()
        return self.current_order

    # ----------------------------------------------------------------
    def _beam_search(self) -> List[str]:
        tasks = list(self.G.nodes)
        if not tasks:
            return []
        best_orders = [[tasks[0]]]
        for t in tasks[1:]:
            new_beam: List[Tuple[float, List[str]]] = []
            for order in best_orders:
                for pos in range(len(order) + 1):
                    cand = order[:pos] + [t] + order[pos:]
                    cost = self._path_cost(cand)
                    new_beam.append((cost, cand))
            new_beam.sort(key=lambda x: x[0])
            best_orders = [c for _, c in new_beam[: self.beam]]
        return best_orders[0]

    def _path_cost(self, order: List[str]) -> float:
        cost = 0.0
        for i in range(len(order) - 1):
            if self.G.has_edge(order[i], order[i + 1]):
                cost += self.G[order[i]][order[i + 1]]["weight"]
        return cost

# ------------------------------------------------------------------
# Simple baseline orderings
# ------------------------------------------------------------------
class OrderingFactory:
    @staticmethod
    def chronological(stream: List[str]) -> List[str]:
        return stream

    @staticmethod
    def random(stream: List[str]) -> List[str]:
        o = stream.copy()
        random.shuffle(o)
        return o

# ------------------------------------------------------------------
# Continual-learning training loop
# ------------------------------------------------------------------
class ContinualLearner:
    """Train a single model on a task stream following the supplied ordering."""

    def __init__(
        self,
        train_cfg: Dict,
        model: torch.nn.Module,
        tokenizer,
        ordering_name: str,
        ordering: List[str],
        seed: int,
    ):
        self.cfg = train_cfg
        self.ordering = ordering
        self.seed = seed
        self.ordering_name = ordering_name

        set_global_seed(seed)

        self.model = model.to(DEVICE)
        self.scaler = GradScaler(enabled=train_cfg.get("fp16", True))
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=train_cfg["weight_decay"],
        )
        self.pad_id = tokenizer.pad_token_id
        self.metrics_log: List[Dict] = []
        self.task_acc_at_learn: Dict[str, float] = {}

    # --------------------------------------------------------------
    #  Evaluation helpers
    # --------------------------------------------------------------
    @torch.inference_mode()
    def _compute_accuracy(self, loader) -> float:
        self.model.eval()
        correct = 0.0
        total = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = self.model(**batch)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).float().sum().item()
            total += batch["labels"].numel()
        return correct / max(total, 1e-12)

    # --------------------------------------------------------------
    def _train_task(self, task_name: str, train_loader, val_loader):
        epochs = self.cfg["epochs_per_task"]
        best_val = -1.0
        for ep in range(epochs):
            self.model.train()
            accum = 0
            t0 = time.time()
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with autocast(enabled=self.cfg.get("fp16", True)):
                    loss = self.model(**batch).loss / self.cfg["grad_accum"]
                self.scaler.scale(loss).backward()
                accum += 1
                if accum == self.cfg["grad_accum"]:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["clip_grad_norm"]
                    )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    accum = 0
            val_acc = self._compute_accuracy(val_loader)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(RESULT_DIR, f"{task_name}_best.pt"),
                )
            print(
                f"[Task {task_name}] epoch {ep + 1}/{epochs} val_acc={val_acc:.4f} "
                f"(best={best_val:.4f}) in {time.time() - t0:.1f}s"
            )
        return best_val

    # --------------------------------------------------------------
    def run_stream(self, dataloaders: Dict[str, Tuple]) -> Tuple[float, float]:
        for task in self.ordering:
            print("\n=========== Training task:", task, "============")
            tr, vl, te = dataloaders[task]
            _ = self._train_task(task, tr, vl)

            # Record accuracy on all seen tasks
            accs = [self._compute_accuracy(dataloaders[t][2]) for t in self.ordering if t in dataloaders]
            self.metrics_log.append({"task": task, "avg_acc": np.mean(accs)})
            self.task_acc_at_learn[task] = accs[-1]

        final_acc = np.mean([self._compute_accuracy(dataloaders[t][2]) for t in self.ordering])
        bwt = np.mean(
            [self._compute_accuracy(dataloaders[t][2]) - self.task_acc_at_learn[t] for t in self.ordering]
        )
        return final_acc, bwt
