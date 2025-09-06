"""src/train.py
Minor bug-fixes & path updates.
(See header of file for previous notes.)

Changes in this patch
---------------------
1.  AttrDict.to_dict now converts every Path instance to str **recursively** so that the
    resulting object is JSON-serialisable.  This fixes the crash that happened in
    ``main.py`` when executing ``json.dumps(cfg.to_dict())``.
2.  No other functional changes – all existing behaviour is preserved.
"""
from __future__ import annotations

import os, random, math, inspect, time, json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any, Union

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from fvcore.nn import FlopCountAnalysis
from codecarbon import OfflineEmissionsTracker
import torch.nn.functional as F

# --------------------------------------------------------------------------
# helpers & utilities -------------------------------------------------------
# --------------------------------------------------------------------------
class AttrDict(SimpleNamespace):
    """Recursively turn a (nested) mapping into an object with attribute access.

    Additionally, ``to_dict`` now converts ``pathlib.Path`` objects to
    ``str`` so that the output can be safely serialised with ``json``.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = AttrDict(**v)  # recursion
            # convert path-like strings automatically so downstream code stays unchanged
            if isinstance(v, str) and (k.endswith('_dir') or k.endswith('_path') or k.endswith('_root')):
                v = Path(v)
            super().__setattr__(k, v)

    # ------------------------------------------------------------------
    def _serialise(self, obj: Any) -> Any:  # pylint: disable=R0201
        """Helper used by :py:meth:`to_dict` – makes *everything* JSON-friendly."""
        if isinstance(obj, AttrDict):
            return {k: self._serialise(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: self._serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialise(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a *pure* Python dict that can be passed to :pyfunc:`json.dumps`."""
        return {k: self._serialise(v) for k, v in self.__dict__.items()}


def set_seed(sd: int):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


def device():
    """Return best available device.  Previously the code crashed on CPU-only hosts."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# model zoo (LoRA wrapped) ---------------------------------------------------
# --------------------------------------------------------------------------
import timm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def vit_base_with_lora(cfg):
    """Vision Transformer backbone with LoRA adapters."""
    base = timm.create_model(cfg.vit_name.split("/")[-1], pretrained=True, num_classes=0)
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=["qkv", "proj"],
    )
    return get_peft_model(base, lora_cfg)


def distilgpt2_with_lora(cfg):
    """NLP backbone example (unused in the current vision experiments)."""
    base = AutoModelForCausalLM.from_pretrained(cfg.gpt_name, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=["q_proj", "v_proj", "fc_in", "fc_out"],
    )
    return get_peft_model(base, lora_cfg)

# --------------------------------------------------------------------------
# schedulers ----------------------------------------------------------------
# --------------------------------------------------------------------------
import ot

# optional GPU Sinkhorn ------------------------------------------------------
try:
    from ot.gpu import sinkhorn as _gpu_sinkhorn  # type: ignore
    _HAS_GPU_SINKHORN = True
except Exception:  # pragma: no cover – any import issue → use CPU fallback
    _HAS_GPU_SINKHORN = False


class SimilarityCache:
    def __init__(self):
        self.otdd: Dict[tuple, float] = {}
        self.grads: Dict[int, torch.Tensor] = {}


class CuriousScheduler:
    """Implementation of CURIOUS (Compute-budgeted Utility-aware Reordering)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.alpha, self.beta, self.gamma = cfg.curious.alpha, cfg.curious.beta, cfg.curious.gamma
        self.tau = cfg.curious.temperature
        self.cache = SimilarityCache()
        self.stability: Dict[int, float] = {}
        self.seen: List[int] = []

    # ------------------------------------------------------------------
    def _cos_conflict(self, g_new: torch.Tensor, g_ref: torch.Tensor) -> float:
        return -F.cosine_similarity(g_new.flatten(), g_ref.flatten(), dim=0).item()

    # ------------------------------------------------------------------
    def _otdd(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
        key = (id(feat_a), id(feat_b))
        if key in self.cache.otdd:
            return self.cache.otdd[key]

        # Compute pair-wise ground cost ------------------------------------------------
        Xa = feat_a.detach().cpu().float().numpy()  # POT expects NumPy arrays
        Xb = feat_b.detach().cpu().float().numpy()
        M = ot.dist(Xa, Xb, metric="euclidean")  # (na, nb)
        a = np.full((Xa.shape[0],), 1.0 / Xa.shape[0])
        b = np.full((Xb.shape[0],), 1.0 / Xb.shape[0])

        if _HAS_GPU_SINKHORN and feat_a.is_cuda:
            # GPU path using POT's CUDA backend – significantly faster
            T = _gpu_sinkhorn(
                torch.tensor(a, device=feat_a.device),
                torch.tensor(b, device=feat_a.device),
                torch.tensor(M, device=feat_a.device),
                reg=0.05,
                numItermax=100,
            )
            d = (T * torch.tensor(M, device=feat_a.device)).sum().item()
        else:
            # Portable CPU fallback ----------------------------------------------------
            T = ot.sinkhorn(a, b, M, reg=0.05, numItermax=100)
            d = float((T * M).sum())

        self.cache.otdd[key] = d
        return d

    # ------------------------------------------------------------------
    def _utility(self, cid: int, feat_new: torch.Tensor, grad_new: torch.Tensor) -> float:
        s_d = min((self._otdd(feat_new, self.cache.grads[i][1]) for i in self.seen), default=0.0)
        s_g = min((self._cos_conflict(grad_new, self.cache.grads[i][0]) for i in self.seen), default=0.0)
        s_s = self.stability.get(cid, 0.0)
        return self.alpha * s_d + self.beta * s_g + self.gamma * s_s

    # ------------------------------------------------------------------
    def propose_next(self, remaining: List[int], feat_bank: Dict[int, torch.Tensor], grad_bank: Dict[int, torch.Tensor]):
        if not self.seen:
            choice = remaining[0]
            self.seen.append(choice)
            return choice
        utilities = [self._utility(cid, feat_bank[cid], grad_bank[cid]) for cid in remaining]
        probs = torch.softmax(torch.tensor(utilities) / self.tau, dim=0)
        idx = torch.multinomial(probs, 1).item()
        choice = remaining[idx]
        self.seen.append(choice)
        return choice

    # ------------------------------------------------------------------
    def update_after_task(self, task_id: int, feat: torch.Tensor, grad: torch.Tensor, acc_drop: float):
        self.cache.grads[task_id] = (grad.detach().cpu(), feat.detach().cpu())
        self.stability[task_id] = acc_drop


class RandomScheduler:
    def order(self, n_tasks):
        return random.sample(range(n_tasks), n_tasks)


class ChronoScheduler:
    def order(self, n_tasks):
        return list(range(n_tasks))


class DotddScheduler:
    """Greedy ordering based on data similarity (dOTDD)."""

    def __init__(self, feat_bank):
        self.feat_bank = feat_bank

    def order(self, task_ids: List[int]):
        remaining = task_ids.copy()
        order: List[int] = []
        while remaining:
            if not order:
                order.append(remaining.pop(0))
                continue
            last = order[-1]
            dists = [
                (
                    cid,
                    torch.norm(self.feat_bank[last].mean(0) - self.feat_bank[cid].mean(0)).item(),
                )
                for cid in remaining
            ]
            cid = min(dists, key=lambda x: x[1])[0]
            order.append(cid)
            remaining.remove(cid)
        return order


class GradOnlyScheduler:
    """Greedy ordering based on gradient conflict only."""

    def __init__(self, grad_bank):
        self.grad_bank = grad_bank

    def order(self, task_ids: List[int]):
        remaining = task_ids.copy()
        order: List[int] = []
        while remaining:
            if not order:
                order.append(remaining.pop(0))
                continue
            last = order[-1]
            confs = [
                (
                    cid,
                    -F.cosine_similarity(
                        self.grad_bank[last].flatten(), self.grad_bank[cid].flatten(), dim=0
                    ).item(),
                )
                for cid in remaining
            ]
            cid = min(confs, key=lambda x: x[1])[0]
            order.append(cid)
            remaining.remove(cid)
        return order

# --------------------------------------------------------------------------
# Continual learner ---------------------------------------------------------
# --------------------------------------------------------------------------
from evaluate import CLMatrix  # fixed import


class ContinualLearner:
    """Vision-only continual learner – loosely adapted from the monolithic script."""

    def __init__(self, stream, cfg: AttrDict, scheduler_name: str):
        self.cfg, self.stream = cfg, stream
        self.dev = device()

        # create backbone + heads ------------------------------------------------
        self.model = vit_base_with_lora(cfg).to(self.dev)
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(768, 10) for _ in range(len(stream.tasks))
        ]).to(self.dev)

        # optimiser -------------------------------------------------------------
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        self.scaler = GradScaler()
        self.cl_matrix = CLMatrix(len(stream.tasks))

        # pick scheduler --------------------------------------------------------
        self.sched: Any
        self._init_scheduler(scheduler_name)

    # ------------------------------------------------------------------
    def _init_scheduler(self, name: str):
        if name == "CURIOUS":
            self.sched = CuriousScheduler(self.cfg)
        elif name == "Random":
            self.sched = RandomScheduler()
        elif name == "Chrono":
            self.sched = ChronoScheduler()
        else:
            # For dOTDD & GradOnly we need feature/gradient banks – they will be built later
            self.sched = name  # temporary placeholder

    # ------------------------------------------------------------------
    def _forward(self, x, task_id):
        feats = self.model(x)
        return self.heads[task_id](feats)

    # ------------------------------------------------------------------
    def _compute_feat_grad(self, loader: torch.utils.data.DataLoader, task_id: int):
        x, y = next(iter(loader))
        x, y = x.to(self.dev), y.to(self.dev)
        self.opt.zero_grad()
        with autocast(dtype=torch.float16 if self.dev.type == "cuda" else torch.float32):
            feats = self.model(x)
            logits = self.heads[task_id](feats)
            loss = F.cross_entropy(logits, y)
        self.scaler.scale(loss).backward()
        g = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        return feats.detach(), g.detach()

    # ------------------------------------------------------------------
    def train(self):
        n_tasks = len(self.stream.tasks)
        task_ids = list(range(n_tasks))

        # ------------------------------------------------------------------
        # pre-compute banks (needed for all schedulers except Chrono/Random)
        feat_bank, grad_bank = {}, {}
        for tid in task_ids:
            dl = self.stream.loader(self.stream.tasks[tid]["train"])
            f, g = self._compute_feat_grad(dl, tid)
            feat_bank[tid], grad_bank[tid] = f, g

        # if placeholder scheduler, instantiate with banks now -------------
        if isinstance(self.sched, str):
            if self.sched == "dOTDD":
                self.sched = DotddScheduler(feat_bank)
            elif self.sched == "GradOnly":
                self.sched = GradOnlyScheduler(grad_bank)
            else:
                raise ValueError(f"Unknown scheduler {self.sched}")

        # create order -----------------------------------------------------------
        if isinstance(self.sched, (DotddScheduler, GradOnlyScheduler)):
            order = self.sched.order(task_ids)
        elif isinstance(self.sched, (RandomScheduler, ChronoScheduler)):
            order = self.sched.order(n_tasks)
        else:  # CURIOUS
            remaining, order = task_ids.copy(), []
            while remaining:
                cid = self.sched.propose_next(remaining, feat_bank, grad_bank)
                order.append(cid)
                remaining.remove(cid)

        # ------------------------------------------------------------------
        tracker = OfflineEmissionsTracker(project_name="exp", output_dir=str(self.cfg.paths.work_dir))
        tracker.start()

        for seen_idx, task_id in enumerate(order):
            train_loader = self.stream.loader(self.stream.tasks[task_id]["train"])
            for _ in range(self.cfg.epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(self.dev), y.to(self.dev)
                    self.opt.zero_grad()
                    with autocast(dtype=torch.float16 if self.dev.type == "cuda" else torch.float32):
                        logits = self._forward(x, task_id)
                        loss = F.cross_entropy(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()

            # quick evaluation --------------------------------------------------
            for eval_id in range(seen_idx + 1):
                ev_task = order[eval_id]
                acc = self._evaluate_task(ev_task)
                self.cl_matrix.update(seen_idx, eval_id, acc)

            # scheduler feedback ------------------------------------------------
            if isinstance(self.sched, CuriousScheduler):
                self.sched.update_after_task(
                    task_id, feat_bank[task_id], grad_bank[task_id], acc_drop=0.0
                )

        tracker.stop()
        return self.cl_matrix.final_metrics()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_task(self, task_id: int):
        loader = self.stream.loader(self.stream.tasks[task_id]["train"])  # train split as proxy
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.dev), y.to(self.dev)
            preds = self._forward(x, task_id).argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            if total > 2000:  # evaluation budget guardrail
                break
        return correct / total
