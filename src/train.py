"""
train.py – model/scheduler definitions and the single-task training loop
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import torchmetrics

# ---------------------------------------------------------------------------
# Load global configuration --------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: dict[str, Any] = yaml.safe_load(f)

device = CONFIG["device"] if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
#  INFLUENCE GRAPH WITH KRONECKER SKETCH ------------------------------------
# ---------------------------------------------------------------------------
class InfluenceGraph:
    """Sparse additive Kronecker sketch of ΔF and ΔT matrices."""

    def __init__(self, dim: int, sketch_density: float = 0.01, device: str = "cpu"):
        self.dim = dim
        self.k = max(1, int(dim * sketch_density))
        self.device = device
        # random projection matrices (fixed)
        self.R = torch.randn(self.k, dim, device=device)
        self.S_F = torch.zeros(self.k, device=device)
        self.S_T = torch.zeros(self.k, device=device)

    # ------------------------------------------------------------------
    def _proj(self, vec: torch.Tensor) -> torch.Tensor:
        return self.R @ vec  # (k,)

    # ------------------------------------------------------------------
    def add_edge(self, i: int, j: int, dF: float, dT: float):
        e_i = torch.zeros(self.dim, device=self.device)
        e_j = torch.zeros(self.dim, device=self.device)
        e_i[i] = 1.0
        e_j[j] = 1.0
        v = self._proj(torch.kron(e_i, e_j))
        self.S_F += dF * v
        self.S_T += dT * v

    # ------------------------------------------------------------------
    def estimate(self, i: int, j: int) -> Tuple[float, float]:
        e_i = torch.zeros(self.dim, device=self.device)
        e_j = torch.zeros(self.dim, device=self.device)
        e_i[i] = 1.0
        e_j[j] = 1.0
        v = self._proj(torch.kron(e_i, e_j))
        dF = (v * self.S_F).sum().item() / self.k
        dT = (v * self.S_T).sum().item() / self.k
        return dF, dT

# ---------------------------------------------------------------------------
#  POINTER NETWORK SCHEDULER -------------------------------------------------
# ---------------------------------------------------------------------------
class PointerNet(nn.Module):
    def __init__(self, inp_dim: int, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(inp_dim, hidden, n_layers, batch_first=True)
        self.decoder = nn.LSTM(inp_dim, hidden, n_layers, batch_first=True)
        self.pointer = nn.Linear(hidden, hidden)
        self.attn_v = nn.Parameter(torch.randn(hidden))

    # ------------------------------------------------------------------
    def forward(self, enc_inputs: torch.Tensor) -> torch.Tensor:
        """enc_inputs: (B, T, D) – returns logits (B, T, T)"""
        B, T, D = enc_inputs.shape
        enc_out, (h, c) = self.encoder(enc_inputs)
        dec_inp = enc_inputs.new_zeros(B, 1, D)  # start-token (zeros)
        logits = []
        for _ in range(T):
            dec_out, (h, c) = self.decoder(dec_inp, (h, c))
            # Bahdanau attention
            score = torch.tanh(self.pointer(enc_out) + dec_out)
            score = torch.einsum("btd,d->bt", score, self.attn_v)
            logits.append(score)
            # greedy step
            idx = score.argmax(-1)
            dec_inp = enc_inputs.gather(1, idx[:, None, None].expand(-1, 1, D))
        return torch.stack(logits, dim=1)

# ---------------------------------------------------------------------------
#  TASK-CLIP SCHEDULER WRAPPER ----------------------------------------------
# ---------------------------------------------------------------------------
class TaskCLIPScheduler:
    def __init__(self, num_tasks: int, feature_dim: int = 16, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or CONFIG["scheduler"]
        self.g = InfluenceGraph(num_tasks, sketch_density=self.cfg["sketch_density"], device=device)
        self.pointer = PointerNet(inp_dim=feature_dim,
                                  hidden=self.cfg["pointer"]["hidden_size"],
                                  n_layers=self.cfg["pointer"]["n_layers"]).to(device)
        self.lambda_param = torch.tensor([self.cfg["lambda_init"]], device=device, requires_grad=True)
        self.opt_lambda = optim.Adam([self.lambda_param], lr=self.cfg["lambda_lr"])
        self.opt_ptr = optim.Adam(self.pointer.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    def propose_order(self, buffer_task_ids: Sequence[int]) -> List[int]:
        """Return a permutation of buffer_task_ids using beam search."""
        beam: List[Tuple[float, List[int]]] = [(0.0, [])]
        for _ in range(len(buffer_task_ids)):
            new_beam: List[Tuple[float, List[int]]] = []
            for cost, seq in beam:
                remaining = [t for t in buffer_task_ids if t not in seq]
                if not remaining:
                    new_beam.append((cost, seq))
                    continue
                for t in remaining:
                    dF, dT = (self.g.estimate(seq[-1], t) if seq else (0.0, 0.0))
                    C = self.lambda_param.item() * dF - (1 - self.lambda_param.item()) * dT
                    new_beam.append((cost + C, seq + [t]))
            beam = sorted(new_beam, key=lambda x: x[0])[: self.cfg["beam_width"]]
        return beam[0][1]

    # ------------------------------------------------------------------
    def update_after_task(self, task_prev: int, task_next: int, dF: float, dT: float, reward: float):
        self.g.add_edge(task_prev, task_next, dF, dT)
        # Update λ with policy-gradient-like surrogate
        self.opt_lambda.zero_grad()
        loss = -reward * torch.log(self.lambda_param.clamp(1e-3, 1 - 1e-3))
        loss.backward()
        self.opt_lambda.step()
        self.lambda_param.data.clamp_(0.0, 1.0)
        # Pointer-net fine-tuning could be added here (omitted for brevity)

# ---------------------------------------------------------------------------
#  CONTINUAL LEARNER WRAPPERS ------------------------------------------------
# ---------------------------------------------------------------------------
class DERPlusPlusWrapper:
    """Light wrapper that lazily clones DER++ and exposes the observe API."""

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        repo = pathlib.Path("external/der_plus_plus")
        if not repo.exists():
            subprocess.check_call(["git", "clone", "--depth", "1",
                                   "https://github.com/giacomo-cgn/der-plus-plus",
                                   str(repo)])
        sys.path.insert(0, str(repo))
        from src.models.derpp import DERPP  # type: ignore

        self.impl = DERPP(backbone, buffer_size=512, alpha=0.1, beta=0.5)

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.impl.observe(x, y)


class LiDERWrapper:
    """Light wrapper for LiDER baseline."""

    def __init__(self, backbone: nn.Module):
        repo = pathlib.Path("external/lider")
        if not repo.exists():
            subprocess.check_call(["git", "clone", "--depth", "1",
                                   "https://github.com/aimagelab/LiDER", str(repo)])
        sys.path.insert(0, str(repo))
        from methods.lider import LiDER  # type: ignore

        self.impl = LiDER(backbone=backbone, lambda_align=0.5, lambda_dis=1.0, mem_percent=0.02)

    # ------------------------------------------------------------------
    def observe(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.impl.observe(x, y)

# ---------------------------------------------------------------------------
#  TRAINING LOOP FOR A SINGLE TASK ------------------------------------------
# ---------------------------------------------------------------------------

def train_task(learner, dataloader: DataLoader, epochs: int, amp: bool = True) -> float:
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)
    for _ in range(epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = learner.observe(xb, yb)
            scaler.scale(loss).backward()
            # Some external implementations expose "opt"; check existence
            if hasattr(learner.impl, "opt"):
                scaler.step(learner.impl.opt)
                learner.impl.opt.zero_grad()
            scaler.update()
            # Update running accuracy
            with torch.no_grad():
                preds = learner.backbone(xb)
                accuracy.update(preds, yb)
    return accuracy.compute().item()
