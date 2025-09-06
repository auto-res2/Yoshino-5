"""src/evaluate.py
Evaluation utilities: continual-learning metric tracking and plotting helpers.
"""
from __future__ import annotations

from typing import Dict
import torch
from torchmetrics.classification import MulticlassAccuracy

class ContinualMetrics:
    """Keeps the task × task accuracy matrix and provides CL metrics."""

    def __init__(self, n_tasks: int, n_classes: int, device: str = "cpu"):
        self.acc_matrix = torch.zeros((n_tasks, n_tasks))
        self.metric = MulticlassAccuracy(num_classes=n_classes).to(device)
        self.n_tasks = n_tasks

    # ------------------------------------------------------------------
    def update(self, task_seen: int, task_eval: int, preds: torch.Tensor, tgt: torch.Tensor):
        acc = self.metric(preds, tgt)
        self.acc_matrix[task_seen, task_eval] = acc.item()

    # ------------------------------------------------------------------
    def final_results(self) -> Dict[str, float]:
        final_acc = self.acc_matrix[-1].mean().item()
        bwt = (
            self.acc_matrix[-1, :-1] - torch.diag(self.acc_matrix, diagonal=0)[:-1]
        ).mean().item()
        min_acc = self.acc_matrix.min().item()
        return {
            "Final_ACC": final_acc * 100,
            "BWT": bwt * 100,
            "min_ACC": min_acc * 100,
        }
