"""src/evaluate.py
Metric utilities separated from training so they can easily be reused by
alternative learners or for post-hoc analysis only.
"""
from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["CLMatrix"]


class CLMatrix:
    """In-memory accuracy matrix for continual learning (rows = after task i, columns = task j)."""
    def __init__(self, n_tasks: int):
        self.mat = torch.zeros((n_tasks, n_tasks))
        self.n = n_tasks

    # ------------------------------------------------------------------
    def update(self, task_seen: int, task_eval: int, acc: float):
        self.mat[task_seen, task_eval] = acc

    # ------------------------------------------------------------------
    def final_metrics(self):
        final_acc = self.mat[-1].mean().item()
        diag = torch.diag(self.mat)
        bwt = (self.mat[-1, :-1] - diag[:-1]).mean().item()
        min_acc = self.mat.min().item()
        return {"Final_ACC": final_acc * 100, "BWT": bwt * 100, "min_ACC": min_acc * 100}

    # ------------------------------------------------------------------
    def save_csv(self, path: Path):
        pd.DataFrame(self.mat.numpy()).to_csv(path, index=False)
