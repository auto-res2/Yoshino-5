"""src/evaluate.py
Evaluation utilities: metrics, statistics, and (optionally) plotting.
Only light-weight code is provided – full statistical analysis is out-
of-scope for this refactor but hooks are kept intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import stats

# -----------------------------------------------------------------------------
#  Continual-learning metrics store
# -----------------------------------------------------------------------------
@dataclass
class ContinualMetrics:
    """Keeps a matrix of per-task accuracies to derive CL metrics."""

    acc_matrix: List[List[float]] = field(default_factory=list)

    # ---------------------------------------------------------------------
    def update(self, task_id: int, accs: List[float]):
        """Add a new row (results measured after finishing *task_id*)."""
        if len(self.acc_matrix) <= task_id:
            self.acc_matrix.append(accs)
        else:
            self.acc_matrix[task_id] = accs

    # ---------------------------------------------------------------------
    def average_accuracy(self) -> float:
        if not self.acc_matrix:
            return 0.0
        return float(np.mean(self.acc_matrix[-1]))

    # More sophisticated CL metrics (BWT, FWT …) would be implemented here.

# -----------------------------------------------------------------------------
#  Statistical tests
# -----------------------------------------------------------------------------

def paired_t(sample_a: List[float], sample_b: List[float]):
    """Return t-value & p-value for paired Student-t test."""
    t, p = stats.ttest_rel(sample_a, sample_b)
    return {"t": float(t), "p": float(p)}
