"""
evaluate.py – continual-learning metrics, statistics and plotting helpers
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# -----------------------------------------------------------------------------
# Metrics ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def backward_transfer(acc_matrix: np.ndarray) -> float:
    """Average forgetting across tasks (Lopez-Paz & Ranzato, 2017)."""
    t = acc_matrix.shape[0]
    diffs = np.tril(acc_matrix, -1) - np.diag(acc_matrix)
    return float(diffs.sum() / (t * (t - 1) / 2))


def forward_transfer(acc_matrix: np.ndarray) -> float:
    t = acc_matrix.shape[0]
    diffs = np.triu(acc_matrix, 1) - np.diag(acc_matrix)
    return float(diffs.sum() / (t * (t - 1) / 2))

# -----------------------------------------------------------------------------
# Statistics ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def ci95(vals: List[float]):
    mean = float(np.mean(vals))
    se = st.sem(vals)
    h = float(se * 1.96)
    return mean, h


def paired_ttest(sample_a: List[float], sample_b: List[float]):
    return st.ttest_rel(sample_a, sample_b)

# -----------------------------------------------------------------------------
# Plotting --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def line_plot(
    xs, ys, err, *, xlabel: str, ylabel: str, title: str, fname: Path | str
):
    plt.figure()
    plt.plot(xs, ys, label=title)
    plt.fill_between(xs, ys - err, ys + err, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(str(fname), bbox_inches="tight")
