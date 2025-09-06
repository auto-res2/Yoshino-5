from __future__ import annotations

"""
evaluate.py – continual-learning metrics, statistics and plotting helpers
"""
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
    xs,
    ys,
    err,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    fname: Path | str,
):
    """Utility which *always* writes into .research/iteration6/images.

    Irrespective of the filename suggested by the caller we enforce the new
    requirement that all images must live under
    ``.research/iteration6/images``.  Only the basename of *fname* is
    preserved.  The target directory is created on-demand.
    """

    # Resolve central images directory (repo-root/.research/iteration6/images)
    root = Path(__file__).resolve().parent.parent
    images_dir = root / ".research" / "iteration6" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(fname, (str, Path)):
        fname = Path(fname).name  # keep only the final component
    else:  # pragma: no cover – defensive
        raise TypeError("fname must be path-like")

    final_path = images_dir / fname

    plt.figure()
    plt.plot(xs, ys, label=title)
    plt.fill_between(xs, ys - err, ys + err, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(str(final_path), bbox_inches="tight")
