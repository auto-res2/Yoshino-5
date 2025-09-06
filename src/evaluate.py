"""evaluate.py – metrics, statistical analysis & plotting utilities"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

log = logging.getLogger("agsc.eval")

# ---------------------------------------------------------------------------
# Continual-learning metrics -------------------------------------------------
# ---------------------------------------------------------------------------

def acc(pred: np.ndarray, label: np.ndarray) -> float:  # noqa: D401
    """Classification accuracy."""

    return float((pred == label).mean())


def bwt(acc_matrix: np.ndarray) -> float:  # noqa: D401
    """Backward transfer as defined in the CL literature."""

    t = acc_matrix.shape[0]
    dif = np.tril(acc_matrix, -1) - np.diag(acc_matrix)
    return float(dif.sum() / (t * (t - 1) / 2))


def fwt(acc_matrix: np.ndarray) -> float:  # noqa: D401
    """Forward transfer."""

    t = acc_matrix.shape[0]
    dif = np.triu(acc_matrix, 1) - np.diag(acc_matrix)
    return float(dif.sum() / (t * (t - 1) / 2))

# ---------------------------------------------------------------------------
# Statistics -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def ci95(vals: List[float]):
    m = float(np.mean(vals))
    se = float(st.sem(vals)) if len(vals) > 1 else 0.0
    return m, 1.96 * se


def paired_t(a: List[float], b: List[float]):
    return st.ttest_rel(a, b)

# ---------------------------------------------------------------------------
# Plot helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _annotate(ax):
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def bar(values: dict, title: str, fname: Path):
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    x = list(values.keys())
    y = list(values.values())
    ax.bar(x, y, color="steelblue")
    _annotate(ax)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticklabels(x, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
