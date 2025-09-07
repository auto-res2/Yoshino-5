"""
evaluate.py – evaluation utilities, statistics & figures
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from .train import _DEVICE  # re-use global device

# -----------------------------------------------------------------------------
# 1.  Metrics
# -----------------------------------------------------------------------------

def evaluate_acc(model, loader):
    """Simple exact-match accuracy for classification-style tasks."""
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            out = model(**batch).logits.argmax(dim=-1)
            correct += (out == batch["labels"]).float().sum().item()
            total += batch["labels"].numel()
    return correct / max(total, 1)

# -----------------------------------------------------------------------------
# 2.  Plotting helpers
# -----------------------------------------------------------------------------

def save_barplot(data: Dict[str, float], title: str, fname: Path):
    """Save a PDF barplot with value annotations."""
    mpl.use("Agg")  # headless rendering
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=list(data.keys()), y=list(data.values()), palette="crest")
    for i, v in enumerate(data.values()):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0, max(data.values()) * 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"[FIG] saved {fname}")
