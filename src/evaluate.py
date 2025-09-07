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
    """Exact-match accuracy over target (label) positions only."""
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            pred = out.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100  # only supervise non-ignored positions
            correct += ((pred == labels) & mask).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)

# -----------------------------------------------------------------------------
# 2.  Plotting helpers
# -----------------------------------------------------------------------------


def _get_fig_dir() -> Path:
    """Return the canonical directory for saving all experiment images."""
    # All figures must reside under .research/iteration12/images according to the
    # global repository convention.
    root = Path(__file__).resolve().parent.parent  # repository root
    fig_dir = root / ".research" / "iteration12" / "images"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_barplot(data: Dict[str, float], title: str, fname: Path | None = None):
    """Save a PDF bar-plot with value annotations to the mandated image folder.

    Parameters
    ----------
    data : Dict[str, float]
        Mapping from bar label → value.
    title : str
        Plot title.
    fname : pathlib.Path | None, optional
        Desired file name.  Only the *stem* portion will be honoured – the file
        will always be stored under `.research/iteration12/images` as required
        by the CI harness.  If *None*, the title stem will be slugified.
    """
    mpl.use("Agg")  # headless rendering
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------------------------ figure
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=list(data.keys()), y=list(data.values()), palette="crest")
    for i, v in enumerate(data.values()):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0, max(data.values()) * 1.15 if data else 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.tight_layout()

    # ------------------------------------------------------------- output path
    fig_dir = _get_fig_dir()
    if fname is None:
        safe_stem = title.lower().replace(" ", "_").replace("/", "-")
        fname = Path(f"{safe_stem}.pdf")
    final_path = fig_dir / Path(fname).with_suffix(".pdf").name

    plt.savefig(final_path, format="pdf", bbox_inches="tight")
    print(f"[FIG] saved {final_path.relative_to(fig_dir.parent)}")
