import os
from typing import Dict, List

import math
import numpy as np
import seaborn as sns
import matplotlib as mpl

mpl.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

# Images are stored in the research directory requested by the task
IMG_DIR = os.path.join(".research", "iteration6", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def plot_bar(data: Dict[str, float], title: str, filename: str):
    """Create a PDF barplot with annotated numbers."""
    names, vals = list(data.keys()), list(data.values())
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=names, y=vals, palette="crest")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, filename)
    plt.savefig(path, bbox_inches="tight", format="pdf")
    print("[FIG] saved", path)

# ------------------------------------------------------------------
# Statistical helper
# ------------------------------------------------------------------

def summary_ci(values: List[float]):
    """Return mean and 95% confidence interval of a sample list."""
    if len(values) == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci = 1.96 * std / math.sqrt(len(values))
    return mean, ci
