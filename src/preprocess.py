"""
preprocess.py – data loading, preprocessing and reproducibility helpers
"""
from __future__ import annotations

import contextlib
import importlib
import pathlib
import random
from typing import Any, Tuple, List, Sequence

import numpy as np
import torch
import torchvision
from torchvision import transforms as T  # noqa: F401  (kept for future use)
import yaml

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: dict[str, Any] = yaml.safe_load(f)

def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Dataset helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def get_cifar100_split(root: str | pathlib.Path, two_class: bool = True):
    """Return full train/test datasets and a list of tasks (50 × 2-class)."""
    root = str(root)
    full_train = torchvision.datasets.CIFAR100(root, train=True, download=True)
    full_test = torchvision.datasets.CIFAR100(root, train=False, download=True)
    class_order = list(range(100))
    tasks = []
    for t in range(0, 100, 2):
        classes = class_order[t : t + 2]
        tr_idx = [i for i, (_, y) in enumerate(full_train) if y in classes]
        te_idx = [i for i, (_, y) in enumerate(full_test) if y in classes]
        tasks.append({
            "classes": classes,
            "train_idx": tr_idx,
            "test_idx": te_idx,
        })
    return full_train, full_test, tasks

# ---------------------------------------------------------------------------
# miniImageNet helper (optional torchmeta) ----------------------------------
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _optional_import(name: str):
    try:
        module = importlib.import_module(name)
        yield module
    except ImportError:
        yield None

def get_miniimagenet_split(root: str | pathlib.Path):
    with _optional_import("torchmeta.datasets") as tm:
        if tm is None:
            raise RuntimeError("torchmeta not installed – required for miniImageNet")
        ds = tm.MiniImagenet(root=root, split="train", download=True)
        tasks = []
        for c in range(100):
            class_idxs = ds.class_to_idx[c]
            random.shuffle(class_idxs)
            n = len(class_idxs)
            tr, va, te = np.split(class_idxs, [int(0.5 * n), int(0.6 * n)])
            tasks.append({
                "classes": [c],
                "train_idx": tr.tolist(),
                "test_idx": te.tolist(),
            })
        return ds, ds, tasks
