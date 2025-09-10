from __future__ import annotations

"""
preprocess.py – data loading, preprocessing and reproducibility helpers
The previous implementation loaded CIFAR-100 without any transform, so
`torchvision.datasets.CIFAR100` returned PIL images.  When these images
were batched by `torch.utils.data.DataLoader`, the default collate
function raised

    TypeError: default_collate: batch must contain tensors, numpy arrays …

because it cannot collate arbitrary Python objects (PIL.Image.Image).

Fix: add a *minimal* transform pipeline that converts the PIL image to a
`torch.FloatTensor` in the `[0,1]` range.  No normalisation or data
augmentation is required for the unit-tests driving this repository – we
just need a tensor so that collation works.
"""

import contextlib
import importlib
import pathlib
import random
from typing import Any, Tuple, List, Sequence

import numpy as np
import torch
import torchvision
from torchvision import transforms as T  # noqa: F401 (kept for future use)
import yaml

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: dict[str, Any] = yaml.safe_load(f)


def _set_global_seed(seed: int):
    """Set seeds for python, numpy and (if available) CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Dataset helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

# Minimal transformation – converts PIL → Tensor so that DataLoader can collate
_BASIC_CIFAR_TRANSFORM = torchvision.transforms.ToTensor()

def get_cifar100_split(root: str | pathlib.Path, two_class: bool = True):
    """Return full train/test datasets and a list of tasks (50 × 2-class).

    The function now ensures that every sample is a *tensor*, preventing the
    `default_collate` TypeError that occurred when PIL images were returned.
    """
    root = str(root)
    full_train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=_BASIC_CIFAR_TRANSFORM)
    full_test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=_BASIC_CIFAR_TRANSFORM)

    class_order = list(range(100))
    tasks = []
    step = 2 if two_class else 1
    for t in range(0, 100, step):
        classes = class_order[t : t + step]
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
        tasks: List[dict[str, Any]] = []
        for c in range(100):
            class_idxs = ds.class_to_idx[c]
            random.shuffle(class_idxs)
            n = len(class_idxs)
            tr, _va, te = np.split(class_idxs, [int(0.5 * n), int(0.6 * n)])
            tasks.append({
                "classes": [c],
                "train_idx": tr.tolist(),
                "test_idx": te.tolist(),
            })
        return ds, ds, tasks