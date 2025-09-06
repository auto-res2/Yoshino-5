"""preprocess.py – dataset helpers for CLUE, SuperGLUE and Split-CIFAR"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from datasets import load_dataset
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100

log = logging.getLogger("agsc.data")

# ---------------------------------------------------------------------------
# NLP helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------

class CLUE:
    """Offline CLUE++ loader (multi-task)."""

    def __init__(self, tasks: List[Dict[str, str]]):
        self.ds: Dict[str, Any] = {}
        for t in tasks:
            name, hf = t["name"], t["hf"]
            if "/" in hf:
                dset, config = hf.split("/")
            else:  # pragma: no cover – never happens with official configs
                dset, config = hf, None
            self.ds[name] = load_dataset(dset, config)

    def dataset(self, name: str):
        return self.ds[name]


class SuperGLUEStream:
    """Randomised online SuperGLUE task stream."""

    TASKS = [
        "cb",
        "copa",
        "multirc",
        "record",
        "wic",
        "wsc",
        "boolq",
        "rte",
        "ax_b",
        "ax_g",
    ]

    def __init__(self, seed: int):
        self.order = self.TASKS.copy()
        random.Random(seed).shuffle(self.order)
        self._idx = 0

    # Iterator-style helpers --------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.order):
            raise StopIteration
        t = self.order[self._idx]
        self._idx += 1
        return t

    # Convenience ------------------------------------------------------------
    def peek(self, k: int):
        return self.order[self._idx : self._idx + k]


# ---------------------------------------------------------------------------
# Vision helper --------------------------------------------------------------
# ---------------------------------------------------------------------------

def split_cifar(num_tasks: int = 20):
    """Return the Avalanche Split-CIFAR100 benchmark with *num_tasks* splits."""

    return SplitCIFAR100(
        seed=0,
        shuffle=True,
        n_experiences=num_tasks,
        train_transform=transforms.Compose(
            [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        eval_transform=transforms.Compose([transforms.ToTensor()]),
    )
