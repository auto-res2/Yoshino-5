"""
preprocess.py – dataset loading & task stream helpers
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from datasets import load_dataset
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100

# -----------------------------------------------------------------------------
# NLP streams (CLUE++, SuperGLUE) ---------------------------------------------
# -----------------------------------------------------------------------------

class CLUESplit:
    """Pre-loads all CLUE++ datasets as specified in the config file."""

    def __init__(self, task_cfg: List[Dict[str, Any]]):
        self.datasets: Dict[str, Any] = {}
        for t in task_cfg:
            name = t["name"]
            hf_name = t["hf_name"]
            self.datasets[name] = load_dataset(hf_name)

    def get_dataset(self, name: str):
        return self.datasets[name]


class SuperGLUEStream:
    """Generates a random task ordering of SuperGLUE tasks given a seed."""

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
        self.order = self._random_order(seed)
        self.idx = 0

    def _random_order(self, seed: int):
        rng = random.Random(seed)
        tasks = self.TASKS.copy()
        rng.shuffle(tasks)
        return tasks

    def next_task(self):
        if self.idx >= len(self.order):
            raise StopIteration
        t = self.order[self.idx]
        self.idx += 1
        return t

# -----------------------------------------------------------------------------
# Vision: Split-CIFAR100 -------------------------------------------------------
# -----------------------------------------------------------------------------

def split_cifar100(num_experiences: int = 20):
    """Return an Avalanche benchmark for Split-CIFAR100."""
    return SplitCIFAR100(
        seed=0,
        shuffle=True,
        train_transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        eval_transform=transforms.Compose([transforms.ToTensor()]),
        n_experiences=num_experiences,
    )
