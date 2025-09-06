"""
preprocess.py – dataset loading & task stream helpers
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100

# -----------------------------------------------------------------------------
# Helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _hf_split_name(hf_name: str) -> Tuple[str, str | None]:
    """Split a HuggingFace dataset identifier in the form `dataset/config`.

    Returns a tuple (dataset, config) where *config* can be ``None`` if no
    slash is present.  This allows configuration files to specify either style
    (``clue/afqmc`` *or* ``clue`` + ``afqmc``).
    """
    if "/" in hf_name:
        dataset, config = hf_name.split("/", 1)
        return dataset, config
    return hf_name, None

# -----------------------------------------------------------------------------
# NLP streams (CLUE++, SuperGLUE) ---------------------------------------------
# -----------------------------------------------------------------------------

class CLUESplit:
    """Pre-loads all CLUE++ datasets as specified in the config file."""

    def __init__(self, task_cfg: List[Dict[str, Any]]):
        self.datasets: Dict[str, Any] = {}
        for t in task_cfg:
            name: str = t["name"]
            hf_name: str = t["hf_name"]
            dataset_id, config_name = _hf_split_name(hf_name)
            if config_name is None:
                ds = load_dataset(dataset_id)
            else:
                ds = load_dataset(dataset_id, config_name)
            self.datasets[name] = ds

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
