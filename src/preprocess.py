"""
preprocess.py – dataset loading & task stream helpers
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100

logger = logging.getLogger("agsc.preprocess")

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
    """Pre-loads all CLUE++ datasets as specified in the config file.

    The CLUE benchmark has one peculiarity: the *WSC* task is distributed under
    the builder-config name ``cluewsc2020`` rather than plain ``wsc``.  To keep
    user-facing configuration files intuitive we transparently map the short
    identifier ``wsc`` to the official builder name.  Additional aliases can be
    registered in the ``_CONFIG_ALIASES`` map below when needed.
    """

    _CONFIG_ALIASES: Dict[str, str] = {
        "wsc": "cluewsc2020",
    }

    def __init__(self, task_cfg: List[Dict[str, Any]]):
        self.datasets: Dict[str, Any] = {}
        for t in task_cfg:
            name: str = t["name"]
            hf_name: str = t["hf_name"]
            dataset_id, config_name = _hf_split_name(hf_name)

            # -----------------------------------------------------------------
            # Load via 🤗 Datasets – with graceful alias fallback --------------
            # -----------------------------------------------------------------
            try:
                if config_name is None:
                    ds = load_dataset(dataset_id)
                else:
                    ds = load_dataset(dataset_id, config_name)
            except ValueError as exc:
                # Handle known aliases (e.g. wsc -> cluewsc2020)
                if (
                    dataset_id == "clue"
                    and config_name in self._CONFIG_ALIASES
                ):
                    alias = self._CONFIG_ALIASES[config_name]  # type: ignore[index]
                    logger.warning(
                        "Config '%s/%s' not found – retrying with official name '%s'.",
                        dataset_id,
                        config_name,
                        alias,
                    )
                    ds = load_dataset(dataset_id, alias)
                else:
                    raise exc  # Re-raise if we do not know how to fix it

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
