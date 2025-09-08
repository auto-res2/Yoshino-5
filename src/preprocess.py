"""src/preprocess.py
Dataset streams, data-loading & image preprocessing utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Avalanche provides convenient CL benchmarks.
try:
    from avalanche.benchmarks import split_cifar100
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install avalanche-lib") from e

# -----------------------------------------------------------------------------
#  Normalisation constants
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
#  Image transforms
# -----------------------------------------------------------------------------
try:
    randaugment = transforms.RandAugment  # available in torchvision>=0.13
except AttributeError:  # pragma: no cover
    # fallback to identity transform if RandAugment is missing
    class _Identity:
        def __call__(self, img):
            return img
    randaugment = _Identity

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    randaugment(num_ops=2, magnitude=10) if callable(randaugment) else randaugment(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------------------------------------------------------------
#  Stream helpers
# -----------------------------------------------------------------------------

def get_stream(name: str, n_experiences: int, ordering: str):
    """Return an Avalanche benchmark stream according to *name*."""
    if name == "SplitCIFAR100":
        benchmark = split_cifar100.SplitCIFAR100(
            n_experiences=n_experiences,
            seed=0,
            fixed_class_order=None if ordering == "random" else list(range(100)),
            return_task_id=True,
        )
        return benchmark
    raise NotImplementedError(f"Unknown stream: {name}")


def make_loader(dataset, batch_size: int, cfg, shuffle: bool = True):
    """Standard PyTorch DataLoader adhering to config settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.SYSTEM["num_workers"],
        pin_memory=cfg.SYSTEM["pin_memory"],
        drop_last=False,
    )
