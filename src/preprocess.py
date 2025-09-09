"""src/preprocess.py
Dataset construction, configuration helpers and reproducibility utilities.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from avalanche.benchmarks import SplitCIFAR100, PermutedMNIST, SplitMiniImageNet

# ----------------------------------------------------------------------------
# Deterministic behaviour
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------
# Config loader – looks for env var CLIP_CONF or falls back to default file
# ----------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    custom = os.environ.get("CLIP_CONF", None)
    cfg_path = custom if custom and os.path.exists(custom) else path
    with open(cfg_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

# ----------------------------------------------------------------------------
# Benchmark helpers
# ----------------------------------------------------------------------------

def get_benchmark(name: str, **kwargs):
    if name == "SplitCIFAR100":
        return SplitCIFAR100(**kwargs)
    if name == "PermutedMNIST":
        return PermutedMNIST(**kwargs)
    if name == "SplitMiniImageNet":
        return SplitMiniImageNet(**kwargs)
    raise ValueError(name)
