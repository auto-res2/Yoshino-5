from __future__ import annotations
"""src/preprocess.py
Dataset construction, configuration helpers and reproducibility utilities.

The original implementation imported ``SplitMiniImageNet`` unconditionally from
``avalanche``.  Recent releases of Avalanche no longer expose this symbol at
module level which caused an ``ImportError`` at import-time – even though
``SplitMiniImageNet`` is only required by the (currently disabled) Experiment 3.

To restore compatibility we
1. try to import ``SplitMiniImageNet`` and gracefully degrade to a lightweight
   stub when it is not available;
2. gate ``get_benchmark('SplitMiniImageNet')`` behind a run-time check so that
   if the symbol is missing we raise a clear, informative error instead of
   crashing at module import time.

All other functionality is unchanged.
"""

import os
import random
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch
import yaml
from avalanche.benchmarks import SplitCIFAR100, PermutedMNIST

# -----------------------------------------------------------------------------
# Optional SplitMiniImageNet import (may be absent depending on Avalanche ver.)
# -----------------------------------------------------------------------------
try:
    # Present in some Avalanche versions (<0.5.0).  Newer versions expose a
    # "MiniImageNet" benchmark generator instead.  We attempt the import and, on
    # failure, fall back to a stub so that the rest of the codebase continues
    # to load even when the symbol is unavailable.
    from avalanche.benchmarks import SplitMiniImageNet  # type: ignore
except ImportError:  # pragma: no cover – execution path when symbol missing
    SplitMiniImageNet = None  # type: ignore[misc,assignment]
    if TYPE_CHECKING:  # Mypy ‑ silence "name defined as Any" complaints
        from types import ModuleType as _SplitMiniImageNetType
        SplitMiniImageNet = ...  # type: ignore[assignment]

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
        if SplitMiniImageNet is None:
            raise ImportError(
                "SplitMiniImageNet is not available in this Avalanche version. "
                "Please install an earlier release (<0.5) or adapt the code to "
                "use the new MiniImageNet benchmark utilities."
            )
        return SplitMiniImageNet(**kwargs)  # type: ignore[func-returns-value]
    raise ValueError(name)
