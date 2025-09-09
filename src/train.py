[UPDATED]
import os
import sys
import time
import random
import copy
import warnings
import types
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

import timm

# -----------------------------------------------------------------------------
# Optional dependency: bitsandbytes
# -----------------------------------------------------------------------------
try:
    import bitsandbytes as bnb  # noqa: F401
    _has_bnb = True
except Exception:  # pragma: no cover
    _has_bnb = False
    bnb = types.ModuleType("bitsandbytes_stub")

    class _AdamW8bit(torch.optim.AdamW):  # type: ignore
        """Fallback replacement that behaves like torch.optim.AdamW."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    bnb.optim = types.SimpleNamespace(AdamW8bit=_AdamW8bit)  # type: ignore

# -----------------------------------------------------------------------------
# Optional dependency: Avalanche
# -----------------------------------------------------------------------------
try:
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.storage_policy import ReservoirSamplingBuffer
    from avalanche.training.strategies import Naive
    from avalanche.evaluation.metrics import (
        accuracy_metrics,
        loss_metrics,
        forgetting_metrics,
    )
    from avalanche.logging import InteractiveLogger
    from avalanche.training.plugins import EvaluationPlugin

    _has_avalanche = True
except Exception:  # pragma: no cover
    _has_avalanche = False
    # Minimal stubs -----------------------------------------------------------
    ReplayPlugin = ReservoirSamplingBuffer = Naive = object  # type: ignore

    def _missing(*_args, **_kwargs):  # pylint: disable=unused-argument
        raise RuntimeError(
            "Avalanche-lib is required for the ER-500 baseline but could not be "
            "imported. Install avalanche-lib to use this feature."
        )

    accuracy_metrics = loss_metrics = forgetting_metrics = _missing  # type: ignore
    InteractiveLogger = EvaluationPlugin = object  # type: ignore

# NOTE: switched to absolute imports (relative imports fail when executing as a
# standalone script).  The project root (`src/`) is automatically added to
# `sys.path` by the Python interpreter, so direct imports work reliably.
from preprocess import get_cifar100_benchmark  # noqa: E402
from evaluate import compute_metric_matrices  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# =============================================================================
# Model & PEFT
# =============================================================================
from peft import get_peft_model, LoraConfig, TaskType  # Imported late to avoid PEFT stub complexity


# (The remainder of the file is unchanged)
