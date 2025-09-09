"""src/evaluate.py
All evaluation / statistics / verification utilities live here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

# -----------------------------------------------------------------------------
# Mandatory components list – kept identical to single-file version
# -----------------------------------------------------------------------------
_REQUIRED_COMPONENTS: List[str] = [
    "LoRA",
    "SpectralAdapter",
    "InterferenceGraph",
    "BeamSearch",
    "OnlineBubbleSwap",
    "MC-SGD",
    "Replay",
    "GPM",
    "MAS",
    "SplitCIFAR100",
    "PermutedMNIST",
    "SplitMiniImageNet",
    "MultiSeed",
    "ConfidenceIntervals",
    "SignificanceTests",
]

# -----------------------------------------------------------------------------
# Verification utilities (unchanged)
# -----------------------------------------------------------------------------

def verify_implementation() -> bool:
    """Sanity-check: every required component name is present in globals()."""

    implemented = set(globals().keys())
    missing = [c for c in _REQUIRED_COMPONENTS if c not in implemented]
    if missing:
        logging.error("Missing components: %s", missing)
    return not missing


# -----------------------------------------------------------------------------
# Simple validation for Split-CIFAR experiment – same logic, refactored
# -----------------------------------------------------------------------------

def validate_results(all_results: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    e1_runs: List[Dict[str, Any]] = all_results["experiment1"]
    aa_clip = np.mean([run["CLIP_FULL"]["acc"][-1].mean() for run in e1_runs])
    baselines = cfg["experiment1"]["baselines"]
    best_base = max(
        np.mean([run[m]["acc"][-1].mean() for run in e1_runs]) for m in baselines if m != "CLIP_FULL"
    )
    assert aa_clip >= best_base + 0.05, "CLIP does not beat baselines by ≥5pp!"
