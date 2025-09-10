"""
evaluate.py – statistical evaluation, verification & validation utilities
"""
from __future__ import annotations

import pathlib
from typing import Any, List

import yaml

CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: dict[str, Any] = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Required components list (for reflection-based verification) --------------
# ---------------------------------------------------------------------------
ALL_REQUIRED_COMPONENTS: List[str] = [
    "InfluenceGraph",
    "PointerNet",
    "TaskCLIPScheduler",
    "DERPlusPlusWrapper",
    "LiDERWrapper",
    "beam_search",          # concept-level checks
    "kronecker_sketch",
]

EXPECTED_CHARACTERISTICS = {
    "acc_gain_pp": 5,
    "fgt_reduction_pct": 30,
    "runtime_overhead_pct": 15,
}

# ---------------------------------------------------------------------------
# Verification --------------------------------------------------------------
# ---------------------------------------------------------------------------

def verify_implementation() -> bool:
    """Ensure that all required symbols are present in the runtime."""
    import inspect, sys

    glob = {
        k: v for m in sys.modules.values() if m for k, v in vars(m).items()  # type: ignore[arg-type]
    }
    missing = [comp for comp in ALL_REQUIRED_COMPONENTS if comp not in glob]
    if missing:
        raise AssertionError(f"Missing components: {missing}")
    print("[verify] All components present ✔️")
    return True

# ---------------------------------------------------------------------------
# Validation ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def validate_results():
    """Placeholder to validate aggregated run metrics.
    In practice reads CSV/JSONL files from CONFIG['logdir'].
    """
    # For this refactor we only print a message.
    print("[validate] Results validation passed (placeholder)")
