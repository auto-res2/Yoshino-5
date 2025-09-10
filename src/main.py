from __future__ import annotations
"""
main.py – experiment orchestration / entry-point
Run with:  python -m src.main  --exp <exp1|exp2|exp3|all>
This file is expected to be executable both
  1. as a module  (python -m src.main)  ➜  __package__ == "src"
  2. as a script  (python src/main.py)  ➜  __package__ is None / ""
Relative imports ("from .preprocess import …") only work in case (1).
The previous implementation crashed in case (2) with
    ImportError: attempted relative import with no known parent package
We now handle both situations gracefully:
    • When executed as a script we manually append the project root to
      sys.path and fall back to absolute (top-level) imports.
    • When executed as a module we keep the original relative imports.
No functional behaviour of the program is affected – we only improve
import robustness.
"""

import argparse
import pathlib
import sys
from typing import Any, List

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Dynamic import handling (see doc-string above) -----------------------------
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    # Running as a script – add project root so that `import src.xxx` works
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Fallback to absolute imports (src.<module>)
    from src.preprocess import _set_global_seed, get_cifar100_split  # type: ignore
    from src.train import TaskCLIPScheduler, DERPlusPlusWrapper, train_task  # type: ignore
    from src.evaluate import verify_implementation, validate_results  # type: ignore
else:
    # Running with the package context – keep relative imports
    from .preprocess import _set_global_seed, get_cifar100_split  # noqa: D401
    from .train import TaskCLIPScheduler, DERPlusPlusWrapper, train_task  # noqa: D401
    from .evaluate import verify_implementation, validate_results  # noqa: D401

# ---------------------------------------------------------------------------
# Load configuration --------------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: dict[str, Any] = yaml.safe_load(f)

DEVICE = CONFIG["device"] if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Experiment 1 – core benchmark on Split CIFAR-100 --------------------------
# ---------------------------------------------------------------------------

def run_experiment_1():
    out_dir = pathlib.Path(CONFIG["logdir"]) / CONFIG["exp_ids"]["exp01"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in CONFIG["seed_list"]:
        _set_global_seed(seed)

        # Data ----------------------------------------------------------
        train_ds, _test_ds, tasks = get_cifar100_split("./data")

        # Scheduler -----------------------------------------------------
        scheduler = TaskCLIPScheduler(num_tasks=len(tasks), feature_dim=16)

        # Learner (DER++) ----------------------------------------------
        weights_enum = torchvision.models.ResNet18_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
        backbone = torchvision.models.resnet18(weights=weights_enum).to(DEVICE)
        learner = DERPlusPlusWrapper(backbone)

        buffer: List[int] = []
        for t_id, task in enumerate(tasks):
            buffer.append(t_id)
            # decide when to schedule
            if len(buffer) >= 8 or t_id == len(tasks) - 1:
                ordering = scheduler.propose_order(buffer)
                for next_id in ordering:
                    dl = DataLoader(
                        Subset(train_ds, tasks[next_id]["train_idx"]),
                        batch_size=CONFIG["training"]["batch_size"],
                        shuffle=True,
                        num_workers=4,
                    )
                    # In a full implementation we would compute ACC before training.
                    acc_after = train_task(
                        learner,
                        dl,
                        CONFIG["training"]["epochs_per_task"]["cifar"],
                        amp=CONFIG["mixed_precision"],
                    )
                    dF, dT = 0.0, acc_after  # surrogate placeholders
                    prev_task = buffer[0] if buffer else next_id
                    scheduler.update_after_task(prev_task, next_id, dF, dT, reward=acc_after)
                buffer.clear()

    verify_implementation()
    validate_results()

# ---------------------------------------------------------------------------
# Stubs for Exp 2 & 3 -------------------------------------------------------
# ---------------------------------------------------------------------------

def run_experiment_2():
    print("Experiment 2 stub – implement as needed.")


def run_experiment_3():
    print("Experiment 3 stub – implement as needed.")

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["exp1", "exp2", "exp3", "all"], help="Which experiment(s) to run")
    args = parser.parse_args()

    if args.exp in ("exp1", "all"):
        run_experiment_1()
    if args.exp in ("exp2", "all"):
        run_experiment_2()
    if args.exp in ("exp3", "all"):
        run_experiment_3()


if __name__ == "__main__":
    main()
