"""
main.py – experiment orchestration / entry-point
Run with:  python -m src.main  --exp <exp1|exp2|exp3|all>
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, List

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Subset

from .preprocess import _set_global_seed, get_cifar100_split
from .train import (TaskCLIPScheduler, DERPlusPlusWrapper, train_task)
from .evaluate import verify_implementation, validate_results

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
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(DEVICE)
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
