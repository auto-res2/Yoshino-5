"""src/main.py
Entry-point that orchestrates the full benchmark (Mini-CTrL-40) **and** a small
smoke-test (CIFAR-10-C).  The configuration is read from `config/config.yaml`
with PyYAML, converted into an AttrDict and then forwarded to the remaining
modules.
"""
from __future__ import annotations

import json, argparse
from pathlib import Path
from typing import Dict

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from .train import AttrDict, set_seed, ContinualLearner
from .preprocess import MiniCtrl40Stream, CIFARSmoke

# --------------------------------------------------------------------
EXPERIMENTS = ["EXP1_FULL_BENCHMARK", "EXP2_SMOKE"]


def load_cfg(cfg_path: Path) -> AttrDict:
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    return AttrDict(**raw)


# --------------------------------------------------------------------
# EXPERIMENT 1 --------------------------------------------------------
# --------------------------------------------------------------------

def exp1(cfg: AttrDict):
    print("\n================  EXPERIMENT 1 – Full Benchmark  ================")
    stream = MiniCtrl40Stream(cfg, cfg.batch_size_vision)
    scheds = ["CURIOUS", "Random", "Chrono", "dOTDD", "GradOnly"]
    aggregated: Dict[str, list] = {s: [] for s in scheds}
    for sd in cfg.seeds:
        set_seed(sd)
        for sname in scheds:
            print(f"[Seed {sd}] Scheduler={sname}")
            learner = ContinualLearner(stream, cfg, sname)
            res = learner.train()
            aggregated[sname].append(res)
            print(json.dumps(res, indent=2))

    # quick bar-plot of final accuracy --------------------------------
    fig_acc = Path(cfg.paths.fig_dir) / "final_accuracy.pdf"
    fig_acc.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    means = [sum(d["Final_ACC"] for d in aggregated[s]) / len(aggregated[s]) for s in scheds]
    sns.barplot(x=scheds, y=means)
    for i, v in enumerate(means):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center")
    plt.ylabel("Final ACC (%)")
    plt.savefig(fig_acc, bbox_inches="tight")
    print("[INFO] Figure saved →", fig_acc)


# --------------------------------------------------------------------
# EXPERIMENT 2 --------------------------------------------------------
# --------------------------------------------------------------------

def exp2(cfg: AttrDict):
    print("\n================  EXPERIMENT 2 – Smoke Test  ================")
    stream = CIFARSmoke(cfg, cfg.batch_size_vision)
    scheds = ["CURIOUS", "CleanFirst", "CorruptFirst"]
    orders = {"CleanFirst": [0, 1], "CorruptFirst": [1, 0]}
    results: Dict[str, Dict[str, float]] = {}

    for s in scheds:
        if s == "CURIOUS":
            learner = ContinualLearner(stream, cfg, "CURIOUS")
        else:  # monkey-patch deterministic order into Chrono scheduler
            learner = ContinualLearner(stream, cfg, "Chrono")
            learner.sched.order = lambda n_tasks, order=orders[s]: order
        res = learner.train()
        print(json.dumps(res, indent=2))
        results[s] = res

    # bar-plot of BWT --------------------------------------------------
    fig_bwt = Path(cfg.paths.fig_dir) / "bwt_smoke.pdf"
    plt.figure(); vals = [results[s]["BWT"] for s in scheds]
    sns.barplot(x=scheds, y=vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.2, f"{v:.1f}", ha="center")
    plt.ylabel("BWT (%)")
    plt.savefig(fig_bwt, bbox_inches="tight")
    print("[INFO] Figure saved →", fig_bwt)

# --------------------------------------------------------------------
# main ----------------------------------------------------------------
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=Path(__file__).resolve().parents[1] / "config" / "config.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.fig_dir.mkdir(parents=True, exist_ok=True)

    print("[CONFIG]", json.dumps(cfg.to_dict(), indent=2))
    exp1(cfg)
    exp2(cfg)


if __name__ == "__main__":
    main()
