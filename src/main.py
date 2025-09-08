"""src/main.py
Entry-point that wires everything together.
Execute with  `python -m src.main` from project root.
"""
from __future__ import annotations

import os
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import yaml

from .preprocess import get_stream, make_loader, train_tf, val_tf
from .train import BackboneFactory, GostScheduler, Trainer
from .evaluate import ContinualMetrics

# -----------------------------------------------------------------------------
#  Misc helpers
# -----------------------------------------------------------------------------

def _dict_to_namespace(d):
    """Recursively convert a dict to SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


def load_cfg(cfg_path: str | os.PathLike = "config/config.yaml") -> SimpleNamespace:
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#  Experiment driver
# -----------------------------------------------------------------------------

def run_experiment():
    cfg = load_cfg()
    Path(cfg.SYSTEM.results_dir).mkdir(parents=True, exist_ok=True)

    backbone = BackboneFactory.build(cfg)
    scheduler = GostScheduler(cfg, backbone)
    trainer = Trainer(cfg, backbone, scheduler)

    metrics = ContinualMetrics()

    for stream_cfg in cfg.DATA.streams:
        for ordering in stream_cfg.orderings:
            for seed in cfg.SYSTEM.seeds:
                set_all_seeds(seed)
                stream = get_stream(stream_cfg.name, stream_cfg.n_experiences, ordering)

                for task_id, exp in enumerate(stream.train_stream):
                    print(f"Seed={seed} | {stream_cfg.name} | {ordering} | Task={task_id}")
                    train_set = exp.dataset.map_transforms({"train": train_tf})
                    val_set = exp.dataset.map_transforms({"eval": val_tf})

                    train_loader = make_loader(train_set, stream_cfg.batch_size, cfg)
                    val_loader = make_loader(val_set, stream_cfg.batch_size, cfg, shuffle=False)

                    trainer.train_task(exp, train_loader, val_loader, task_id)
                    scheduler.observe_task(task_id, train_set)

                # Placeholder for metric aggregation per stream
                metrics.update(task_id, [])

    print("Experiment finished ✔")


if __name__ == "__main__":
    run_experiment()
