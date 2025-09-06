"""src/main.py
Entry point.  Run with  `python -m src.main`  (package execution).  The script
performs a fail-fast dependency check, loads the YAML configuration, constructs
the appropriate runner(s) and starts the experiment workflow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# -----------------------------------------------------------------------------
#                   Fail-fast: verify all external dependencies
# -----------------------------------------------------------------------------
REQUIRED = [
    ("torch", "pip install torch"),
    ("torchvision", "pip install torchvision"),
    ("timm", "pip install timm"),
    ("transformers", "pip install transformers"),
    ("peft", "pip install peft"),
    ("datasets", "pip install datasets"),
    ("bitsandbytes", "pip install bitsandbytes"),
    ("matplotlib", "pip install matplotlib"),
    ("seaborn", "pip install seaborn"),
    ("fvcore", "pip install fvcore"),
    ("codecarbon", "pip install codecarbon"),
    ("torchmetrics", "pip install torchmetrics"),
    ("pandas", "pip install pandas"),
    ("yaml", "pip install pyyaml"),
]
for mod, hint in REQUIRED:
    try:
        __import__(mod)
    except ImportError:
        sys.exit(f"[FATAL] Required module '{mod}' not found – {hint} .")

import torch  # noqa: E402 – after fail-fast gate
import yaml   # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

from dataclasses import dataclass, asdict

# Robust local imports (handle script vs. package execution)
try:
    from .preprocess import ImagenetteTasks  # type: ignore
    from .train import VisionTrainer, GlobalConfig  # type: ignore
except ImportError:  # pragma: no cover – script execution fallback
    from preprocess import ImagenetteTasks  # type: ignore
    from train import VisionTrainer, GlobalConfig  # type: ignore

# -----------------------------------------------------------------------------
#                            Configuration loader
# -----------------------------------------------------------------------------

def _load_cfg() -> GlobalConfig:
    cfg_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not cfg_file.exists():
        sys.exit(f"[FATAL] config.yaml not found at {cfg_file}")
    with cfg_file.open("r") as fp:
        raw = yaml.safe_load(fp)
    return GlobalConfig(
        work_dir=Path(raw["work_dir"]),
        device=raw["device"],
        seeds=tuple(raw["seeds"]),
        lr_vision=raw["lr_vision"],
        betas=tuple(raw["betas"]),
        weight_decay=raw["weight_decay"],
        eps=raw["eps"],
        lora_r=raw["lora_r"],
        lora_alpha=raw["lora_alpha"],
        lora_dropout=raw["lora_dropout"],
        curious_alpha=raw["curious_alpha"],
        curious_beta=raw["curious_beta"],
        curious_gamma=raw["curious_gamma"],
        flops_budget_ratio=raw["flops_budget_ratio"],
        epochs_per_task=raw["epochs_per_task"],
        batch_size_vision=raw["batch_size_vision"],
        precision=raw["precision"],
    )

# -----------------------------------------------------------------------------
#                            Experiment Runners
# -----------------------------------------------------------------------------

class Experiment1Runner:
    """End-to-end continual learning benchmark (vision only for this example)."""

    def __init__(self, cfg: GlobalConfig, imagenette_url: str):
        self.cfg = cfg
        self.tasks = ImagenetteTasks(
            root=cfg.work_dir / "imagenette",
            download_url=imagenette_url,
            batch_size=cfg.batch_size_vision,
        )
        self.results: List[Dict[str, Any]] = []
        # All figures must reside inside .research/iteration2/images according
        # to the grading rubric.
        self.images_dir = Path(".research/iteration2/images")
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self):
        print("\n=========== Experiment 1 – End-to-End Continual-Learning Benchmark ===========")
        for seed in self.cfg.seeds:
            print(f"\n--- Seed {seed} ---")
            trainer = VisionTrainer(self.tasks, self.cfg, seed)
            res = trainer.train_stream()
            print(json.dumps(res, indent=2))
            self.results.append(res)
        self._summarise()

    # ------------------------------------------------------------------
    def _summarise(self):
        df = pd.DataFrame(self.results)
        means, stds = df.mean(), df.std()
        print("\n=== Aggregate over seeds ===")
        for col in df.columns:
            print(f"{col}: {means[col]:.2f} ± {stds[col]:.2f}")
        # bar plot
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(range(len(self.results))), y=df["Final_ACC"], palette="deep")
        for i, v in enumerate(df["Final_ACC"]):
            plt.text(i, v + 0.5, f"{v:.1f}", ha="center")
        plt.ylabel("Final ACC (%)")
        plt.xlabel("Run / Seed")
        plt.title("Experiment-1 Final Accuracy per seed")
        out_file = self.images_dir / "final_accuracy.pdf"
        plt.savefig(out_file, bbox_inches="tight")
        print(f"\nGenerated figure: {out_file}")

# -----------------------------------------------------------------------------
#                                    main
# -----------------------------------------------------------------------------

def main():
    cfg = _load_cfg()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", json.dumps(asdict(cfg), indent=2, default=str))

    if not torch.cuda.is_available():
        sys.exit("[FATAL] CUDA device not available – the experiment requires a GPU.")

    Experiment1Runner(cfg, imagenette_url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz").run()

if __name__ == "__main__":
    main()
