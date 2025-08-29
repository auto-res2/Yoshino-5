import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

# Support running as a script (no package) and as a module
try:
    from .preprocess import run as preprocess_run
    from .train import build_models
    from .evaluate import run_all
except ImportError:
    from preprocess import run as preprocess_run
    from train import build_models
    from evaluate import run_all


def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        print(f"Config file not found at {cfg_path}, using defaults.")
        return {}
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TinyPRM-Cascade (toy) runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) or {}

    # Set deterministic seeds
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Paths
    data_dir = Path(cfg.get("data_dir", "data"))
    models_dir = Path(cfg.get("models_dir", "models"))
    output_dir = Path(cfg.get("output_dir", ".research/iteration1"))
    # Force all images to be saved in the required directory
    images_dir = Path(".research/iteration2/images")

    # Preprocess (toy dataset)
    preprocess_run(data_dir, n_train=int(cfg.get("n_train", 200)))

    # Build toy models (no heavy training in this toy implementation)
    models = build_models(cfg)

    # Evaluate experiments 1–3 and save plots in images_dir
    summary = run_all(models, output_dir, images_dir, cfg)

    # Print concise summary to stdout
    print("\n===== TinyPRM-Cascade (toy) Summary =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
