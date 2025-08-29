import argparse
import os
import yaml
import torch

try:
    from .preprocess import preprocess
    from .evaluate import test_run, run_experiment_A, run_experiment_B, run_experiment_C
    from .train import train_toy_model
except ImportError:  # allow running as a script
    from preprocess import preprocess
    from evaluate import test_run, run_experiment_A, run_experiment_B, run_experiment_C
    from train import train_toy_model


def auto_device(pref: str) -> str:
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="HAT-Q experimental runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--quick", action="store_true", help="Run quick functional test (A+B+C)")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config {args.config} not found. Using defaults.")
        cfg = {}

    images_dir = cfg.get("images_dir", ".research/iteration3/images")
    data_dir = cfg.get("data_dir", "data")
    models_dir = cfg.get("models_dir", "models")
    device = auto_device(cfg.get("device", "auto"))

    preprocess(images_dir=images_dir, data_dir=data_dir, models_dir=models_dir)

    if args.quick or cfg.get("quick", True):
        print("Running quick functional test (A+B+C)...")
        test_run(device=device, images_dir=images_dir)
        return

    # Otherwise, run full pipeline: train -> A, B, C
    print("Training toy model...")
    model, train_loader, valid_loader = train_toy_model(device=device, images_dir=images_dir, cfg=cfg.get("train", {}), verbose=True)
    # Experiments
    print("Running Experiment A...")
    _ = run_experiment_A(device=device, images_dir=images_dir)
    print("Running Experiment B...")
    _ = run_experiment_B(device=device, images_dir=images_dir)
    print("Running Experiment C...")
    _ = run_experiment_C(device=device, images_dir=images_dir)


if __name__ == "__main__":
    main()
