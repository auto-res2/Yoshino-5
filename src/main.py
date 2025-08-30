import os
import json
import yaml
import random
from typing import List

try:
    from .train import (
        set_seed,
        ensure_dir,
        Problem,
        load_dataset,
        experiment3_distillation,
        save_model,
        StudentController,
        load_model,
    )
    from .evaluate import experiment1_end_to_end, experiment2_causal_ablation
    from .preprocess import preprocess_main
except ImportError:  # fallback when running as a script
    from train import (
        set_seed,
        ensure_dir,
        Problem,
        load_dataset,
        experiment3_distillation,
        save_model,
        StudentController,
        load_model,
    )
    from evaluate import experiment1_end_to_end, experiment2_causal_ablation
    from preprocess import preprocess_main


DEFAULT_CFG = {
    "seed": 42,
    "images_dir": ".research/iteration2/images",
    "data_path": "data/synthetic.jsonl",
    "models_dir": "models",
    "results_dir": "data",
    "budgets": [6e9, 1.2e10, 2.4e10],
    "distill_flop_cap": 1.2e10,
    "train": {"epochs": 3, "lr": 1e-3, "train_size": 40, "val_size": 20},
    "dataset": {"n_total": 60, "include_traps": True},
}


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
        # shallow merge
        cfg = DEFAULT_CFG.copy()
        cfg.update(user or {})
        # nested merges
        for k in ["train", "dataset"]:
            if k in user:
                tmp = cfg[k].copy()
                tmp.update(user[k] or {})
                cfg[k] = tmp
        return cfg
    return DEFAULT_CFG.copy()


def split_dataset(all_probs: List[Problem], train_size: int, val_size: int, seed: int):
    random.Random(seed).shuffle(all_probs)
    assert train_size + val_size <= len(all_probs), "Not enough samples for the requested split"
    train_probs = all_probs[:train_size]
    val_probs = all_probs[train_size : train_size + val_size]
    return train_probs, val_probs


def main():
    cfg_path = os.path.join("config", "config.yaml")
    cfg = load_config(cfg_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    images_dir = cfg.get("images_dir", ".research/iteration2/images")
    models_dir = cfg.get("models_dir", "models")
    results_dir = cfg.get("results_dir", "data")
    ensure_dir(images_dir)
    ensure_dir(models_dir)
    ensure_dir(results_dir)

    # 1) Preprocess (generate synthetic dataset)
    data_path = cfg.get("data_path", "data/synthetic.jsonl")
    if not os.path.exists(data_path):
        preprocess_main(cfg)
    else:
        print(f"[Preprocess] Using existing dataset at {data_path}")

    all_probs = load_dataset(data_path)

    # 2) Split
    train_size = int(cfg.get("train", {}).get("train_size", 40))
    val_size = int(cfg.get("train", {}).get("val_size", 20))
    train_probs, val_probs = split_dataset(all_probs, train_size, val_size, seed)
    print(f"[Split] train={len(train_probs)} val={len(val_probs)} (from N={len(all_probs)})")

    # 3) Train (distillation)
    epochs = int(cfg.get("train", {}).get("epochs", 3))
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    flop_cap = float(cfg.get("distill_flop_cap", 1.2e10))

    print("\n========== TRAINING (Experiment 3) ==========")
    exp3 = experiment3_distillation(train_probs, val_probs, images_dir, epochs=epochs, lr=lr, flop_cap=flop_cap)
    student_model = exp3["student_model"]

    model_path = os.path.join(models_dir, "student_model.pt")
    save_model(student_model, model_path)
    print(f"[Model] Saved student model to {model_path}")

    # Controller
    controller = StudentController(student_model, device="cpu")

    # 4) Evaluate (Experiment 1)
    print("\n========== EVALUATION (Experiment 1) ==========")
    budgets = [float(x) for x in cfg.get("budgets", [6e9, 1.2e10, 2.4e10])]
    exp1_json = os.path.join(results_dir, "exp1_results.json")
    _ = experiment1_end_to_end(val_probs, budgets, controller, images_dir, exp1_json)
    print(f"[Results] Saved Experiment 1 results JSON to {exp1_json}")

    # 5) Evaluate (Experiment 2)
    print("\n========== EVALUATION (Experiment 2) ==========")
    # Build reference logs at mid budget using BACS
    mid_cap = float(budgets[min(1, len(budgets) - 1)])
    try:
        from .train import SyntheticBackbone, SimplePRM, DEFAULT_COSTS, run_episode, Tools
    except ImportError:
        from train import SyntheticBackbone, SimplePRM, DEFAULT_COSTS, run_episode, Tools

    backbone = SyntheticBackbone()
    prm = SimplePRM()
    tools = Tools()

    ref_logs = {}
    for pb in val_probs:
        ep = run_episode(pb, backbone, prm, lambda p, t, ps, f: controller.decide(p, t, ps, f, lam=1.0), DEFAULT_COSTS, mid_cap, tools)
        ref_logs[pb.pid] = ep

    exp2_json = os.path.join(results_dir, "exp2_results.json")
    _ = experiment2_causal_ablation(val_probs, ref_logs, controller, images_dir, exp2_json)
    print(f"[Results] Saved Experiment 2 results JSON to {exp2_json}")

    print("\n===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()
