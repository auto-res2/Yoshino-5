# -*- coding: utf-8 -*-
"""
Main entry point for running EXACT experiments end-to-end.
Run from project root using:  python -m src.main  or  python src/main.py
This will:
  - Generate synthetic teacher data (toy) for math and commonsense
  - Verify steps, extract minimal core, build datasets
  - Train three models per task: EXACT, CoT-KD baseline, Answer-only baseline
  - Evaluate and save high-quality PDF plots under .research/iteration3/images
"""
import os
import json
from typing import Dict, List

import torch
import yaml

# Support running as a module or as a script
try:
    from .preprocess import (
        set_seed, get_device, ensure_dir,
        make_synthetic_math_teacher_data,
        build_exact_examples_from_teacher_math,
        ToyRetriever, make_synthetic_commonsense_teacher_data, build_exact_examples_from_teacher_csqa,
    )
    from .train import get_tokenizer_and_model, train_exact_language_model
    from .evaluate import (
        evaluate_experiment1_math,
        evaluate_experiment2_commonsense,
        evaluate_experiment3_mgsm,
    )
except Exception:  # noqa: E722
    from preprocess import (  # type: ignore
        set_seed, get_device, ensure_dir,
        make_synthetic_math_teacher_data,
        build_exact_examples_from_teacher_math,
        ToyRetriever, make_synthetic_commonsense_teacher_data, build_exact_examples_from_teacher_csqa,
    )
    from train import get_tokenizer_and_model, train_exact_language_model  # type: ignore
    from evaluate import (  # type: ignore
        evaluate_experiment1_math,
        evaluate_experiment2_commonsense,
        evaluate_experiment3_mgsm,
    )


DEFAULT_CONFIG_PATH = os.path.join("config", "config.yaml")
IMAGES_DIR = os.path.join(".research", "iteration3", "images")
MODELS_DIR = os.path.join("models")


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_env(device: str):
    print("[Environment] torch=", torch.__version__)
    print("[Environment] device=", device)
    if device == "cuda":
        print("[Environment] CUDA device count=", torch.cuda.device_count())
        print("[Environment] CUDA name=", torch.cuda.get_device_name(0))


def _save_training_loss_fig(losses: List[float], title: str, filename: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5,3))
    sns.lineplot(x=list(range(len(losses))), y=losses)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Training loss")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def run():
    cfg_path = os.environ.get("EXACT_CONFIG", DEFAULT_CONFIG_PATH)
    cfg = _load_config(cfg_path)
    seed = int(cfg.get("general", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("general", {}).get("device", "auto"))
    _print_env(device)

    ensure_dir(IMAGES_DIR)
    ensure_dir(MODELS_DIR)

    # ========= EXPERIMENT 1: Math =========
    if cfg.get("exp1", {}).get("run", True):
        print("\n===== EXPERIMENT 1 (Math) — START =====")
        toy_n = int(cfg.get("exp1", {}).get("n", 12))
        model_name = cfg.get("exp1", {}).get("model_name", "sshleifer/tiny-gpt2")
        epochs = int(cfg.get("exp1", {}).get("epochs", 2))
        batch_size = int(cfg.get("exp1", {}).get("batch_size", 2))
        lr = float(cfg.get("exp1", {}).get("lr", 5e-5))
        core_weight = float(cfg.get("exp1", {}).get("core_weight", 2.0))
        lambda_neg = float(cfg.get("exp1", {}).get("lambda_neg", 0.5))
        schedule = cfg.get("exp1", {}).get("schedule", [0.0, 0.5])
        use_lora = bool(cfg.get("exp1", {}).get("use_lora", False))

        tokenizer, _ = get_tokenizer_and_model(model_name, device)  # model created later per run
        teacher = make_synthetic_math_teacher_data(n=toy_n)
        exact_items = build_exact_examples_from_teacher_math(teacher, tokenizer)
        split = max(1, int(0.8 * len(exact_items)))
        train_items = exact_items[:split]
        val_items = exact_items[split:]
        # Baselines
        cot_train = [{**ex, "core_indices": [i for i in range(len(ex["steps"]))], "rejected_mask": [0]*len(ex["steps"]) } for ex in train_items]
        ans_only_train = [{**ex, "core_indices": [], "rejected_mask": [0]*len(ex["steps"]) } for ex in train_items]

        print("[Train] Training EXACT model (Math)...")
        _, exact_model = tokenizer, None
        exact_model, exact_logs = train_exact_language_model(
            train_items, tokenizer, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=core_weight, lambda_neg=lambda_neg, curriculum_schedule=schedule, use_lora=use_lora, device=device)
        _save_training_loss_fig(exact_logs["train_losses"], "EXACT training loss (Math)", os.path.join(IMAGES_DIR, "training_loss_exp1_exact.pdf"))

        print("[Train] Training CoT-KD baseline (Math)...")
        cot_model, cot_logs = train_exact_language_model(
            cot_train, tokenizer, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=1.0, lambda_neg=0.0, curriculum_schedule=[0.0]*epochs, use_lora=use_lora, device=device)
        _save_training_loss_fig(cot_logs["train_losses"], "CoT-KD training loss (Math)", os.path.join(IMAGES_DIR, "training_loss_exp1_baseline.pdf"))

        print("[Train] Training Answer-only baseline (Math)...")
        ans_model, ans_logs = train_exact_language_model(
            ans_only_train, tokenizer, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=1.0, lambda_neg=0.0, curriculum_schedule=[1.0]*epochs, use_lora=use_lora, device=device)

        models_math = {"exact": exact_model, "cot": cot_model, "ans": ans_model}
        res_exp1 = evaluate_experiment1_math(models_math, tokenizer, val_items, images_dir=IMAGES_DIR, device=device)
        print("===== EXPERIMENT 1 (Math) — END =====\n")
    else:
        res_exp1 = None
        tokenizer = None
        exact_model = None

    # ========= EXPERIMENT 2: Commonsense =========
    if cfg.get("exp2", {}).get("run", True):
        print("\n===== EXPERIMENT 2 (Commonsense) — START =====")
        toy_n = int(cfg.get("exp2", {}).get("n", 9))
        model_name = cfg.get("exp2", {}).get("model_name", "sshleifer/tiny-gpt2")
        epochs = int(cfg.get("exp2", {}).get("epochs", 2))
        batch_size = int(cfg.get("exp2", {}).get("batch_size", 2))
        lr = float(cfg.get("exp2", {}).get("lr", 5e-5))
        core_weight = float(cfg.get("exp2", {}).get("core_weight", 2.0))
        lambda_neg = float(cfg.get("exp2", {}).get("lambda_neg", 0.6))
        schedule = cfg.get("exp2", {}).get("schedule", [0.0, 0.5])
        use_lora = bool(cfg.get("exp2", {}).get("use_lora", False))

        tokenizer2, _ = get_tokenizer_and_model(model_name, device)
        # Build toy retriever
        toy_passages = [
            "Dolphins are mammals and breathe air.",
            "Sharks are fishes and have gills.",
            "Eagles are birds of prey.",
            "Bees collect nectar from flowers to make honey.",
            "Winter is usually the coldest season.",
        ]
        retriever = ToyRetriever(toy_passages)
        teacher = make_synthetic_commonsense_teacher_data(n=toy_n)
        exact_items = build_exact_examples_from_teacher_csqa(teacher, retriever)
        split = max(1, int(0.8 * len(exact_items)))
        train_items = exact_items[:split]
        val_items = exact_items[split:]

        cot_train = [{**ex, "core_indices": [i for i in range(len(ex["steps"]))], "rejected_mask": [0]*len(ex["steps"]) } for ex in train_items]
        ans_only_train = [{**ex, "core_indices": [], "rejected_mask": [0]*len(ex["steps"]) } for ex in train_items]

        print("[Train] Training EXACT model (Commonsense)...")
        exact_model_cs, exact_logs_cs = train_exact_language_model(
            train_items, tokenizer2, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=core_weight, lambda_neg=lambda_neg, curriculum_schedule=schedule, use_lora=use_lora, device=device)
        _save_training_loss_fig(exact_logs_cs["train_losses"], "EXACT training loss (CS)", os.path.join(IMAGES_DIR, "training_loss_exp2_exact.pdf"))

        print("[Train] Training CoT-KD baseline (Commonsense)...")
        cot_model_cs, cot_logs_cs = train_exact_language_model(
            cot_train, tokenizer2, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=1.0, lambda_neg=0.0, curriculum_schedule=[0.0]*epochs, use_lora=use_lora, device=device)

        print("[Train] Training Answer-only baseline (Commonsense)...")
        ans_model_cs, ans_logs_cs = train_exact_language_model(
            ans_only_train, tokenizer2, model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr,
            core_weight=1.0, lambda_neg=0.0, curriculum_schedule=[1.0]*epochs, use_lora=use_lora, device=device)

        models_cs = {"exact": exact_model_cs, "cot": cot_model_cs, "ans": ans_model_cs}
        # Wrap verify fn for eval usage
        def verify_fn(problem: str, step: str):
            return build_exact_examples_from_teacher_csqa  # placeholder to satisfy linter (unused)
        # Build a closure that calls retriever-backed verifier
        def _verify(problem: str, step: str):
            try:
                from .preprocess import verify_step_rag  # type: ignore
            except Exception:  # noqa: E722
                from preprocess import verify_step_rag  # type: ignore
            return verify_step_rag(problem, step, retriever)

        res_exp2 = evaluate_experiment2_commonsense(models_cs, tokenizer2, val_items, verify_fn=_verify, images_dir=IMAGES_DIR, device=device)
        print("===== EXPERIMENT 2 (Commonsense) — END =====\n")
    else:
        res_exp2 = None

    # ========= EXPERIMENT 3: Cross-lingual MGSM-style =========
    if cfg.get("exp3", {}).get("run", True):
        print("\n===== EXPERIMENT 3 (Cross-lingual MGSM-style) — START =====")
        # Reuse tokenizer/model from exp1 EXACT if available; else create tiny
        if tokenizer is None:
            model_name = cfg.get("exp3", {}).get("model_name", "sshleifer/tiny-gpt2")
            tokenizer, model_fallback = get_tokenizer_and_model(model_name, device)
            model_use = model_fallback
        else:
            model_use = None
        if res_exp1 is not None:
            # We trained models in this process; rebuild from locals
            # The EXACT model from exp1 scope was named exact_model
            # To access it here, we fetch from Python's locals/globals; instead, re-train tiny if unavailable
            try:
                model_use = exact_model  # type: ignore # noqa
            except Exception:
                pass
        if model_use is None:
            # as a safe fallback
            model_name = cfg.get("exp3", {}).get("model_name", "sshleifer/tiny-gpt2")
            tokenizer, model_use = get_tokenizer_and_model(model_name, device)
        res_exp3 = evaluate_experiment3_mgsm(model_use, tokenizer, images_dir=IMAGES_DIR, device=device)
        print("===== EXPERIMENT 3 (Cross-lingual MGSM-style) — END =====\n")
    else:
        res_exp3 = None

    # Summarize
    print("[Summary] Figures saved under:", IMAGES_DIR)
    expected_figs = [
        "training_loss_exp1_exact.pdf", "training_loss_exp1_baseline.pdf", "accuracy_exp1.pdf",
        "faithfulness_exp1.pdf", "epr_exp1.pdf", "cot_length_exp1.pdf", "inference_latency_exp1.pdf",
        "training_loss_exp2_exact.pdf", "accuracy_exp2.pdf", "hallucination_exp2.pdf", "cot_length_exp2.pdf",
        "confusion_matrix_exp2_exact.pdf", "confusion_matrix_exp2_baseline.pdf",
        "accuracy_exp3.pdf", "cot_length_exp3.pdf"
    ]
    for fn in expected_figs:
        path = os.path.join(IMAGES_DIR, fn)
        print(f" - {path} {'(exists)' if os.path.exists(path) else '(missing — may be skipped in config or due to optional deps)'}")


def main():
    run()


if __name__ == "__main__":
    main()
