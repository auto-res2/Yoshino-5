import argparse
import os
import json
from typing import List

import numpy as np
import torch

# Support running as a script or as a package
try:
    from .preprocess import set_seed, default_warmup_texts, build_warmup_loader, load_model_and_tokenizer, get_image_dir
    from .train import patch_model_with_hiqua, HiQuAPredictor, warmup_train_predictor
    from .evaluate import (
        decode_benchmark,
        run_once_decode,
        plot_training_loss,
        plot_tokens_per_second,
        plot_budget_sweep,
        oracle_delta_kl,
        predictor_scores,
        plot_predictor_correlation,
        plot_overhead_bar,
    )
except ImportError:  # pragma: no cover - fallback when executed as script
    from preprocess import set_seed, default_warmup_texts, build_warmup_loader, load_model_and_tokenizer, get_image_dir
    from train import patch_model_with_hiqua, HiQuAPredictor, warmup_train_predictor
    from evaluate import (
        decode_benchmark,
        run_once_decode,
        plot_training_loss,
        plot_tokens_per_second,
        plot_budget_sweep,
        oracle_delta_kl,
        predictor_scores,
        plot_predictor_correlation,
        plot_overhead_bar,
    )

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML must be installed. Add it to requirements.txt.")


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def experiment1(cfg: dict):
    print("=== Experiment 1 — End-to-end accuracy/latency/memory under a promotion budget ===")
    set_seed(cfg.get('seed', 42))
    model_name = cfg.get('model_name', 'hf-internal-testing/tiny-random-LlamaForCausalLM')
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    warmup_steps = int(cfg.get('warmup_steps', 10))
    decode_tokens = int(cfg.get('decode_tokens', 16))
    target_ratio = float(cfg.get('target_ratio', 0.15))

    model, tokenizer, device = load_model_and_tokenizer(model_name, device)
    model = patch_model_with_hiqua(model, chunks=int(cfg.get('chunks', 8)), group_size=int(cfg.get('group_size', 64)))

    predictor = HiQuAPredictor(hidden_size=model.config.hidden_size, chunks=int(cfg.get('chunks', 8)), depth=2, d_model=128, k_ctx=8).to(device)

    texts = cfg.get('warmup_texts', None) or default_warmup_texts()
    loader = build_warmup_loader(tokenizer, texts, max_len=int(cfg.get('warmup_max_len', 96)), batch_size=int(cfg.get('warmup_batch_size', 1)))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    losses = warmup_train_predictor(model, predictor, loader, steps=warmup_steps, lr=float(cfg.get('warmup_lr', 5e-4)), device=device)

    img1 = plot_training_loss(losses, fname='training_loss_hiqua.pdf')

    prompts = cfg.get('prompts', [
        "Q: What is the derivative of x^2? A:",
        "Question: Who wrote 1984? Answer:",
        "Compute: 159 + 284 =",
        "Translate: Bonjour -> English:",
    ])
    stats = decode_benchmark(model, predictor, tokenizer, prompts, max_new_tokens=decode_tokens, target_ratio=target_ratio)

    img2 = plot_tokens_per_second(stats, fname='tokens_per_second_hiqua.pdf')

    print("[Exp1] Decode stats:")
    print(json.dumps(stats, indent=2))
    print(f"[Exp1] Saved figures: {img1}, {img2}")


def experiment2(cfg: dict):
    print("=== Experiment 2 — Ablations: residual codebook and budget controller ===")
    set_seed(cfg.get('seed', 123))
    model_name = cfg.get('model_name', 'hf-internal-testing/tiny-random-LlamaForCausalLM')
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer, device = load_model_and_tokenizer(model_name, device)
    model = patch_model_with_hiqua(model, chunks=int(cfg.get('chunks', 8)), group_size=int(cfg.get('group_size', 64)))
    predictor = HiQuAPredictor(model.config.hidden_size, chunks=int(cfg.get('chunks', 8))).to(device)

    texts = cfg.get('warmup_texts', None) or default_warmup_texts()
    loader = build_warmup_loader(tokenizer, texts, max_len=int(cfg.get('warmup_max_len', 96)), batch_size=int(cfg.get('warmup_batch_size', 1)))
    _ = warmup_train_predictor(model, predictor, loader, steps=int(cfg.get('warmup_steps', 10)), lr=float(cfg.get('warmup_lr', 5e-4)), device=device)

    prompt = cfg.get('ablation_prompt', 'Q: Simplify 24/6 + 7. A:')
    budgets = cfg.get('budgets', [0.05, 0.10, 0.15, 0.20])
    tps_list = []
    ratio_list = []
    for r in budgets:
        tok_s, ratio = run_once_decode(model, predictor, tokenizer, prompt, max_new_tokens=int(cfg.get('decode_tokens', 96)), target_ratio=float(r), use_bandit=True)
        print(f"[Exp2] Full HiQuA r*={float(r):.2f} -> tokens/s={tok_s:.2f}, avg_ratio={ratio:.3f}")
        tps_list.append(tok_s); ratio_list.append(ratio)

    # No residual
    from train import set_residual_enabled
    set_residual_enabled(model, False)
    tok_s_nores, ratio_nores = run_once_decode(model, predictor, tokenizer, prompt, max_new_tokens=int(cfg.get('decode_tokens', 96)), target_ratio=float(cfg.get('target_ratio', 0.15)), use_bandit=True)
    set_residual_enabled(model, True)

    # No bandit
    tok_s_noband, ratio_noband = run_once_decode(model, predictor, tokenizer, prompt, max_new_tokens=int(cfg.get('decode_tokens', 96)), target_ratio=float(cfg.get('target_ratio', 0.15)), use_bandit=False)

    img_tps, img_ratio = plot_budget_sweep(budgets, tps_list, ratio_list)

    results = {
        "budgets": budgets,
        "tokens_per_s": tps_list,
        "achieved_ratio": ratio_list,
        "no_residual": {"tokens_per_s": tok_s_nores, "ratio": ratio_nores},
        "no_bandit": {"tokens_per_s": tok_s_noband, "ratio": ratio_noband},
    }
    print("[Exp2] Results summary:")
    print(json.dumps(results, indent=2))
    print(f"[Exp2] Saved figures: {img_tps}, {img_ratio}")


def experiment3(cfg: dict):
    print("=== Experiment 3 — Predictor quality, robustness proxy, and micro-overhead ===")
    set_seed(cfg.get('seed', 7))
    model_name = cfg.get('model_name', 'hf-internal-testing/tiny-random-LlamaForCausalLM')
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer, device = load_model_and_tokenizer(model_name, device)
    model = patch_model_with_hiqua(model, chunks=int(cfg.get('chunks', 8)), group_size=int(cfg.get('group_size', 64)))
    predictor = HiQuAPredictor(model.config.hidden_size, chunks=int(cfg.get('chunks', 8))).to(device)

    # Light warm-up
    texts = cfg.get('warmup_texts', [
        "Physics: Force equals mass times acceleration.",
        "Biology: DNA encodes genes.",
    ])
    loader = build_warmup_loader(tokenizer, texts, max_len=int(cfg.get('warmup_max_len', 96)), batch_size=int(cfg.get('warmup_batch_size', 1)))
    _ = warmup_train_predictor(model, predictor, loader, steps=int(cfg.get('warmup_steps', 10)), lr=float(cfg.get('warmup_lr', 5e-4)), device=device)

    # A) Oracle correlation and AUC-like
    samples = cfg.get('predictor_samples', [
        "Q: What is the capital of Germany? A:",
        "Compute: 51 * 7 =",
        "Explain photosynthesis briefly.",
        "Translate: Bonjour ->",
    ])
    spearmans = []
    auc_likes = []
    for txt in samples:
        from train import HiQuAPredictor  # to hint types for the linter only
        delta = oracle_delta_kl(model, predictor, tokenizer, txt, layer_idx=0)
        scores = predictor_scores(model, predictor, tokenizer, txt)
        y_true = []
        y_score = []
        offset = 0
        for name in ['q', 'k', 'v', 'o', 'up', 'gate', 'down']:
            d = np.array(delta[name])  # more negative => more important
            s = scores[offset:offset + predictor.chunks]
            y_true.append((-d).tolist())
            y_score.append(s.tolist())
            offset += predictor.chunks
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        # Spearman and pairwise concordance
        sp = 0.0 if (np.std(y_true) < 1e-8 or np.std(y_score) < 1e-8) else float(np.corrcoef(np.argsort(y_true), np.argsort(y_score))[0,1])
        try:
            if len(y_true) > 1:
                sp = spearman_corr(y_true, y_score)
        except Exception:
            pass
        pairs = min(2000, max(2, len(y_true) * 2))
        idx = np.random.randint(0, len(y_true), size=(pairs, 2))
        conc = float(np.mean(((y_true[idx[:, 0]] - y_true[idx[:, 1]]) * (y_score[idx[:, 0]] - y_score[idx[:, 1]])) > 0))
        spearmans.append(sp)
        auc_likes.append(conc)
    print("[Exp3] Predictor-Oracle correlation:")
    print(json.dumps({"spearman_per_sample": spearmans, "auc_like_per_sample": auc_likes, "spearman_mean": float(np.mean(spearmans)), "auc_like_mean": float(np.mean(auc_likes))}, indent=2))
    img_corr = plot_predictor_correlation(spearmans, auc_likes)

    # C) Microbenchmark overhead (base vs ~15% promoted randomly)
    prompt = cfg.get('overhead_prompt', 'A ' * 256)
    inp = tokenizer(prompt, return_tensors='pt').to(next(model.parameters()).device)
    L = len(model.model.layers)
    names = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
    chunks = int(cfg.get('chunks', 8))
    all_false = {k: torch.zeros(1, chunks, dtype=torch.bool, device=inp.input_ids.device) for k in names}
    g0 = [all_false for _ in range(L)]
    if torch.cuda.is_available():
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        torch.cuda.synchronize(); start.record()
        with hiqua_gating(g0):
            _ = model(**inp)
        end.record(); torch.cuda.synchronize(); t0 = float(start.elapsed_time(end))
    else:
        t0_ = time.time();
        with hiqua_gating(g0):
            _ = model(**inp)
        t0 = float((time.time() - t0_) * 1000.0)

    g1 = []
    for _ in range(L):
        g1.append({k: (torch.rand(1, chunks, device=inp.input_ids.device) < 0.15) for k in names})
    if torch.cuda.is_available():
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        torch.cuda.synchronize(); start.record()
        with hiqua_gating(g1):
            _ = model(**inp)
        end.record(); torch.cuda.synchronize(); t1 = float(start.elapsed_time(end))
    else:
        t1_ = time.time();
        with hiqua_gating(g1):
            _ = model(**inp)
        t1 = float((time.time() - t1_) * 1000.0)

    overhead_ms = max(0.0, t1 - t0)
    print(json.dumps({"overhead_ms": overhead_ms, "base_ms": t0, "promoted_ms": t1}, indent=2))
    img_over = plot_overhead_bar(t0, t1)
    print(f"[Exp3] Saved figures: {img_corr}, {img_over}")


def main():
    parser = argparse.ArgumentParser(description='HiQuA Emulator Experiments')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Ensure image directory exists
    os.makedirs(get_image_dir(), exist_ok=True)

    runs = cfg.get('runs', ['exp1', 'exp2', 'exp3'])
    if 'exp1' in runs:
        experiment1(cfg)
    if 'exp2' in runs:
        experiment2(cfg)
    if 'exp3' in runs:
        experiment3(cfg)

    print('All selected experiments completed. Figures saved to', get_image_dir())


if __name__ == '__main__':
    main()
