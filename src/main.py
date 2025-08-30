import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch

try:
    from .preprocess import set_seed, get_device, ensure_dir, SyntheticDataConfig, generate_language_data, generate_classification_data
    from .train import build_model_and_controller, TrainConfig, train_controller_and_scales
    from .evaluate import evaluate_ppl, decode_throughput, evaluate_classification, compute_offline_labels, run_experiment3_controller, run_experiment2_microbench
except ImportError:  # fallback when running as a script
    from preprocess import set_seed, get_device, ensure_dir, SyntheticDataConfig, generate_language_data, generate_classification_data
    from train import build_model_and_controller, TrainConfig, train_controller_and_scales
    from evaluate import evaluate_ppl, decode_throughput, evaluate_classification, compute_offline_labels, run_experiment3_controller, run_experiment2_microbench


def run_experiment1_end2end(images_dir: str, device: torch.device, fast: bool = True):
    print("[Experiment 1] End-to-end HATQ effectiveness on synthetic tasks")
    os.makedirs(images_dir, exist_ok=True)

    vocab_size = 256
    d_model = 64 if fast else 128
    n_layers = 2 if fast else 4
    n_heads = 4
    d_ff = d_model * 4
    planes = 8
    max_len = 512
    model, ctrl = build_model_and_controller(vocab_size, d_model, n_layers, n_heads, d_ff, planes, max_len, device)
    gb_rnn, mask_block, budget_head = ctrl

    data_cfg = SyntheticDataConfig(vocab_size=vocab_size, seq_len=64 if fast else 128, batch_size=8, n_batches=64)
    train_data = generate_language_data(data_cfg, device)
    val_data = generate_language_data(SyntheticDataConfig(vocab_size=vocab_size, seq_len=64, batch_size=8, n_batches=16), device)

    tr_cfg = TrainConfig(steps=100 if fast else 400, lr=2e-3, lambda_nll=0.8)
    print("[Experiment 1] Training controller + scales (short) ...")
    losses = train_controller_and_scales(model, train_data, ctrl, tr_cfg, print_every=25 if fast else 50)

    plt.figure(figsize=(5,4))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training loss (controller+scales)')
    plt.savefig(os.path.join(images_dir, "training_loss_hatq.pdf"), bbox_inches="tight")
    plt.close()

    ppl_hatq = evaluate_ppl(model, val_data, ctrl, ablation='hatq')
    ppl_global = evaluate_ppl(model, val_data, ctrl, ablation='global_only')
    ppl_static = evaluate_ppl(model, val_data, None, ablation='controller_off')
    print(f"[Experiment 1] Perplexity: HATQ={ppl_hatq:.2f} | global-only={ppl_global:.2f} | static-4b={ppl_static:.2f}")

    Xc, yc = generate_classification_data(n_samples=256 if fast else 512, seq_len=32, vocab_size=vocab_size, device=device)
    acc_hatq, cm_hatq = evaluate_classification(model, Xc, yc, ctrl, ablation='hatq')
    acc_global, _ = evaluate_classification(model, Xc, yc, ctrl, ablation='global_only')
    acc_static, cm_static = evaluate_classification(model, Xc, yc, None, ablation='controller_off')
    print(f"[Experiment 1] Classification accuracy: HATQ={acc_hatq:.3f} | global-only={acc_global:.3f} | static-4b={acc_static:.3f}")

    if cm_hatq is not None:
        plt.figure(figsize=(4,3))
        sns.heatmap(cm_hatq, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion matrix (HATQ)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(images_dir, "confusion_matrix_hatq.pdf"), bbox_inches="tight")
        plt.close()

    lengths = [16, 64, 128] if fast else [16, 128, 512]
    tps_hatq, tps_static, bits_hatq = [], [], []
    mem_hatq_list, mem_static_list = [], []
    for L in lengths:
        prompts = torch.randint(0, vocab_size, (4, L), device=device)
        tps_h, mem_h, bits_h = decode_throughput(model, prompts, ctrl, max_new_tokens=min(64, 2*L), ablation='hatq')
        tps_s, mem_s, bits_s = decode_throughput(model, prompts, None, max_new_tokens=min(64, 2*L), ablation='controller_off')
        tps_hatq.append(tps_h); tps_static.append(tps_s)
        bits_hatq.append(bits_h); mem_hatq_list.append(mem_h); mem_static_list.append(mem_s)
        print(f"[Experiment 1] L={L} tokens/sec: HATQ={tps_h:.1f}, static-4b={tps_s:.1f}; avg_bits(HATQ)={bits_h:.2f}")

    plt.figure(figsize=(5,4))
    plt.plot(lengths, tps_hatq, marker='o', label='HATQ')
    plt.plot(lengths, tps_static, marker='s', label='static-4b')
    plt.xlabel('Total length (prompt+gen)')
    plt.ylabel('Tokens/sec (synthetic)')
    plt.title('Throughput comparison')
    plt.legend()
    plt.savefig(os.path.join(images_dir, "tokens_per_sec_hatq_vs_static.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5,4))
    bars = ['HATQ', 'global-only', 'static-4b']
    vals = [acc_hatq, acc_global, acc_static]
    sns.barplot(x=bars, y=vals)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Classification accuracy')
    plt.savefig(os.path.join(images_dir, "accuracy_hatq_vs_static.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5,4))
    sns.histplot(bits_hatq, bins=8, kde=False)
    plt.xlabel('Effective avg bits (per run)')
    plt.title('Distribution of avg bits (HATQ)')
    plt.savefig(os.path.join(images_dir, "bits_histogram_hatq.pdf"), bbox_inches="tight")
    plt.close()

    print("[Experiment 1] Plots saved to:")
    for fn in ["training_loss_hatq.pdf", "tokens_per_sec_hatq_vs_static.pdf", "accuracy_hatq_vs_static.pdf", "confusion_matrix_hatq.pdf", "bits_histogram_hatq.pdf"]:
        print(" -", os.path.join(images_dir, fn))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--run_all', action='store_true', help='Override config and run all experiments')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get('seed', 123))
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    images_dir = cfg.get('images_dir', '.research/iteration2/images')
    ensure_dir(images_dir)

    do_run_all = args.run_all or bool(cfg.get('run_all', True))
    fast = bool(cfg.get('fast', True))

    if do_run_all:
        run_experiment1_end2end(images_dir, device, fast=fast)
        _ = run_experiment2_microbench(images_dir, device=device)
        vocab_size = 256
        d_model = 64 if fast else 128
        n_layers = 2 if fast else 4
        n_heads = 4
        d_ff = d_model * 4
        planes = 8
        max_len = 512
        model, ctrl = build_model_and_controller(vocab_size, d_model, n_layers, n_heads, d_ff, planes, max_len, device)
        data_cfg = SyntheticDataConfig(vocab_size=vocab_size, seq_len=64, batch_size=8, n_batches=16)
        train_data = generate_language_data(data_cfg, device)
        tr_cfg = TrainConfig(steps=80 if fast else 200, lr=2e-3, lambda_nll=0.8)
        print("[Main] Training controller for Experiment 3 (short)...")
        _ = train_controller_and_scales(model, train_data, ctrl, tr_cfg, print_every=20 if fast else 50)
        labels_offline = compute_offline_labels(model, train_data[:4])
        _ = run_experiment3_controller(model, train_data[:2], ctrl, images_dir, labels_offline)
        print("[Main] All experiments completed. Check generated PDF figures in:", images_dir)
    else:
        print("==== HATQ Smoke Test (quick functionality) ====")
        vocab_size = 128
        d_model = 64
        model, ctrl = build_model_and_controller(vocab_size, d_model, n_layers=2, n_heads=4, d_ff=128, planes=8, max_len=256, device=device)
        data_cfg = SyntheticDataConfig(vocab_size=vocab_size, seq_len=48, batch_size=4, n_batches=16)
        train_data = generate_language_data(data_cfg, device)
        tr_cfg = TrainConfig(steps=30, lr=3e-3, lambda_nll=0.8)
        print("[Smoke] Training controller + scales (very short)...")
        _ = train_controller_and_scales(model, train_data, ctrl, tr_cfg, print_every=10)
        val_data = generate_language_data(SyntheticDataConfig(vocab_size=vocab_size, seq_len=48, batch_size=4, n_batches=4), device)
        ppl = evaluate_ppl(model, val_data, ctrl, ablation='hatq')
        print(f"[Smoke] Perplexity(HATQ) ~ {ppl:.2f}")
        prompts = torch.randint(0, vocab_size, (2, 24), device=device)
        tps, mem, bits = decode_throughput(model, prompts, ctrl, max_new_tokens=16, ablation='hatq')
        print(f"[Smoke] tokens/sec={tps:.1f}, peak_mem={mem if mem>=0 else 'N/A'}, avg_bits={bits:.2f}")
        _ = run_experiment2_microbench(images_dir, device=device)
        labels_offline = compute_offline_labels(model, train_data, tau=0.01)
        _ = run_experiment3_controller(model, train_data[:2], ctrl, images_dir, labels_offline)
        print("==== Smoke Test Completed — PDF figures saved in:", images_dir)


if __name__ == "__main__":
    main()
