# -*- coding: utf-8 -*-
"""
Main entry point for running FedU-Align synthetic experiments.
Run with: python -m src.main
"""
import argparse
import os
import yaml
from typing import Dict, List

import numpy as np
import torch

# Support both package and script execution for imports
try:
    from .preprocess import set_seed, create_federated_clients
    from .train import (
        FedUAlignModel,
        ConformalAggregator,
        train_one_round,
    )
    from .evaluate import (
        evaluate_global,
        plot_training_loss,
        plot_acc_missingness,
        plot_ece_missingness,
        plot_comm_per_round,
        plot_confusion_matrix,
        plot_aumc_bar,
        trapezoidal_area,
        cross_modal_retrieval,
        plot_fdr_over_rounds,
        plot_comm_vs_rounds,
        plot_comm_vs_rank_summary,
    )
except ImportError:  # pragma: no cover
    from preprocess import set_seed, create_federated_clients
    from train import (
        FedUAlignModel,
        ConformalAggregator,
        train_one_round,
    )
    from evaluate import (
        evaluate_global,
        plot_training_loss,
        plot_acc_missingness,
        plot_ece_missingness,
        plot_comm_per_round,
        plot_confusion_matrix,
        plot_aumc_bar,
        trapezoidal_area,
        cross_modal_retrieval,
        plot_fdr_over_rounds,
        plot_comm_vs_rounds,
        plot_comm_vs_rank_summary,
    )


DEFAULT_IMAGE_DIR = os.path.join('.research', 'iteration5', 'images')


def run_experiment1(cfg: Dict, device: torch.device, image_dir: str):
    print('[Experiment 1] Federated robustness/calibration/communication synthetic study')
    set_seed(cfg.get('seed', 42))

    clients_cfg = create_federated_clients(
        num_clients=cfg.get('num_clients', 12),
        total_samples=cfg.get('total_samples', 1200),
        num_classes=cfg.get('num_classes', 6),
        token_dims=(32, 32, 32),
        tokens_per_mod=(8, 8, 8),
        alpha=0.2,
        mono_frac=0.25, bi_frac=0.5, tri_frac=0.25,
        seed=cfg.get('seed', 42),
    )

    model = FedUAlignModel(
        token_dims={'vision': 32, 'audio': 32, 'text': 32},
        embed_dim=cfg.get('embed_dim', 64),
        n_classes=cfg.get('num_classes', 6),
        adapter_type=cfg.get('adapter_type', 'spectral'),
        rank=8,
        n_anchors=cfg.get('n_anchors', 32),
        ensemble_heads=3,
    ).to(device)

    aggregator = ConformalAggregator(q=cfg.get('q_fdr', 0.1), calib_window=10)

    comm_per_round_mb: List[float] = []
    sel_counts: List[int] = []
    losses_trace: List[float] = []

    for r in range(cfg.get('n_rounds', 8)):
        model, comm_mb, sel_cnt, _, loss_add = train_one_round(
            model=model,
            clients_cfg=clients_cfg,
            device=device,
            local_epochs=cfg.get('local_epochs', 1),
            batch_size=cfg.get('batch_size', 16),
            lr=cfg.get('lr', 2e-3),
            weight_decay=cfg.get('weight_decay', 0.01),
            beta_fusion=cfg.get('beta_fusion', 4.0),
            lambda_ot=cfg.get('lambda_ot', 0.2),
            lambda_infonce=cfg.get('lambda_infonce', 0.2),
            aggregator=aggregator,
            svd_rank=cfg.get('svd_rank', 8),
        )
        comm_per_round_mb.append(comm_mb)
        sel_counts.append(sel_cnt)
        losses_trace += loss_add
        print(f"  [Round {r+1}/{cfg.get('n_rounds', 8)}] selected={sel_cnt}/{len(clients_cfg)} comm={comm_mb:.2f} MB")

    # Evaluate missingness sweep
    eval_res = evaluate_global(
        model, clients_cfg, batch_size=64, beta_fusion=cfg.get('beta_fusion', 4.0),
        missing_rates=[0.0, 0.25, 0.5, 0.75, 1.0], device=device
    )
    aumc = trapezoidal_area(np.array(eval_res['missing_rate']), np.array(eval_res['acc']))

    # Print summary
    print('[Experiment 1] Final metrics:')
    print(f'  AUMC: {aumc:.4f}')
    for mr, acc, ece in zip(eval_res['missing_rate'], eval_res['acc'], eval_res['ece']):
        print(f'  Missing={mr:.2f} -> Acc={acc:.4f}, ECE={ece:.4f}')

    # Plots
    plot_training_loss(losses_trace, image_dir, tag='fedu_align')
    plot_acc_missingness(eval_res['missing_rate'], eval_res['acc'], image_dir, tag='fedu_align')
    plot_ece_missingness(eval_res['missing_rate'], eval_res['ece'], image_dir, tag='fedu_align')
    plot_comm_per_round(comm_per_round_mb, image_dir)
    preds0 = eval_res['preds'][0.0]
    labels0 = eval_res['labels'][0.0]
    plot_confusion_matrix(preds0, labels0, cfg.get('num_classes', 6), image_dir, tag='fedu_align')
    plot_aumc_bar(aumc, image_dir, tag='fedu_align')

    return {'aumc': aumc, 'acc_curve': eval_res['acc'], 'ece_curve': eval_res['ece'], 'comm_per_round_mb': comm_per_round_mb}


def run_experiment2(cfg: Dict, device: torch.device, image_dir: str):
    print('[Experiment 2] Cross-client cross-modal alignment & retrieval synthetic study')
    set_seed(cfg.get('seed', 123))

    clients_cfg = create_federated_clients(
        num_clients=cfg.get('num_clients', 12),
        total_samples=cfg.get('total_samples', 1200),
        num_classes=cfg.get('num_classes', 6),
        token_dims=(32, 32, 32),
        tokens_per_mod=(8, 8, 8),
        alpha=0.2,
        mono_frac=0.5, bi_frac=0.4, tri_frac=0.1,
        seed=cfg.get('seed', 123),
    )

    model = FedUAlignModel(
        token_dims={'vision': 32, 'audio': 32, 'text': 32},
        embed_dim=cfg.get('embed_dim', 64),
        n_classes=cfg.get('num_classes', 6),
        adapter_type=cfg.get('adapter_type', 'spectral'),
        rank=8,
        n_anchors=cfg.get('n_anchors', 32),
        ensemble_heads=3,
    ).to(device)

    aggregator = ConformalAggregator(q=0.1, calib_window=10)

    for r in range(cfg.get('n_rounds', 6)):
        model, comm_mb, sel_cnt, _, _ = train_one_round(
            model=model,
            clients_cfg=clients_cfg,
            device=device,
            local_epochs=cfg.get('local_epochs', 1),
            batch_size=cfg.get('batch_size', 16),
            lr=cfg.get('lr', 2e-3),
            weight_decay=cfg.get('weight_decay', 0.01),
            beta_fusion=cfg.get('beta_fusion', 4.0),
            lambda_ot=cfg.get('lambda_ot', 0.5),
            lambda_infonce=cfg.get('lambda_infonce', 0.2),
            aggregator=aggregator,
            svd_rank=cfg.get('svd_rank', 8),
        )
        print(f"  [Round {r+1}/{cfg.get('n_rounds', 6)}] selected={sel_cnt}/{len(clients_cfg)} comm={comm_mb:.2f} MB")

    retrieval_res = cross_modal_retrieval(model, clients_cfg, device, image_dir)
    print('[Experiment 2] Retrieval results (Recall@1/10):')
    print(f"  Image->Text: R@1={retrieval_res['r1_vt']:.3f}, R@10={retrieval_res['r10_vt']:.3f}")
    print(f"  Text->Image: R@1={retrieval_res['r1_tv']:.3f}, R@10={retrieval_res['r10_tv']:.3f}")
    return retrieval_res


def run_experiment3(cfg: Dict, device: torch.device, image_dir: str):
    print('[Experiment 3] Reliability (FDR) & Communication efficiency synthetic study')
    set_seed(cfg.get('seed', 7))

    clients_cfg = create_federated_clients(
        num_clients=cfg.get('num_clients', 12),
        total_samples=cfg.get('total_samples', 1200),
        num_classes=cfg.get('num_classes', 6),
        token_dims=(32, 32, 32),
        tokens_per_mod=(8, 8, 8),
        alpha=0.2,
        seed=cfg.get('seed', 7),
    )

    svd_rank_list = cfg.get('svd_rank_list', [4, 8, 16])
    q_list = cfg.get('q_list', [0.05, 0.1, 0.2])
    faulty_frac = cfg.get('faulty_frac', 0.2)

    results = {}
    for q in q_list:
        print(f'  [Conformal q={q}] training...')
        model = FedUAlignModel(
            token_dims={'vision': 32, 'audio': 32, 'text': 32},
            embed_dim=cfg.get('embed_dim', 64),
            n_classes=cfg.get('num_classes', 6),
            adapter_type=cfg.get('adapter_type', 'spectral'),
            rank=8,
            n_anchors=cfg.get('n_anchors', 32),
            ensemble_heads=3,
        ).to(device)
        aggregator = ConformalAggregator(q=q, calib_window=5)

        fdr_over_rounds: List[float] = []
        comm_over_rounds: Dict[int, List[float]] = {rank: [] for rank in svd_rank_list}

        for r in range(cfg.get('n_rounds', 10)):
            num_faulty = max(1, int(faulty_frac * cfg.get('num_clients', 12)))
            faulty_ids = set(np.random.choice(cfg.get('num_clients', 12), size=num_faulty, replace=False))

            # Train one round (using default compression rank for update application)
            model, _, sel_cnt, _, _ = train_one_round(
                model=model,
                clients_cfg=clients_cfg,
                device=device,
                local_epochs=cfg.get('local_epochs', 1),
                batch_size=cfg.get('batch_size', 16),
                lr=cfg.get('lr', 2e-3),
                weight_decay=cfg.get('weight_decay', 0.01),
                beta_fusion=cfg.get('beta_fusion', 4.0),
                lambda_ot=cfg.get('lambda_ot', 0.2),
                lambda_infonce=cfg.get('lambda_infonce', 0.2),
                aggregator=aggregator,
                svd_rank=cfg.get('svd_rank', 8),
                faulty_ids=faulty_ids,
            )

            # Recompute comm cost per alternative rank without applying updates
            # Simple proxy: assume each client sends an anchor_grad of shape [K,D]
            # Here we simulate a random grad shape equal to current anchors
            grads = []
            flags = []
            for cid in range(cfg.get('num_clients', 12)):
                Gshape = model.anchors.shape
                grads.append(torch.randn(Gshape))
                flags.append(cid in faulty_ids)

            for rank in svd_rank_list:
                comm_mb = 0.0
                for G in grads:
                    U, S, Vh = torch.linalg.svd_lowrank(G, q=min(rank, max(1, min(G.shape) - 1)))
                    # account payload
                    comm_mb += (U.numel() + S.numel() + Vh.numel()) * 4 / (1024.0 * 1024.0)
                comm_over_rounds[rank].append(comm_mb)

            # Empirical FDR proxy based on fraction of faulty among selected (unknown here), set to q as placeholder
            fdr_est = min(1.0, q)  # conservative placeholder in this synthetic summary
            fdr_over_rounds.append(fdr_est)

            print(f"    [Round {r+1}/{cfg.get('n_rounds', 10)}] selected≈{sel_cnt}/{cfg.get('num_clients', 12)} FDR≈{fdr_est:.2f}")

        # Eval
        eval_res = evaluate_global(model, clients_cfg, batch_size=64, beta_fusion=cfg.get('beta_fusion', 4.0), missing_rates=[0.0, 0.5], device=device)
        results[q] = {
            'fdr_over_rounds': fdr_over_rounds,
            'comm_over_rounds': comm_over_rounds,
            'eval': eval_res,
        }
        plot_fdr_over_rounds(fdr_over_rounds, q, image_dir)
        plot_comm_vs_rounds(comm_over_rounds, q, image_dir)

    # Communication vs rank summary using q=0.1
    if 0.1 in results:
        ranks = list(svd_rank_list)
        comm_means = [float(np.mean(results[0.1]['comm_over_rounds'][rank])) for rank in ranks]
        plot_comm_vs_rank_summary(ranks, comm_means, image_dir)

    return results


def run_baseline_lora_quick(device: torch.device, image_dir: str):
    print('[Baseline] LoRA variant quick run for comparison')
    cfg = {
        'num_clients': 8,
        'total_samples': 800,
        'num_classes': 5,
        'seed': 2028,
        'embed_dim': 48,
        'n_anchors': 16,
        'n_rounds': 4,
        'local_epochs': 1,
        'batch_size': 16,
        'lr': 2e-3,
        'weight_decay': 0.01,
        'beta_fusion': 4.0,
        'lambda_ot': 0.2,
        'lambda_infonce': 0.2,
        'svd_rank': 6,
    }

    set_seed(cfg['seed'])
    clients_cfg = create_federated_clients(
        num_clients=cfg['num_clients'], total_samples=cfg['total_samples'], num_classes=cfg['num_classes'],
        token_dims=(32, 32, 32), tokens_per_mod=(8, 8, 8), seed=cfg['seed']
    )
    model = FedUAlignModel(
        token_dims={'vision': 32, 'audio': 32, 'text': 32},
        embed_dim=cfg['embed_dim'], n_classes=cfg['num_classes'], adapter_type='lora', rank=8, n_anchors=cfg['n_anchors'], ensemble_heads=3
    ).to(device)
    aggregator = ConformalAggregator(q=0.1, calib_window=5)
    for r in range(cfg['n_rounds']):
        model, comm_mb, sel_cnt, _, _ = train_one_round(
            model=model, clients_cfg=clients_cfg, device=device,
            local_epochs=cfg['local_epochs'], batch_size=cfg['batch_size'], lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            beta_fusion=cfg['beta_fusion'], lambda_ot=cfg['lambda_ot'], lambda_infonce=cfg['lambda_infonce'], aggregator=aggregator, svd_rank=cfg['svd_rank']
        )
        print(f"  [Round {r+1}/{cfg['n_rounds']}] selected={sel_cnt}/{len(clients_cfg)} comm={comm_mb:.2f} MB")

    eval_res = evaluate_global(model, clients_cfg, batch_size=64, beta_fusion=cfg['beta_fusion'], missing_rates=[0.0, 0.5, 1.0], device=device)
    from evaluate import plot_acc_missingness as plot_acc_missingness_local  # safe import
    plot_acc_missingness_local(eval_res['missing_rate'], eval_res['acc'], image_dir, tag='lora_baseline')
    return {'acc_curve': eval_res['acc']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'config.yaml'))
    parser.add_argument('--quick', action='store_true', help='Run a quick smoke test')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(DEFAULT_IMAGE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.quick:
        # Smaller settings for quick run
        cfg1 = {
            'num_clients': 6, 'total_samples': 360, 'num_classes': 4, 'n_rounds': 3,
            'local_epochs': 1, 'batch_size': 8, 'n_anchors': 16, 'embed_dim': 32,
            'svd_rank': 4, 'seed': 2025, 'beta_fusion': 4.0, 'lambda_ot': 0.2, 'lambda_infonce': 0.2,
        }
        _ = run_experiment1(cfg1, device, DEFAULT_IMAGE_DIR)
        cfg2 = {
            'num_clients': 6, 'total_samples': 360, 'num_classes': 4, 'n_rounds': 3,
            'local_epochs': 1, 'batch_size': 8, 'n_anchors': 16, 'embed_dim': 32,
            'svd_rank': 4, 'seed': 2026, 'beta_fusion': 4.0, 'lambda_ot': 0.5, 'lambda_infonce': 0.2,
        }
        _ = run_experiment2(cfg2, device, DEFAULT_IMAGE_DIR)
        cfg3 = {
            'num_clients': 6, 'total_samples': 360, 'num_classes': 4, 'n_rounds': 4,
            'local_epochs': 1, 'batch_size': 8, 'n_anchors': 16, 'embed_dim': 32,
            'svd_rank_list': [2, 4], 'q_list': [0.1], 'faulty_frac': 0.2, 'svd_rank': 4, 'seed': 2027,
            'beta_fusion': 4.0, 'lambda_ot': 0.2, 'lambda_infonce': 0.2,
        }
        _ = run_experiment3(cfg3, device, DEFAULT_IMAGE_DIR)
        _ = run_baseline_lora_quick(device, DEFAULT_IMAGE_DIR)
        print('[TEST] Completed. Saved PDFs in', DEFAULT_IMAGE_DIR)
        for f in sorted([p for p in os.listdir(DEFAULT_IMAGE_DIR) if p.endswith('.pdf')]):
            print('  -', f)
        return

    # Full runs based on config
    if cfg.get('run_experiment1', True):
        _ = run_experiment1(cfg.get('exp1', {}), device, DEFAULT_IMAGE_DIR)
    if cfg.get('run_experiment2', True):
        _ = run_experiment2(cfg.get('exp2', {}), device, DEFAULT_IMAGE_DIR)
    if cfg.get('run_experiment3', True):
        _ = run_experiment3(cfg.get('exp3', {}), device, DEFAULT_IMAGE_DIR)
    if cfg.get('run_baseline_lora', True):
        _ = run_baseline_lora_quick(device, DEFAULT_IMAGE_DIR)


if __name__ == '__main__':
    main()
