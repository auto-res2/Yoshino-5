import os
import argparse
import time
import yaml
import numpy as np
import torch

# Support running both as a package (python -m src.main) and as a script (python src/main.py)
try:
    from .preprocess import (
        set_seed,
        get_device,
        make_linear_tasks,
        make_vision_tasks,
        make_nlp_tasks,
        build_M_from_signatures,
        greedy_order_from_M,
        StormController,
        compute_signature_from_features,
    )
    from .train import (
        train_sequence_synthetic,
        train_sequence_vision,
        train_sequence_nlp,
    )
    from .evaluate import (
        compute_forgetting_and_stability,
        plot_accuracy_curves,
        plot_losses,
    )
except ImportError:  # fallback for script execution
    from preprocess import (
        set_seed,
        get_device,
        make_linear_tasks,
        make_vision_tasks,
        make_nlp_tasks,
        build_M_from_signatures,
        greedy_order_from_M,
        StormController,
        compute_signature_from_features,
    )
    from train import (
        train_sequence_synthetic,
        train_sequence_vision,
        train_sequence_nlp,
    )
    from evaluate import (
        compute_forgetting_and_stability,
        plot_accuracy_curves,
        plot_losses,
    )

IMAGES_DIR = os.path.join('.research', 'iteration2', 'images')


def run_experiment1_synthetic(cfg):
    device = get_device()
    seed = cfg['seed']
    set_seed(seed)
    quick = cfg['quick']
    T = 10 if not quick else 6
    ntr = 4000 if not quick else 1000
    nte = 1000 if not quick else 400
    angle_pattern = cfg.get('angle_pattern', 'sweep')

    print(f"[Exp1] Synthetic | seed={seed} | pattern={angle_pattern} | T={T} | device={device}")
    tasks, _ = make_linear_tasks(T=T, n_train=ntr, n_test=nte, d=64, D=256, angle_pattern=angle_pattern, seed=seed)

    # Signatures and pairwise M
    t0 = time.time()
    G = [compute_signature_from_features(t['phi_tr'], t['ytr'], num_classes=2, epochs=1, device=device) for t in tasks]
    M = build_M_from_signatures(G, alpha=1.0, beta=1.0)
    t1 = time.time()
    print(f"[Exp1] Signatures computed in {t1 - t0:.3f}s")

    # Orders
    order_storm = greedy_order_from_M(M)
    order_rand = np.random.default_rng(seed).permutation(T).tolist()
    order_sim = greedy_order_from_M(build_M_from_signatures(G, alpha=1.0, beta=0.0))
    order_inter = greedy_order_from_M(build_M_from_signatures(G, alpha=0.0, beta=1.0))

    # Train per order
    acc_mats = {}
    loss_curves = {}
    results = {}
    for name, order in [('storm', order_storm), ('random', order_rand), ('sim_only', order_sim), ('inter_only', order_inter)]:
        acc_mat, losses = train_sequence_synthetic(tasks, order, epochs_per_task=(2 if quick else 3), device=device)
        acc_mats[name] = acc_mat
        loss_curves[name] = losses
        metrics = compute_forgetting_and_stability(acc_mat)
        results[name] = metrics
        print(f"[Exp1][{name}] final_avg_acc={metrics['final_avg_acc']:.4f} | avg_forgetting={metrics['avg_forgetting']:.4f} | min_acc={metrics['min_acc']:.4f} | wc_drop={metrics['wc_drop']:.4f}")

    t2 = time.time()
    print(f"[Exp1] Total runtime (includes training orders) ~ {t2 - t0:.2f}s. Probe share ~ {(t1 - t0) / max(1e-6, (t2 - t0)) * 100:.1f}%.")

    # Plots
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_accuracy_curves(acc_mats, save_path=os.path.join(IMAGES_DIR, 'exp1_accuracy_storm_vs_baselines.pdf'))
    plot_losses(loss_curves, save_path=os.path.join(IMAGES_DIR, 'exp1_training_loss_storm_vs_baselines.pdf'))
    print(f"[Exp1] Saved plots into {IMAGES_DIR}")

    return results


def run_experiment2_vision(cfg):
    device = get_device()
    seed = cfg['seed']
    set_seed(seed)
    quick = cfg['quick']
    dataset = cfg.get('dataset', 'cifar10')
    num_tasks = cfg.get('num_tasks', 5)
    buffer_size = cfg.get('buffer_size', 3)

    print(f"[Exp2] Vision CL | dataset={dataset} | seed={seed} | device={device}")
    tasks, total_classes, backbone_state, feat_dim, controller = make_vision_tasks(
        dataset=dataset, num_tasks=num_tasks, seed=seed, samples_per_class=(200 if not quick else 60), quick=quick
    )

    # Orders
    order_storm = controller.greedy_order(list(range(len(tasks))))
    order_random = np.random.default_rng(seed).permutation(len(tasks)).tolist()
    controller_sim = StormController(alpha=1.0, beta=0.0, buffer_size=buffer_size)
    controller_sim.G = controller.G
    order_sim = controller_sim.greedy_order(list(range(len(tasks))))

    # Train and evaluate
    acc_mats = {}
    loss_curves = {}
    results = {}

    print(f"[Exp2] Training STORM order: {order_storm}")
    acc_storm, loss_storm = train_sequence_vision(tasks, order_storm, backbone_state, feat_dim, total_classes, device=device, quick=quick)
    print(f"[Exp2] Training Random order: {order_random}")
    acc_rand, loss_rand = train_sequence_vision(tasks, order_random, backbone_state, feat_dim, total_classes, device=device, quick=quick)
    print(f"[Exp2] Training Similarity-only order: {order_sim}")
    acc_sim, loss_sim = train_sequence_vision(tasks, order_sim, backbone_state, feat_dim, total_classes, device=device, quick=quick)

    acc_mats['storm'] = acc_storm
    acc_mats['random'] = acc_rand
    acc_mats['sim_only'] = acc_sim
    loss_curves['storm'] = loss_storm
    loss_curves['random'] = loss_rand
    loss_curves['sim_only'] = loss_sim

    for k, A in acc_mats.items():
        m = compute_forgetting_and_stability(A)
        results[k] = m
        print(f"[Exp2][{k}] final_avg_acc={m['final_avg_acc']:.4f} | avg_forgetting={m['avg_forgetting']:.4f} | min_acc={m['min_acc']:.4f} | wc_drop={m['wc_drop']:.4f}")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_accuracy_curves(acc_mats, save_path=os.path.join(IMAGES_DIR, 'exp2_accuracy_vision_storm_vs_baselines.pdf'))
    plot_losses(loss_curves, save_path=os.path.join(IMAGES_DIR, 'exp2_training_loss_vision_storm_vs_baselines.pdf'))
    print(f"[Exp2] Saved plots into {IMAGES_DIR}")

    return results


def run_experiment3_nlp(cfg):
    device = get_device()
    seed = cfg['seed']
    set_seed(seed)
    quick = cfg['quick']
    num_tasks = cfg.get('num_tasks', 5)
    buffer_size = cfg.get('buffer_size', 3)

    print(f"[Exp3] NLP CL (synthetic BoW) | seed={seed} | device={device}")
    tasks, n_classes, backbone_state, feat_dim, controller = make_nlp_tasks(num_tasks=num_tasks, seed=seed,
                                                                            vocab_size=1000, feat_dim=256,
                                                                            n_classes=4, ntr=(400 if quick else 800), nte=200, quick=quick)

    # Orders
    order_storm = controller.greedy_order(list(range(len(tasks))))
    order_random = np.random.default_rng(seed).permutation(len(tasks)).tolist()
    controller_sim = StormController(alpha=1.0, beta=0.0, buffer_size=buffer_size)
    controller_sim.G = controller.G
    order_sim = controller_sim.greedy_order(list(range(len(tasks))))

    print(f"[Exp3] Training STORM order: {order_storm}")
    acc_storm, loss_storm = train_sequence_nlp(tasks, order_storm, backbone_state, feat_dim, n_classes, device=device, quick=quick)
    print(f"[Exp3] Training Random order: {order_random}")
    acc_rand, loss_rand = train_sequence_nlp(tasks, order_random, backbone_state, feat_dim, n_classes, device=device, quick=quick)
    print(f"[Exp3] Training Similarity-only order: {order_sim}")
    acc_sim, loss_sim = train_sequence_nlp(tasks, order_sim, backbone_state, feat_dim, n_classes, device=device, quick=quick)

    acc_mats = {'storm': acc_storm, 'random': acc_rand, 'sim_only': acc_sim}
    loss_curves = {'storm': loss_storm, 'random': loss_rand, 'sim_only': loss_sim}

    results = {}
    for k, A in acc_mats.items():
        m = compute_forgetting_and_stability(A)
        results[k] = m
        print(f"[Exp3][{k}] final_avg_acc={m['final_avg_acc']:.4f} | avg_forgetting={m['avg_forgetting']:.4f} | min_acc={m['min_acc']:.4f} | wc_drop={m['wc_drop']:.4f}")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_accuracy_curves(acc_mats, save_path=os.path.join(IMAGES_DIR, 'exp3_accuracy_nlp_storm_vs_baselines.pdf'))
    plot_losses(loss_curves, save_path=os.path.join(IMAGES_DIR, 'exp3_training_loss_nlp_storm_vs_baselines.pdf'))
    print(f"[Exp3] Saved plots into {IMAGES_DIR}")

    return results


def load_config(path: str) -> dict:
    if path is None or not os.path.isfile(path):
        # Defaults
        return {
            'seed': 0,
            'quick': True,
            'run_exp1': True,
            'run_exp2': True,
            'run_exp3': True,
            'dataset': 'cifar10',
            'num_tasks': 5,
            'buffer_size': 3,
            'angle_pattern': 'sweep',
        }
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='STORM experiments runner')
    parser.add_argument('--config', type=str, default='config/experiment.yaml', help='YAML config path')
    parser.add_argument('--quick', action='store_true', help='Quick mode (small, fast)')
    parser.add_argument('--no-exp1', action='store_true', help='Skip experiment 1 (synthetic)')
    parser.add_argument('--no-exp2', action='store_true', help='Skip experiment 2 (vision)')
    parser.add_argument('--no-exp3', action='store_true', help='Skip experiment 3 (nlp)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg['quick'] = True
    if args.no_exp1:
        cfg['run_exp1'] = False
    if args.no_exp2:
        cfg['run_exp2'] = False
    if args.no_exp3:
        cfg['run_exp3'] = False

    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("========== STORM: Similarity & inTerference–Optimised cuRRiculuM ==========")
    print(f"Device: {get_device()} | Quick mode: {cfg['quick']}")

    if cfg.get('run_exp1', True):
        _ = run_experiment1_synthetic(cfg)
    else:
        print("[Main] Skipping Exp1")

    if cfg.get('run_exp2', True):
        try:
            _ = run_experiment2_vision(cfg)
        except Exception as e:
            print(f"[Main] Exp2 (vision) skipped due to: {e}")
    else:
        print("[Main] Skipping Exp2")

    if cfg.get('run_exp3', True):
        _ = run_experiment3_nlp(cfg)
    else:
        print("[Main] Skipping Exp3")

    print("All plots saved as high-quality PDFs under .research/iteration2/images")


if __name__ == '__main__':
    main()
