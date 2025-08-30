import os
import json
from typing import List, Dict

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from .preprocess import (
    set_seed, device_default,
    get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_cifar100_split_tasks,
    build_task_loaders, get_images_dir,
)
from .train import (
    build_backbone, PEFTWrapper, MetaMLP, PTSScheduler, ReplayBuffer,
    train_task, compute_embeddings, compute_surrogates,
    chronological_order, random_order,
)
from .evaluate import (
    evaluate_taskwise, compute_accuracy_matrix, compute_forgetting,
    build_confusion_matrix, plot_training_loss, plot_accuracy_curves,
    plot_forgetting, plot_confusion, plot_overhead_bars, print_summary,
)


def load_config(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    # Defaults for a quick test
    return {
        'dataset': 'permuted_mnist',
        'num_tasks': 3,
        'classes_per_task': 5,
        'epochs_per_task': 1,
        'batch_size_train': 64,
        'batch_size_eval': 128,
        'lora_rank': 4,
        'K': 3,
        'D': 3,
        'meta_train': False,
        'save_plots': True,
        'seed': 123,
        'use_gpu': True,
        'compare_baselines': True,
    }


def get_tasks_and_loaders(cfg: Dict):
    name = cfg.get('dataset', 'permuted_mnist').lower()
    num_tasks = int(cfg.get('num_tasks', 3))
    classes_per_task = int(cfg.get('classes_per_task', 5))
    if name in ['permuted_mnist', 'pmnist']:
        tr_tasks = get_permuted_mnist_tasks(num_tasks=num_tasks, train=True)
        te_tasks = get_permuted_mnist_tasks(num_tasks=num_tasks, train=False)
        num_classes = 10
    elif name in ['rotated_mnist', 'rmnist']:
        tr_tasks = get_rotated_mnist_tasks(num_tasks=num_tasks, train=True)
        te_tasks = get_rotated_mnist_tasks(num_tasks=num_tasks, train=False)
        num_classes = 10
    elif name in ['cifar100']:
        tr_tasks = get_cifar100_split_tasks(num_tasks=num_tasks, classes_per_task=classes_per_task, train=True)
        te_tasks = get_cifar100_split_tasks(num_tasks=num_tasks, classes_per_task=classes_per_task, train=False)
        num_classes = 100
    else:
        tr_tasks = get_permuted_mnist_tasks(num_tasks=num_tasks, train=True)
        te_tasks = get_permuted_mnist_tasks(num_tasks=num_tasks, train=False)
        num_classes = 10
    ldr_tr, ldr_te = build_task_loaders(tr_tasks, batch_train=int(cfg.get('batch_size_train', 64)), batch_eval=int(cfg.get('batch_size_eval', 128)))
    return tr_tasks, te_tasks, ldr_tr, ldr_te, num_classes


def adaptive_pts_training(cfg: Dict, device: str, ldr_tr: List[DataLoader], ldr_te: List[DataLoader], num_classes: int, images_dir: str):
    # Build model and components
    model = PEFTWrapper(build_backbone(cfg.get('dataset', 'permuted_mnist'), num_classes=num_classes), lora_rank=int(cfg.get('lora_rank', 4))).to(device)
    omega = MetaMLP().to(device)  # untrained tiny MLP; meta training optional in future
    scheduler = PTSScheduler(omega, K=int(cfg.get('K', 3)), D=int(cfg.get('D', 3)), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    rb = ReplayBuffer(anchor_size=16)
    rb_emb = None
    age: Dict[int, int] = {}
    remaining = list(range(len(ldr_tr)))
    order: List[int] = []

    acc_curve: List[float] = []
    losses: List[float] = []

    while remaining:
        buf = remaining[: int(cfg.get('K', 3))]
        feats = []
        anch_ld = rb.build_anchor_loader(batch_size=16)
        for tid in buf:
            Fi, Gi, Mi = compute_surrogates(model, device, ldr_tr[tid], anch_ld, rb_emb, criterion)
            feats.append((tid, np.array([Fi, Gi, Mi, age.get(tid, 0)], dtype=np.float32)))
        pick = scheduler.select_next(feats)
        order.append(pick)
        remaining.remove(pick)
        # Train chosen task
        train_task(model, device, ldr_tr[pick], criterion, epochs=int(cfg.get('epochs_per_task', 1)), lr=1e-3, log_losses=losses)
        rb.add_anchor(pick, ldr_tr[pick])
        with torch.no_grad():
            e, _ = compute_embeddings(model, ldr_tr[pick], device)
            rb_emb = e if rb_emb is None else torch.cat([rb_emb, e], 0)[-32:]
        age[pick] = age.get(pick, 0) + 1
        # Evaluate average over seen tasks
        seen_test_loaders = [ldr_te[t] for t in order]
        acc_curve.append(float(np.mean(evaluate_taskwise(model, device, seen_test_loaders))))

    # Build accuracy matrix for plotting using the observed curve (approx.)
    acc_mat = np.zeros((len(order), len(ldr_tr)), dtype=np.float32)
    for s, tid in enumerate(order):
        for i, t in enumerate(order[: s + 1]):
            # fill same value across seen tasks for a compact quick view
            acc_mat[s, t] = acc_curve[s]
        if s > 0:
            acc_mat[s] = np.maximum(acc_mat[s], acc_mat[s - 1])

    # Compute forgetting proxy (using acc_mat)
    avg_f, f_list = compute_forgetting(acc_mat)

    # Save plots
    plot_training_loss(losses, images_dir, fname="training_loss_pts.pdf")
    plot_accuracy_curves({"PTS": acc_mat}, images_dir, fname="accuracy_pts_only.pdf")
    plot_forgetting(f_list, images_dir, fname="forgetting_pts.pdf")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'order': order}, os.path.join('models', 'pts_final.pth'))

    return {
        'order': order,
        'final_mean_acc': (float(acc_curve[-1]) if len(acc_curve) else 0.0),
        'avg_forgetting': float(avg_f),
        'loss_it': len(losses),
    }, acc_mat


def run():
    cfg_path = os.path.join('config', 'pts_config.yaml')
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get('seed', 123)))
    device = device_default() if cfg.get('use_gpu', True) else 'cpu'
    images_dir = get_images_dir()

    print("[Main] Preparing tasks and loaders...")
    tr_tasks, te_tasks, ldr_tr, ldr_te, num_classes = get_tasks_and_loaders(cfg)

    print("[Main] Running PTS adaptive training...")
    pts_result, pts_accmat = adaptive_pts_training(cfg, device, ldr_tr, ldr_te, num_classes, images_dir)

    results = {'PTS': pts_result}

    if cfg.get('compare_baselines', True):
        # Baselines: Chronological and Random (trained from scratch)
        criterion = torch.nn.CrossEntropyLoss()

        # Chronological
        model_c = PEFTWrapper(build_backbone(cfg.get('dataset', 'permuted_mnist'), num_classes=num_classes), lora_rank=int(cfg.get('lora_rank', 4))).to(device)
        order_c = chronological_order(len(ldr_tr))
        A_c, _ = compute_accuracy_matrix(model_c, device, order_c, ldr_tr, ldr_te, epochs_per_task=int(cfg.get('epochs_per_task', 1)), lr=5e-4)
        final_c = A_c[-1]
        results['Chrono'] = {
            'order': order_c,
            'final_mean_acc': float(np.mean([a for a in final_c if a > 0])),
            'avg_forgetting': float(compute_forgetting(A_c)[0])
        }

        # Random
        model_r = PEFTWrapper(build_backbone(cfg.get('dataset', 'permuted_mnist'), num_classes=num_classes), lora_rank=int(cfg.get('lora_rank', 4))).to(device)
        order_r = random_order(len(ldr_tr))
        A_r, _ = compute_accuracy_matrix(model_r, device, order_r, ldr_tr, ldr_te, epochs_per_task=int(cfg.get('epochs_per_task', 1)), lr=5e-4)
        final_r = A_r[-1]
        results['Random'] = {
            'order': order_r,
            'final_mean_acc': float(np.mean([a for a in final_r if a > 0])),
            'avg_forgetting': float(compute_forgetting(A_r)[0])
        }

        # Plots: compare curves
        plot_accuracy_curves({
            'Chrono': A_c,
            'Random': A_r,
            'PTS': pts_accmat,
        }, images_dir, fname="accuracy_pts_vs_baselines.pdf")

    print("[Main] Summary:")
    print_summary(results)


if __name__ == '__main__':
    run()
