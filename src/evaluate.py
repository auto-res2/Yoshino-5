import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import train_task, compute_embeddings, ReplayBuffer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def evaluate_taskwise(model: torch.nn.Module, device: str, task_loaders: List[DataLoader]) -> List[float]:
    model.eval()
    accs = []
    for ld in task_loaders:
        correct = 0
        total = 0
        for x, y in ld:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        accs.append(100.0 * correct / max(1, total))
    return accs


def compute_accuracy_matrix(model: torch.nn.Module, device: str, order: List[int], task_loaders_tr: List[DataLoader], task_loaders_te: List[DataLoader], epochs_per_task: int = 1, lr: float = 5e-4) -> Tuple[np.ndarray, List[float]]:
    criterion = torch.nn.CrossEntropyLoss()
    rb = ReplayBuffer(anchor_size=32)
    rb_emb = None
    acc_matrix: List[List[float]] = []
    losses: List[float] = []
    for step, tid in enumerate(order):
        train_task(model, device, task_loaders_tr[tid], criterion, epochs=epochs_per_task, lr=lr, log_losses=losses)
        rb.add_anchor(tid, task_loaders_tr[tid])
        with torch.no_grad():
            e, _ = compute_embeddings(model, task_loaders_tr[tid], device)
            rb_emb = e if rb_emb is None else torch.cat([rb_emb, e], 0)[-32:]
        # evaluate over seen tasks
        seen_test_loaders = [task_loaders_te[t] for t in order[: step + 1]]
        accs = evaluate_taskwise(model, device, seen_test_loaders)
        row = [0.0] * len(task_loaders_tr)
        for i, t in enumerate(order[: step + 1]):
            row[t] = accs[i]
        if step > 0:
            prev = acc_matrix[-1]
            for t in range(len(row)):
                if row[t] == 0.0:
                    row[t] = prev[t]
        acc_matrix.append(row)
    return np.array(acc_matrix, dtype=np.float32), losses


def compute_forgetting(acc_matrix: np.ndarray) -> Tuple[float, List[float]]:
    S, T = acc_matrix.shape
    forgetting_vals = []
    for t in range(T):
        hist = [acc_matrix[s, t] for s in range(S) if acc_matrix[s, t] > 0]
        if len(hist) >= 2:
            forgetting_vals.append(max(hist[:-1]) - hist[-1])
        elif len(hist) == 1:
            forgetting_vals.append(0.0)
    avg_f = float(np.mean(forgetting_vals)) if len(forgetting_vals) else 0.0
    return avg_f, forgetting_vals


def build_confusion_matrix(model: torch.nn.Module, device: str, all_loaders: List[DataLoader], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for ld in all_loaders:
            for x, y in ld:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                for t, p in zip(y.view(-1), preds.view(-1)):
                    cm[int(t.item()), int(p.item())] += 1
    return cm


# ---------------
# Plotting (PDF)
# ---------------

def plot_training_loss(losses: List[float], output_dir: str, fname: str = "training_loss_pts.pdf"):
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='train loss', color='tab:blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-entropy loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(acc_mats: Dict[str, np.ndarray], output_dir: str, fname: str = "accuracy_pts_vs_baselines.pdf"):
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 4))
    for name, A in acc_mats.items():
        avg_seen = [np.mean([a for a in A[s] if a > 0]) for s in range(A.shape[0])]
        plt.plot(avg_seen, label=name)
    plt.xlabel('Task step')
    plt.ylabel('Average accuracy over seen tasks (%)')
    plt.title('ACC vs Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close()


def plot_forgetting(forgetting_vals: List[float], output_dir: str, fname: str = "forgetting_pts.pdf"):
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 4))
    plt.plot(forgetting_vals, marker='o', linestyle='-', color='tab:red')
    plt.xlabel('Task index')
    plt.ylabel('Forgetting per task (pp)')
    plt.title('Average Forgetting per Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close()


def plot_confusion(cm: np.ndarray, output_dir: str, fname: str = "confusion_matrix_pts.pdf"):
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 5))
    cm_norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
    sns.heatmap(cm_norm, cmap='viridis')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('Confusion Matrix (normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close()


def plot_overhead_bars(overheads: Dict[str, float], output_dir: str, fname: str = "overhead_pts.pdf"):
    ensure_dir(output_dir)
    plt.figure(figsize=(5, 4))
    names = list(overheads.keys())
    vals = [overheads[k] for k in names]
    colors = ['tab:gray' if 'baseline' in n else 'tab:green' for n in names]
    plt.bar(names, vals, color=colors)
    plt.ylabel('Overhead (%)')
    plt.title('Surrogate Overhead vs Training')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close()


def print_summary(results: Dict[str, Dict]):
    print(json.dumps(results, indent=2))
