import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compute_forgetting_and_stability(acc_matrix: np.ndarray) -> Dict:
    # acc_matrix: steps x tasks
    final_avg = float(acc_matrix[-1].mean()) if acc_matrix.size else 0.0
    F = []
    for j in range(acc_matrix.shape[1]):
        hist = acc_matrix[:, j]
        F.append(float(hist.max() - hist[-1]))
    avg_forgetting = float(np.mean(F)) if F else 0.0
    mean_over_time = acc_matrix.mean(axis=1) if acc_matrix.size else np.array([0.0])
    min_acc = float(np.min(mean_over_time)) if mean_over_time.size else 0.0
    wc_drop = float(np.max(mean_over_time) - np.min(mean_over_time)) if mean_over_time.size else 0.0
    return dict(final_avg_acc=final_avg, avg_forgetting=avg_forgetting, min_acc=min_acc, wc_drop=wc_drop)


def plot_accuracy_curves(acc_mats: Dict[str, np.ndarray], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6.0, 3.8))
    for name, M in acc_mats.items():
        if M.size == 0:
            continue
        plt.plot(np.arange(1, M.shape[0] + 1), M.mean(axis=1), label=name)
    plt.xlabel('Tasks seen')
    plt.ylabel('Average accuracy')
    plt.title('Average accuracy over sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_losses(losses: Dict[str, List[float]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6.0, 3.8))
    for name, ls in losses.items():
        if len(ls) == 0:
            continue
        xs = np.arange(len(ls))
        plt.plot(xs, ls, label=name, linewidth=1.0)
    plt.xlabel('Update step')
    plt.ylabel('Training loss')
    plt.title('Training loss over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def spearman_correlation_with_empirical(M_values: List[float], empirical_values: List[float]) -> Dict:
    if len(M_values) == 0 or len(M_values) != len(empirical_values):
        return {"rho": 0.0, "p": 1.0}
    rho, p = spearmanr(M_values, empirical_values)
    return {"rho": float(rho), "p": float(p)}
