import os
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix
except Exception:
    confusion_matrix = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_loss(loss_history: Dict[int, List[float]], out_dir: str,
                       title: str = 'Training Loss per Task', filename: str = 'training_loss_dyclos.pdf'):
    ensure_dir(out_dir)
    plt.figure(figsize=(6, 4))
    for task_id, losses in loss_history.items():
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Task {task_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(acc_hist: Dict[int, List[float]], out_dir: str,
                          title: str = 'Per-task Accuracy Over Time', filename: str = 'accuracy_dyclos.pdf'):
    ensure_dir(out_dir)
    plt.figure(figsize=(6, 4))
    for t, accs in acc_hist.items():
        plt.plot(range(1, len(accs) + 1), accs, marker='s', label=f'Task {t}')
    plt.xlabel('After Task #')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_schedule_costs(scheduler, out_dir: str, title: str = 'Schedule Edge Costs', filename: str = 'schedule_costs_dyclos.pdf'):
    ensure_dir(out_dir)
    path = scheduler.trained_prefix
    costs = []
    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        c = scheduler.edge_cost(u, v)
        costs.append(c)
        edges.append(f'{u}->{v}')
    if costs:
        plt.figure(figsize=(max(6, len(costs) * 0.7), 3))
        plt.bar(range(len(costs)), costs)
        plt.xticks(range(len(costs)), edges, rotation=45)
        plt.ylabel('Edge Cost')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(model, dataloader, n_classes: int, device: Optional[str], out_dir: str,
                          title: str = 'Confusion Matrix', filename: str = 'confusion_matrix_dyclos.pdf'):
    if confusion_matrix is None:
        print('[WARN] sklearn not available; skipping confusion matrix plot')
        return
    ensure_dir(out_dir)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch['input_ids'], batch['labels']
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    import numpy as np
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cmn = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cmn, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def summarize_results(results: Dict[str, Any]) -> str:
    return (
        f"Arrival order: {results['arrival_order']}\n"
        f"DyCLOS order: {results['dyclos_order']}\n"
        f"DyCLOS  -> Omega-acc={results['dyclos']['Omega-acc']:.4f}, "
        f"Omega-forg={results['dyclos']['Omega-forg']:.4f}, "
        f"Probe+Schedule Time (s)={results['dyclos']['probe+schedule_time_s']:.2f}\n"
        f"Baseline-> Omega-acc={results['baseline']['Omega-acc']:.4f}, "
        f"Omega-forg={results['baseline']['Omega-forg']:.4f}\n"
    )
