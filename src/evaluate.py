# -*- coding: utf-8 -*-
"""
Evaluation and visualization utilities for FedU-Align synthetic experiments.
Saves all figures as high-quality PDF in the specified image directory.
"""
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import SyntheticMultimodalDataset, SyntheticMMItem
from .train import FederatedClient


# High-quality PDF settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pdf(fig, fname: str, image_dir: str):
    ensure_dir(image_dir)
    path = os.path.join(image_dir, fname)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, y in zip(preds, labels):
        cm[y, p] += 1
    return cm


def trapezoidal_area(x: np.ndarray, y: np.ndarray) -> float:
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    return float(np.trapz(y_sorted, x_sorted))


def evaluate_global(model, clients_cfg, batch_size: int, beta_fusion: float, missing_rates: List[float], device: torch.device) -> Dict:
    all_items = []
    for cfg in clients_cfg:
        all_items.extend(cfg.val_dataset.items)
    results = {'missing_rate': [], 'acc': [], 'ece': []}
    preds_all = {}
    labels_all = {}
    for mr in missing_rates:
        val_ds = SyntheticMultimodalDataset(all_items, owned_modalities=(True, True, True), missing_rate=mr, seed=1234)
        eval_client = FederatedClient(type('Dummy', (), {'client_id': -1, 'train_dataset': val_ds, 'val_dataset': val_ds, 'owns_modalities': (True, True, True)})(), model, device)  # lightweight client wrapper
        out = eval_client.evaluate(val_ds, batch_size=batch_size, beta_fusion=beta_fusion)
        results['missing_rate'].append(mr)
        results['acc'].append(out['acc'])
        results['ece'].append(out['ece'])
        preds_all[mr] = out['preds']
        labels_all[mr] = out['labels']
    return {**results, 'preds': preds_all, 'labels': labels_all}


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_training_loss(losses_trace: List[float], image_dir: str, tag: str = 'fedu_align'):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(losses_trace, label='train_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    save_pdf(fig, f'training_loss_{tag}.pdf', image_dir)


def plot_acc_missingness(missing: List[float], acc: List[float], image_dir: str, tag: str = 'fedu_align'):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(missing, acc, marker='o')
    plt.xlabel('% Modalities Missing')
    plt.ylabel('Accuracy')
    plt.title('Accuracy under Missing Modalities')
    plt.tight_layout()
    save_pdf(fig, f'accuracy_{tag}.pdf', image_dir)


def plot_ece_missingness(missing: List[float], ece: List[float], image_dir: str, tag: str = 'fedu_align'):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(missing, ece, marker='s', color='tab:red')
    plt.xlabel('% Modalities Missing')
    plt.ylabel('ECE')
    plt.title('Calibration vs Missingness')
    plt.tight_layout()
    save_pdf(fig, f'calibration_ece_{tag}.pdf', image_dir)


def plot_comm_per_round(comm_mb: List[float], image_dir: str):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(comm_mb) + 1), comm_mb, marker='^')
    plt.xlabel('Round')
    plt.ylabel('MB per round (uplink approx)')
    plt.title('Communication Cost per Round')
    plt.tight_layout()
    save_pdf(fig, 'communication_cost.pdf', image_dir)


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int, image_dir: str, tag: str = 'fedu_align'):
    cm = confusion_matrix(preds, labels, num_classes)
    fig = plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (0% missing)')
    plt.tight_layout()
    save_pdf(fig, f'confusion_matrix_{tag}.pdf', image_dir)


def plot_aumc_bar(aumc: float, image_dir: str, tag: str = 'fedu_align'):
    fig = plt.figure(figsize=(4, 3))
    plt.bar(['FedU-Align'], [aumc])
    plt.ylabel('AUMC')
    plt.title('Robustness AUMC')
    plt.tight_layout()
    save_pdf(fig, f'aumc_{tag}.pdf', image_dir)


# -----------------------------
# Cross-modal retrieval (Experiment 2)
# -----------------------------

def build_anchor_pooled_features(model, items: List[SyntheticMMItem], modality: str, device: torch.device) -> torch.Tensor:
    model.eval()
    feats = []
    with torch.no_grad():
        for it in items:
            tok = torch.tensor(getattr(it, f'{modality}_tokens'), dtype=torch.float32).unsqueeze(0).to(device)
            z = model.encoders[modality](tok)
            z = model.reft(z)
            dist2 = torch.cdist(z.squeeze(0), model.anchors) ** 2
            P = torch.softmax(-dist2, dim=-1)
            pooled = (P @ model.anchors).mean(dim=0)
            pooled = F.normalize(pooled, dim=0)
            feats.append(pooled.cpu())
    return torch.stack(feats, dim=0)


def recall_at_k(sim: torch.Tensor, k: int, labels_q: np.ndarray, labels_g: np.ndarray) -> float:
    topk = torch.topk(sim, k=k, dim=-1).indices.cpu().numpy()
    hits = 0
    for i in range(sim.size(0)):
        if labels_q[i] in labels_g[topk[i]]:
            hits += 1
    return hits / sim.size(0)


def cross_modal_retrieval(model, clients_cfg, device: torch.device, image_dir: str) -> Dict:
    gallery_items = []
    for cfg in clients_cfg:
        gallery_items.extend(cfg.val_dataset.items)
    mid = max(1, len(gallery_items) // 2)
    query_items = gallery_items[:mid]
    gallery_items = gallery_items[mid:]

    v_q = build_anchor_pooled_features(model, query_items, 'vision', device)
    t_q = build_anchor_pooled_features(model, query_items, 'text', device)
    a_q = build_anchor_pooled_features(model, query_items, 'audio', device)

    v_g = build_anchor_pooled_features(model, gallery_items, 'vision', device)
    t_g = build_anchor_pooled_features(model, gallery_items, 'text', device)
    a_g = build_anchor_pooled_features(model, gallery_items, 'audio', device)

    y_q = np.array([it.label for it in query_items])
    y_g = np.array([it.label for it in gallery_items])

    def cos_sim(x, y):
        x = F.normalize(torch.tensor(x), dim=-1)
        y = F.normalize(torch.tensor(y), dim=-1)
        return x @ y.t()

    sim_vt = cos_sim(v_q, t_g)
    sim_tv = cos_sim(t_q, v_g)
    sim_at = cos_sim(a_q, t_g)
    sim_ta = cos_sim(t_q, a_g)

    r1_vt = recall_at_k(sim_vt, k=1, labels_q=y_q, labels_g=y_g)
    r10_vt = recall_at_k(sim_vt, k=10, labels_q=y_q, labels_g=y_g)
    r1_tv = recall_at_k(sim_tv, k=1, labels_q=y_q, labels_g=y_g)
    r10_tv = recall_at_k(sim_tv, k=10, labels_q=y_q, labels_g=y_g)

    # Plot
    fig = plt.figure(figsize=(5, 3))
    names = ['I→T R@1', 'I→T R@10', 'T→I R@1', 'T→I R@10']
    vals = [r1_vt, r10_vt, r1_tv, r10_tv]
    sns.barplot(x=names, y=vals, palette='viridis')
    plt.ylim(0, 1)
    plt.ylabel('Recall')
    plt.title('Cross-modal Retrieval (FedU-Align)')
    plt.tight_layout()
    save_pdf(fig, 'retrieval_recall_fedu_align.pdf', image_dir)

    return {
        'r1_vt': r1_vt, 'r10_vt': r10_vt,
        'r1_tv': r1_tv, 'r10_tv': r10_tv,
    }


# -----------------------------
# Reliability plots (Experiment 3)
# -----------------------------

def plot_fdr_over_rounds(fdr_over_rounds: List[float], q: float, image_dir: str):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(fdr_over_rounds) + 1), fdr_over_rounds, marker='o')
    plt.axhline(y=q, color='r', linestyle='--', label='target q')
    plt.xlabel('Round')
    plt.ylabel('Empirical FDR')
    plt.title(f'FDR vs Rounds (q={q})')
    plt.legend()
    plt.tight_layout()
    save_pdf(fig, f'fdr_vs_round_q{str(q).replace(".", "_")}.pdf', image_dir)


def plot_comm_vs_rounds(comm_over_rounds: Dict[int, List[float]], q: float, image_dir: str):
    fig = plt.figure(figsize=(5, 3))
    for rank, curve in comm_over_rounds.items():
        plt.plot(range(1, len(curve) + 1), curve, marker='.', label=f'rank={rank}')
    plt.xlabel('Round')
    plt.ylabel('MB per round (uplink approx)')
    plt.title(f'Communication vs Rounds (q={q})')
    plt.legend()
    plt.tight_layout()
    save_pdf(fig, f'communication_vs_round_q{str(q).replace(".", "_")}.pdf', image_dir)


def plot_comm_vs_rank_summary(ranks: List[int], comm_means: List[float], image_dir: str):
    fig = plt.figure(figsize=(5, 3))
    plt.plot(ranks, comm_means, marker='o')
    plt.xlabel('SVD rank')
    plt.ylabel('MB per round (avg)')
    plt.title('Communication vs Compression Rank')
    plt.tight_layout()
    save_pdf(fig, 'communication_vs_rank.pdf', image_dir)
