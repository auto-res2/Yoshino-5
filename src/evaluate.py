import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from train import ABLoRAEff

# High-quality PDF settings
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300


# -----------------------------
# Metrics
# -----------------------------

def temporal_grounding_miou(pred_t: int, gt_t: int, T: int, window: int = 2) -> float:
    ps, pe = max(0, pred_t - window), min(T - 1, pred_t + window)
    gs, ge = max(0, gt_t - window), min(T - 1, gt_t + window)
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs) + 1e-9
    return float(inter / union)


def temporal_consistency_cosine(Ftd: torch.Tensor) -> float:
    T, D = Ftd.shape
    if T < 2:
        return 1.0
    vals = []
    for t in range(T - 1):
        a = Ftd[t] / (Ftd[t].norm() + 1e-9)
        b = Ftd[t + 1] / (Ftd[t + 1].norm() + 1e-9)
        vals.append(float((a * b).sum().item()))
    return float(np.mean(vals))


def motion_complexity_entropy(frames: np.ndarray) -> float:
    mags = []
    for t in range(len(frames) - 1):
        diff = frames[t + 1].astype(np.int16) - frames[t].astype(np.int16)
        mag = np.abs(diff).mean(axis=2)
        mags.append(mag)
    if not mags:
        return 0.0
    hist, _ = np.histogram(np.concatenate([m.flatten() for m in mags]), bins=32, range=(0, 255), density=True)
    hist = hist + 1e-12
    return float(-(hist * np.log(hist)).sum())


def confusion_matrix_numpy(y_true: List[int], y_pred: List[int], n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# -----------------------------
# Evaluation routine
# -----------------------------

@torch.no_grad()
def evaluate_method(attn_block, basis_enc, clip_model, loader: DataLoader, device: str,
                    qa_head, grd_head, K_max: int, mdl_penalty: float,
                    em_func, use_mdl_gamma: bool = True) -> Dict[str, float]:
    attn_block.eval(); qa_head.eval(); grd_head.eval()
    n_classes = 4
    y_true, y_pred = [], []
    miou_list, cons_list, K_list, flow_ent_list = [], [], [], []
    for batch in loader:
        frames = batch['frames'][0].numpy()
        qa_cls = int(batch['qa_cls'][0].item()) if isinstance(batch['qa_cls'], torch.Tensor) else int(batch['qa_cls'][0])
        first_t = int(batch['first_change_t'][0].item()) if isinstance(batch['first_change_t'], torch.Tensor) else int(batch['first_change_t'][0])
        T = int(batch['length'][0].item()) if isinstance(batch['length'], torch.Tensor) else int(batch['length'][0])
        imgs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).float().to(device) / 255.0
        Ftd = clip_model(imgs)  # [T,D]
        gamma = None
        if isinstance(attn_block.adapter, ABLoRAEff):
            if use_mdl_gamma:
                gamma, K_sel, _ = em_func(Ftd, K_max, mdl_penalty)
            else:
                K_sel = min(8, K_max)
                gamma = torch.ones(T, K_sel, device=device) / K_sel
            K_list.append(K_sel)
        y = attn_block(Ftd.unsqueeze(0), gamma)
        qa_logits = qa_head(y)
        grd_scores = grd_head(y)
        pred_cls = int(torch.argmax(qa_logits, dim=-1).item())
        pred_t = int(torch.argmax(grd_scores, dim=-1).item())
        y_true.append(qa_cls); y_pred.append(pred_cls)
        miou_list.append(temporal_grounding_miou(pred_t, first_t, T))
        cons_list.append(temporal_consistency_cosine(y.squeeze(0)))
        flow_ent_list.append(motion_complexity_entropy(frames))
    acc = float(np.mean([yt == yp for yt, yp in zip(y_true, y_pred)]))
    miou = float(np.mean(miou_list))
    cons = float(np.mean(cons_list))
    K_mean = float(np.mean(K_list)) if K_list else 1.0
    flow_ent = float(np.mean(flow_ent_list))
    cm = confusion_matrix_numpy(y_true, y_pred, n_classes)
    return {
        'accuracy': acc,
        'miou': miou,
        'consistency': cons,
        'K_mean': K_mean,
        'flow_entropy': flow_ent,
        'confusion_matrix': cm.tolist()
    }


# -----------------------------
# Plotting helpers (PDF only, saved to out_dir)
# -----------------------------


def plot_training_losses(losses_a: List[float], losses_b: List[float], name_a: str, name_b: str, out_dir: str):
    plt.figure(figsize=(6, 4))
    plt.plot(losses_a, label=name_a)
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/training_loss_ab_lora_pair1.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(losses_b, label=name_b, color='orange')
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss (Baselines)')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/training_loss_baselines_pair2.pdf", bbox_inches='tight')
    plt.close()


def plot_accuracy_bar(results: Dict[str, Dict[str, float]], out_dir: str):
    methods = list(results.keys())
    accs = [results[m]['accuracy'] for m in methods]
    plt.figure(figsize=(5, 4))
    sns.barplot(x=methods, y=accs, palette='Set2')
    plt.ylabel('Accuracy'); plt.title('QA Accuracy across methods')
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(f"{out_dir}/accuracy_multimethods.pdf", bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], fname: str):
    arr = np.array(cm)
    plt.figure(figsize=(4, 4))
    sns.heatmap(arr, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_k_vs_complexity(complexities: List[int], K_means: List[float], out_dir: str):
    plt.figure(figsize=(5, 4))
    plt.plot(complexities, K_means, marker='o')
    plt.xlabel('Segments (complexity)'); plt.ylabel('Selected K (mean)')
    plt.title('AB-LoRA K selection vs complexity')
    plt.tight_layout(); plt.savefig(f"{out_dir}/k_selected_complexity.pdf", bbox_inches='tight')
    plt.close()


def plot_temporal_consistency(results: Dict[str, Dict[str, float]], out_dir: str):
    methods = list(results.keys())
    cons = [results[m]['consistency'] for m in methods]
    plt.figure(figsize=(5, 4))
    sns.barplot(x=methods, y=cons, palette='Set3')
    plt.ylabel('Feature-warped cosine (proxy)'); plt.title('Temporal Consistency')
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(f"{out_dir}/temporal_consistency_multimethods.pdf", bbox_inches='tight')
    plt.close()


def plot_efficiency(params_dict: Dict[str, float], mem_dict: Dict[str, float], out_dir: str):
    methods = list(params_dict.keys())
    params = [params_dict[m] for m in methods]
    mems = [mem_dict[m] for m in methods]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.barplot(x=methods, y=params, ax=ax[0], palette='pastel')
    ax[0].set_title('Trainable Params (M)'); ax[0].tick_params(axis='x', rotation=20)
    sns.barplot(x=methods, y=mems, ax=ax[1], palette='pastel')
    ax[1].set_title('Peak GPU Mem (MB)'); ax[1].tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.savefig(f"{out_dir}/efficiency_multimethods.pdf", bbox_inches='tight')
    plt.close()
