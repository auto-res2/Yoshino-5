import os
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None

# Support running as a script or a package
try:
    from .train import (
        warmup_train_predictor,
        step_forward,
        BudgetBandit,
        set_residual_enabled,
        hiqua_gating,
    )
    from .preprocess import get_image_dir
except ImportError:  # pragma: no cover - fallback for script execution
    from train import (
        warmup_train_predictor,
        step_forward,
        BudgetBandit,
        set_residual_enabled,
        hiqua_gating,
    )
    from preprocess import get_image_dir


# ------------------------------ Helpers ------------------------------

def peak_vram_gb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if _scipy_spearmanr is not None:
        return float(_scipy_spearmanr(x, y).correlation)
    # Fallback: via rank transform + Pearson
    def rankdata(a):
        temp = a.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(a))
        return ranks.astype(np.float64)
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float(np.dot(rx, ry) / (len(rx) - 1))


# ------------------------------ Plotting (PDF to .research/iteration1/images) ------------------------------

def _save_fig_pdf(name: str):
    img_dir = get_image_dir()
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def plot_training_loss(losses: List[float], fname: str = 'training_loss_hiqua.pdf') -> str:
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=np.arange(len(losses)), y=losses)
    plt.xlabel('Warm-up step')
    plt.ylabel('KL loss (teacher->student)')
    plt.title('HiQuA predictor warm-up loss')
    return _save_fig_pdf(fname)


def plot_tokens_per_second(results: List[Dict], fname: str = 'tokens_per_second_hiqua.pdf') -> str:
    labels = [r['prompt'][:24] + ('...' if len(r['prompt']) > 24 else '') for r in results]
    vals = [r['tok_s'] for r in results]
    plt.figure(figsize=(6, 3.5))
    ax = sns.barplot(x=labels, y=vals, color='#4472C4')
    for p, v in zip(ax.patches, vals):
        ax.annotate(f"{v:.1f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=8)
    plt.ylabel('Tokens/s')
    plt.xticks(rotation=30, ha='right')
    plt.title('Decode throughput (HiQuA)')
    return _save_fig_pdf(fname)


def plot_budget_sweep(budgets: List[float], tps: List[float], ratios: List[float]):
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=budgets, y=tps, marker='o')
    plt.xlabel('Target promotion ratio')
    plt.ylabel('Tokens/s')
    plt.title('Throughput vs budget (HiQuA)')
    p1 = _save_fig_pdf('tokens_per_second_budget_hiqua.pdf')

    plt.figure(figsize=(5, 3))
    sns.lineplot(x=budgets, y=ratios, marker='o')
    plt.xlabel('Target promotion ratio')
    plt.ylabel('Achieved promotion ratio')
    plt.title('Promotion ratio tracking (HiQuA)')
    p2 = _save_fig_pdf('promotion_ratio_budget_hiqua.pdf')
    return p1, p2


def plot_predictor_correlation(spearmans: List[float], auc_likes: List[float]):
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=np.arange(len(spearmans)), y=spearmans, marker='o', label='Spearman')
    sns.lineplot(x=np.arange(len(auc_likes)), y=auc_likes, marker='s', label='AUC-like')
    plt.xlabel('Sample index')
    plt.ylabel('Score')
    plt.title('Predictor vs Oracle importance')
    plt.legend()
    return _save_fig_pdf('predictor_correlation_hiqua.pdf')


def plot_overhead_bar(base_ms: float, promoted_ms: float):
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(x=['base', 'promoted'], y=[base_ms, promoted_ms], palette=['#6AA84F', '#CC0000'])
    for p, v in zip(ax.patches, [base_ms, promoted_ms]):
        ax.annotate(f"{v:.1f} ms", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=8)
    plt.ylabel('Latency (ms)')
    plt.title('Masked residual injection overhead')
    return _save_fig_pdf('overhead_microbenchmark_hiqua.pdf')


# ------------------------------ Decode/eval utilities ------------------------------

@torch.no_grad()
def decode_benchmark(model, predictor, tokenizer, prompts: List[str], max_new_tokens: int = 64, target_ratio: float = 0.15):
    device = next(model.parameters()).device
    bandit = BudgetBandit(target_ratio=target_ratio)
    stats = []
    for p in prompts:
        input_ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        attn = torch.ones_like(input_ids)
        past = None
        out_ids = []
        total_ratio = 0.0
        steps = 0
        if torch.cuda.is_available():
            start = torch.cuda.Event(True); end = torch.cuda.Event(True)
            torch.cuda.synchronize(); start.record()
        else:
            t0 = time.time()
        for _ in range(max_new_tokens):
            outputs, ratio = step_forward(model, predictor, input_ids, attn, past_key_values=past, bandit=bandit)
            logits = outputs.logits
            past = outputs.past_key_values
            nxt = torch.argmax(logits[:, -1, :], dim=-1)
            out_ids.append(int(nxt.item()))
            input_ids = nxt[:, None]
            attn = torch.ones_like(input_ids)
            total_ratio += ratio; steps += 1
        if torch.cuda.is_available():
            end.record(); torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = (time.time() - t0) * 1000.0
        tok_s = steps * 1000.0 / max(1e-6, elapsed_ms)
        text = tokenizer.decode(out_ids)
        stats.append({"prompt": p, "tok_s": tok_s, "avg_ratio": total_ratio/max(1,steps), "gen": text})
    return stats


@torch.no_grad()
def run_once_decode(model, predictor, tokenizer, prompt: str, max_new_tokens: int, target_ratio: float, use_bandit: bool = True) -> Tuple[float, float]:
    device = next(model.parameters()).device
    bandit = BudgetBandit(target_ratio=target_ratio, lr=0.2) if use_bandit else None
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    attn = torch.ones_like(input_ids)
    past = None
    total_ratio = 0.0
    steps = 0
    if torch.cuda.is_available():
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        torch.cuda.synchronize(); start.record()
    else:
        t0 = time.time()
    for _ in range(max_new_tokens):
        outputs, ratio = step_forward(model, predictor, input_ids, attn, past_key_values=past, bandit=bandit)
        past = outputs.past_key_values
        nxt = outputs.logits[:, -1, :].argmax(dim=-1)
        input_ids = nxt[:, None]
        attn = torch.ones_like(input_ids)
        total_ratio += ratio
        steps += 1
    if torch.cuda.is_available():
        end.record(); torch.cuda.synchronize(); elapsed_ms = start.elapsed_time(end)
    else:
        elapsed_ms = (time.time() - t0) * 1000.0
    tok_s = steps * 1000.0 / max(1e-6, elapsed_ms)
    return tok_s, total_ratio / max(1, steps)


# ------------------------------ Oracle analysis for predictor quality ------------------------------

@torch.no_grad()
def oracle_delta_kl(model, predictor, tokenizer, text: str, layer_idx: int = 0) -> Dict[str, List[float]]:
    inp = tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=256).to(next(model.parameters()).device)
    L = len(model.model.layers)
    names = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
    chunks = predictor.chunks

    # Teacher: all promoted
    all_true = {k: torch.ones(1, chunks, dtype=torch.bool, device=inp.input_ids.device) for k in names}
    with hiqua_gating([all_true for _ in range(L)]):
        logits_teacher = model(**inp).logits[:, -1, :]

    # Base: none promoted
    all_false = {k: torch.zeros(1, chunks, dtype=torch.bool, device=inp.input_ids.device) for k in names}
    with hiqua_gating([all_false for _ in range(L)]):
        logits_base = model(**inp).logits[:, -1, :]

    pt = F.log_softmax(logits_teacher, dim=-1)
    pb = F.log_softmax(logits_base, dim=-1)
    kl_none = float(F.kl_div(pb, pt, log_target=True, reduction='batchmean').item())

    delta_scores = {name: [] for name in names}
    for name in names:
        for ch in range(chunks):
            gates = [dict(all_false) for _ in range(L)]
            gates[layer_idx] = {k: torch.zeros(1, chunks, dtype=torch.bool, device=inp.input_ids.device) for k in names}
            gates[layer_idx][name][:, ch] = True
            with hiqua_gating(gates):
                logits_s = model(**inp).logits[:, -1, :]
            ps = F.log_softmax(logits_s, dim=-1)
            kl_s = float(F.kl_div(ps, pt, log_target=True, reduction='batchmean').item())
            delta_scores[name].append(kl_s - kl_none)
    return delta_scores


@torch.no_grad()
def predictor_scores(model, predictor, tokenizer, text: str) -> np.ndarray:
    inp = tokenizer(text, return_tensors='pt').to(next(model.parameters()).device)
    out = model(input_ids=inp.input_ids, attention_mask=inp.attention_mask, output_hidden_states=True)
    scores = predictor(out.hidden_states[-1])[0].detach().cpu().numpy()
    return scores
