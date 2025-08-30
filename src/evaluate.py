import os
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

# Matplotlib config for high-quality PDF plots
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import spearmanr as scipy_spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@torch.no_grad()
def extract_answer(text: str) -> Optional[str]:
    anchor = text.rfind('Answer:')
    if anchor == -1:
        return None
    tail = text[anchor+7:]
    digits = ''.join([ch for ch in tail if ch in '0123456789-'])
    return digits if digits else None


@torch.no_grad()
def evaluate_em(model, dataset, tokenizer, device: str = 'cpu', max_new_tokens: int = 8) -> Tuple[float, List[str], List[str]]:
    model.eval()
    preds, golds = [], []
    for ex in dataset.samples:
        prompt = ex['prompt'] + 'Answer:'
        ids = tokenizer.encode(prompt)
        ids = ids[:dataset.max_len] + [tokenizer.pad_id] * max(0, dataset.max_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, attention_mask=(x != 0).long(), max_new_tokens=max_new_tokens)
        gen_text = tokenizer.decode(out[0].tolist())
        pa = extract_answer(gen_text)
        preds.append(pa if pa is not None else '')
        golds.append(ex['answer'])
    em = float(np.mean([p == g for p, g in zip(preds, golds)]))
    return em, preds, golds


@torch.no_grad()
def measure_inference(model, tokenizer, prompts: List[str], device: str = 'cpu', max_new_tokens: int = 16) -> Dict:
    model.eval()
    if torch.cuda.is_available() and device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and device.startswith('cuda') else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and device.startswith('cuda') else None

    toks = 0
    if t0 is not None and t1 is not None:
        torch.cuda.synchronize(); t0.record()
    else:
        import time
        t_start = time.time()

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            toks += len(ids) + max_new_tokens
            ids = ids[:256] + [tokenizer.pad_id] * max(0, 256 - len(ids))
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            _ = model.generate(x, attention_mask=(x != 0).long(), max_new_tokens=max_new_tokens)

    if t0 is not None and t1 is not None:
        t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        dt = ms / 1000.0
        mem_gb = torch.cuda.max_memory_allocated(device=device) / (2**30)
    else:
        import time
        dt = time.time() - t_start
        mem_gb = 0.0

    return {'throughput_toks_per_s': toks / max(dt, 1e-6), 'peak_gb': mem_gb}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if _HAS_SCIPY:
        rho, _ = scipy_spearmanr(x, y)
        return 0.0 if (rho is None or np.isnan(rho)) else float(rho)
    # simple fallback
    rx = x.argsort().argsort(); ry = y.argsort().argsort()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    num = (rx * ry).sum()
    den = math.sqrt((rx**2).sum() * (ry**2).sum() + 1e-8)
    return float(num / den) if den > 0 else 0.0


def save_barplot_pdf(xlabels: List[str], values: List[float], ylabel: str, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4.2, 3.2))
    sns.barplot(x=xlabels, y=values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def save_scatter_pdf(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4.2, 3.2))
    plt.scatter(x, y, s=12)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
