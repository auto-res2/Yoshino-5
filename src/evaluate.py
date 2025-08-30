#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities and experiment runners for MuViC synthetic framework.
Implements SEDP, SEL, GRO, Bayesian fusion (logistic MAP), baseline MIN-K%.
Produces publication-quality PDFs saved under .research/iteration1/images.
"""
from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import chi2, ttest_rel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

try:
    from .preprocess import (
        QAItem,
        TinyCharTokenizer,
        canonical_text,
        generate_dataset,
        paraphrase_question,  # not directly used but exported for completeness
    )
except ImportError:  # Fallback for script execution without package context
    from preprocess import (  # type: ignore
        QAItem,
        TinyCharTokenizer,
        canonical_text,
        generate_dataset,
        paraphrase_question,
    )

try:
    from .train import TinyLM, TrainingRun, build_training_run, eval_paraphrase_accuracy
except ImportError:  # Fallback for script execution without package context
    from train import TinyLM, TrainingRun, build_training_run, eval_paraphrase_accuracy  # type: ignore


# ------------------------------
# Metrics and helpers
# ------------------------------

def set_seed(seed: int = 123):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rouge_l_f1(ref: str, hyp: str) -> float:
    # Simple LCS-based ROUGE-L F1
    def lcs(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    ra = list(ref)
    ha = list(hyp)
    L = lcs(ra, ha)
    if L == 0:
        return 0.0
    prec = L / max(1, len(ha))
    rec = L / max(1, len(ra))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        idx = (probs >= lo) & (probs < hi)
        if idx.sum() == 0:
            continue
        acc = labels[idx].mean()
        conf = probs[idx].mean()
        ece_val += (idx.mean()) * abs(acc - conf)
    return float(ece_val)


# ------------------------------
# SEDP: Self-Ensemble Difficulty Projection
# ------------------------------

def fit_sedp_mapping(A_mid_clean: np.ndarray, A_fin_clean: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(A_mid_clean, A_fin_clean)
    return ir


def sedp_stat(
    A_mid_obs: float, A_fin_obs: float, ir: IsotonicRegression, null_deltas: np.ndarray
) -> Tuple[float, float, float]:
    pred = ir.predict([A_mid_obs])[0]
    delta = A_fin_obs - pred
    p = (np.sum(null_deltas >= delta) + 1) / (len(null_deltas) + 1)
    z = (delta - null_deltas.mean()) / (null_deltas.std() + 1e-8)
    return float(delta), float(p), float(z)


# ------------------------------
# SEL: Sharded Exchangeability Likelihood
# ------------------------------

def make_shards(n_items: int, shard_size: int = 20, seed: int = 123) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    shards = [idx[i : i + shard_size] for i in range(0, n_items, shard_size)]
    return shards


def model_logp_for_texts(
    model: TinyLM, tokenizer: TinyCharTokenizer, texts: List[str], qids: List[str]
) -> np.ndarray:
    vals = []
    for t, qid in zip(texts, qids):
        vals.append(model.avg_logprob(tokenizer, t, qid))
    return np.array(vals, dtype=np.float64)


def sel_shard_pvalue(
    model: TinyLM, tokenizer: TinyCharTokenizer, data: List[QAItem], shard: np.ndarray, R: int = 5
) -> Tuple[float, np.ndarray]:
    # true concatenations and logps
    items = [data[i] for i in shard]
    texts_true = [canonical_text(it) for it in items]
    qids_true = [it.qid for it in items]
    logp_true = model_logp_for_texts(model, tokenizer, texts_true, qids_true)

    answers = [it.a for it in items]
    diffs = []
    rng = np.random.default_rng(0)
    for _ in range(R):
        perm = rng.permutation(len(answers))
        shuf_texts, shuf_qids = [], []
        for i, it in enumerate(items):
            a_shuf = answers[perm[i]]
            shuf_texts.append(
                f"You are a helpful assistant.\nQuestion: {it.q}\nReasoning: {it.r}\nAnswer: {a_shuf}"
            )
            shuf_qids.append(it.qid)
        logp_shuf = model_logp_for_texts(model, tokenizer, shuf_texts, shuf_qids)
        diffs.append(logp_true - logp_shuf)
    diffs = np.stack(diffs, axis=0).mean(axis=0)
    # paired t-test: greater
    t_stat, p = ttest_rel(logp_true, logp_true - diffs, alternative="greater")
    return float(p), logp_true


def sel_dataset(
    model: TinyLM,
    tokenizer: TinyCharTokenizer,
    data: List[QAItem],
    shard_size: int = 20,
    R: int = 5,
    seed: int = 123,
) -> Tuple[float, Dict[int, float]]:
    shards = make_shards(len(data), shard_size=shard_size, seed=seed)
    pvals = []
    shard_scores: Dict[int, float] = {}
    for s_idx, shard in enumerate(shards):
        p, _ = sel_shard_pvalue(model, tokenizer, data, shard, R=R)
        pvals.append(p)
        shard_scores[s_idx] = p
    # Fisher aggregation
    stat = -2 * np.sum(np.log(np.clip(np.array(pvals), 1e-300, 1.0)))
    df = 2 * len(pvals)
    p_dataset = 1 - chi2.cdf(stat, df)
    return float(p_dataset), shard_scores


# ------------------------------
# GRO: Guided Regeneration Overlap
# ------------------------------

def choose_cut(token_ids: List[int], min_tok: int = 15, max_frac: float = 0.6, rng: Optional[np.random.Generator] = None) -> int:
    rng = rng or np.random.default_rng(0)
    n = len(token_ids)
    lo = min(min_tok, n - 1)
    hi = max(int(n * max_frac), lo + 1)
    hi = min(hi, n - 1)
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi))


def gro_item_score(
    model: TinyLM,
    tokenizer: TinyCharTokenizer,
    item: QAItem,
    k_tail: float = 0.2,
    max_new_tokens: int = 128,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, int]:
    full = canonical_text(item)
    ids = tokenizer.encode(full)
    cut = choose_cut(ids, rng=rng)
    prefix = tokenizer.decode(ids[:cut])
    # Generate (memory will complete suffix exactly if contaminated)
    gen_suffix = model.generate(tokenizer, prefix, max_new_tokens=max_new_tokens, qid=item.qid)
    gold_suffix = tokenizer.decode(ids[cut:])
    r = rouge_l_f1(gold_suffix, gen_suffix)
    # token-level logprobs for generated continuation, clipped to gold length for comparability
    gen_full = prefix + gen_suffix[: len(gold_suffix)]
    ids_full = torch.tensor(tokenizer.encode(gen_full), dtype=torch.long, device=model.device)
    if ids_full.numel() <= 1:
        return 0.0, 0.0, cut
    x = ids_full[:-1].unsqueeze(0)
    y = ids_full[1:].unsqueeze(0)
    logits = model(x)
    logprobs = F.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().numpy()
    if tok_lp.size == 0:
        return r, 0.0, cut
    k = max(1, int(len(tok_lp) * k_tail))
    min_k = float(np.mean(np.sort(tok_lp)[:k]))
    return r, min_k, cut


# ------------------------------
# Fusion: Logistic regression (MAP)
# ------------------------------
import torch.nn as nn


class TorchLogistic(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X @ self.w + self.b)


def fit_logistic_MAP(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-2, steps: int = 1000, lr: float = 5e-2, verbose: bool = False
) -> Tuple[np.ndarray, float]:
    device = torch.device("cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    model = TorchLogistic(X.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        optim.zero_grad()
        p = model(X_t)
        loss = F.binary_cross_entropy(p, y_t) + l2 * (model.w @ model.w)
        loss.backward()
        optim.step()
        if verbose and step % 200 == 0:
            print(f"[Fusion MAP] step={step} loss={loss.item():.4f}")
    with torch.no_grad():
        w = model.w.detach().cpu().numpy()
        b = float(model.b.detach().cpu().item())
    return w, b


def predict_proba_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logits = X @ w + b
    return 1 / (1 + np.exp(-logits))


# ------------------------------
# Baseline: MIN-K% PROB
# ------------------------------

def min_k_prob_baseline(model: TinyLM, tokenizer: TinyCharTokenizer, data: List[QAItem], k_tail: float = 0.2) -> np.ndarray:
    scores = []
    for it in data:
        full = canonical_text(it)
        ids = torch.tensor(tokenizer.encode(full), dtype=torch.long)
        if ids.numel() <= 1:
            scores.append(0.0)
            continue
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            logprobs = F.log_softmax(logits, dim=-1)
            tok_lp = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().numpy()
            k = max(1, int(len(tok_lp) * k_tail))
            min_k = float(np.mean(np.sort(tok_lp)[:k]))
            scores.append(-min_k)  # higher => more suspicious
    return np.array(scores)


# ------------------------------
# Feature normalization
# ------------------------------

def rank_normalize(x: np.ndarray) -> np.ndarray:
    order = x.argsort().argsort()
    return order / max(1, len(x) - 1)


# ------------------------------
# Experiment runners
# ------------------------------

def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_plan1_synthetic(seed: int, outdir: str) -> None:
    """Plan 1: Full pipeline on small synthetic math set (clean vs contaminated)."""
    outdir = _ensure_outdir(outdir)
    print("[Plan1] Generating synthetic datasets...")
    ds_math, ds_arc, ds_ood = generate_dataset(n_math=40, n_arc=40, n_ood=40, seed=seed)

    print("[Plan1] Building tokenizer from all canonical texts...")
    all_texts = [canonical_text(it) for it in (ds_math + ds_arc + ds_ood)]
    tok = TinyCharTokenizer(all_texts)

    print("[Plan1] Initializing TinyLM...")
    device = torch.device("cpu")
    model0 = TinyLM(vocab_size=len(tok.itos), hidden=64, device=device)

    print("[Plan1] Simulating training runs and checkpoints...")
    run_clean_math = build_training_run(
        tok, model0.clone(), ds_math, K=6, leak_phase=(0.0, 0.0), include_paraphrases_in_leak=False, seed=seed
    )
    run_cont_math = build_training_run(
        tok, model0.clone(), ds_math, K=6, leak_phase=(0.9, 1.0), include_paraphrases_in_leak=True, seed=seed
    )
    run_clean_ood = build_training_run(
        tok, model0.clone(), ds_ood, K=6, leak_phase=(0.0, 0.0), include_paraphrases_in_leak=False, seed=seed
    )

    # 1) SEDP
    print("[Plan1][SEDP] Evaluating paraphrase accuracies for clean and contaminated runs...")
    mid_c, fin_c = run_clean_math.mid_idx, run_clean_math.final_idx
    A_mid_clean = eval_paraphrase_accuracy(run_clean_math.checkpoints[mid_c], tok, ds_math, use_paraphrase=True)
    A_fin_clean = eval_paraphrase_accuracy(run_clean_math.checkpoints[fin_c], tok, ds_math, use_paraphrase=True)

    A_mid_cont = eval_paraphrase_accuracy(run_cont_math.checkpoints[mid_c], tok, ds_math, use_paraphrase=True)
    A_fin_cont = eval_paraphrase_accuracy(run_cont_math.checkpoints[fin_c], tok, ds_math, use_paraphrase=True)

    # Null deltas from OOD
    A_mid_null = eval_paraphrase_accuracy(run_clean_ood.checkpoints[mid_c], tok, ds_ood, use_paraphrase=True)
    A_fin_null = eval_paraphrase_accuracy(run_clean_ood.checkpoints[fin_c], tok, ds_ood, use_paraphrase=True)

    Am_clean_arr = np.array([A_mid_clean, A_mid_null])
    Af_clean_arr = np.array([A_fin_clean, A_fin_null])
    ir = fit_sedp_mapping(Am_clean_arr, Af_clean_arr)

    null_deltas = np.array([A_fin_null - ir.predict([A_mid_null])[0]])
    delta_clean, p_clean, z_clean = sedp_stat(A_mid_clean, A_fin_clean, ir, null_deltas)
    delta_cont, p_cont, z_cont = sedp_stat(A_mid_cont, A_fin_cont, ir, null_deltas)

    print(
        f"[Plan1][SEDP] Clean: A_mid={A_mid_clean:.3f}, A_fin={A_fin_clean:.3f}, Δ={delta_clean:.3f}, p={p_clean:.3f}, z={z_clean:.2f}"
    )
    print(
        f"[Plan1][SEDP] Contam: A_mid={A_mid_cont:.3f}, A_fin={A_fin_cont:.3f}, Δ={delta_cont:.3f}, p={p_cont:.3f}, z={z_cont:.2f}"
    )

    plt.figure(figsize=(4, 3))
    xs = np.linspace(0, 1, 50)
    plt.plot(xs, ir.predict(xs), label="g(A_mid) isotonic")
    plt.scatter([A_mid_clean, A_mid_cont], [A_fin_clean, A_fin_cont], c=["tab:blue", "tab:red"], label="runs")
    plt.xlabel("A_mid (paraphrase acc)")
    plt.ylabel("A_final (paraphrase acc)")
    plt.legend()
    plt.title("SEDP mapping and observed points")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sedp_mapping.pdf"), bbox_inches="tight")
    plt.close()

    # 2) SEL on final checkpoint
    print("[Plan1][SEL] Computing shard-wise permutation p-values and dataset p-value...")
    p_sel_clean, shard_scores_clean = sel_dataset(run_clean_math.checkpoints[fin_c], tok, ds_math, shard_size=20, R=5, seed=seed)
    p_sel_cont, shard_scores_cont = sel_dataset(run_cont_math.checkpoints[fin_c], tok, ds_math, shard_size=20, R=5, seed=seed)
    print(f"[Plan1][SEL] Clean dataset p-value: {p_sel_clean:.4f}")
    print(f"[Plan1][SEL] Contam dataset p-value: {p_sel_cont:.6f}")

    plt.figure(figsize=(4, 3))
    sns.histplot(-np.log10(np.clip(np.array(list(shard_scores_clean.values())), 1e-12, 1.0)), color="tab:blue", label="clean", stat="density", kde=True, bins=10)
    sns.histplot(-np.log10(np.clip(np.array(list(shard_scores_cont.values())), 1e-12, 1.0)), color="tab:red", label="contam", stat="density", kde=True, bins=10, alpha=0.5)
    plt.xlabel("-log10 p_shard")
    plt.title("SEL shard evidence distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sel_shard_evidence.pdf"), bbox_inches="tight")
    plt.close()

    # 3) GRO per-item on final checkpoint
    print("[Plan1][GRO] Scoring items by guided regeneration overlap...")
    gro_clean = [gro_item_score(run_clean_math.checkpoints[fin_c], tok, it, k_tail=0.2, max_new_tokens=64) for it in ds_math]
    gro_cont = [gro_item_score(run_cont_math.checkpoints[fin_c], tok, it, k_tail=0.2, max_new_tokens=64) for it in ds_math]
    r_clean = np.array([x[0] for x in gro_clean])
    mk_clean = np.array([x[1] for x in gro_clean])
    r_cont = np.array([x[0] for x in gro_cont])
    mk_cont = np.array([x[1] for x in gro_cont])
    print(f"[Plan1][GRO] Clean: mean ROUGE-L={r_clean.mean():.3f}, mean min-k% logprob={mk_clean.mean():.3f}")
    print(f"[Plan1][GRO] Contam: mean ROUGE-L={r_cont.mean():.3f}, mean min-k% logprob={mk_cont.mean():.3f}")

    plt.figure(figsize=(4, 3))
    plt.scatter(r_clean, mk_clean, c="tab:blue", label="clean")
    plt.scatter(r_cont, mk_cont, c="tab:red", label="contam", marker="x")
    plt.xlabel("ROUGE-L F1")
    plt.ylabel("min-k% token logprob")
    plt.legend()
    plt.title("GRO feature space")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gro_feature_scatter.pdf"), bbox_inches="tight")
    plt.close()

    # 4) Fusion on synthetic labels (all contaminated run items = 1)
    print("[Plan1][Fusion] Building per-item features and labels...")
    leaked_ids = set(list(run_cont_math.leak_ids))
    y_true = np.array([1 if it.qid in leaked_ids else 0 for it in ds_math], dtype=np.int64)

    # x1: SEDP dataset score broadcasted
    x1_cont = -math.log(max(p_cont, 1e-12))
    x1_vec = np.array([x1_cont] * len(ds_math))

    # x2: SEL shard score = -log p_shard(i), rank-normalized
    shards = make_shards(len(ds_math), shard_size=20, seed=seed)
    shard_map = {}
    for s_idx, sh in enumerate(shards):
        for i in sh:
            shard_map[i] = s_idx
    sel_scores = np.zeros(len(ds_math), dtype=np.float64)
    for i in range(len(ds_math)):
        s_idx = shard_map[i]
        p_s = shard_scores_cont.get(s_idx, 1.0)
        sel_scores[i] = -math.log(max(p_s, 1e-12))
    x2_vec = rank_normalize(sel_scores)

    # x3: GRO suspicion from features (r, min_k) via logistic MAP
    X_gro = np.column_stack([r_cont, mk_cont])
    w_gro, b_gro = fit_logistic_MAP(X_gro, y_true, l2=1e-3, steps=500, lr=0.1, verbose=False)
    x3_vec = predict_proba_logistic(X_gro, w_gro, b_gro)

    X_fuse = np.column_stack([
        (x1_vec - x1_vec.mean()) / (x1_vec.std() + 1e-8),
        x2_vec,
        x3_vec,
    ])

    print("[Plan1][Fusion] Training logistic fusion model...")
    w_fuse, b_fuse = fit_logistic_MAP(X_fuse, y_true, l2=1e-3, steps=800, lr=0.05, verbose=False)
    probs = predict_proba_logistic(X_fuse, w_fuse, b_fuse)
    ece_val = ece(probs, y_true, n_bins=10)

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec, prec)

    print(f"[Plan1][Fusion] ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}, ECE={ece_val:.3f}")

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"MuViC Fusion (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_muvic.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(rec, prec, label=f"MuViC Fusion (AUC={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_muvic.pdf"), bbox_inches="tight")
    plt.close()

    # Confusion matrix at 0.5
    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (0.5 threshold)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix_fusion.pdf"), bbox_inches="tight")
    plt.close()

    # Reliability diagram
    plt.figure(figsize=(4, 3))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(probs, bins) - 1
    bin_acc, bin_conf = [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append((bins[b] + bins[b + 1]) / 2)
        else:
            bin_acc.append(y_true[m].mean())
            bin_conf.append(probs[m].mean())
    plt.plot(bin_conf, bin_acc, marker="o", label=f"ECE={ece_val:.2f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration (reliability)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration_fusion.pdf"), bbox_inches="tight")
    plt.close()

    # 5) Baseline: MIN-K% PROB
    base_scores = min_k_prob_baseline(run_cont_math.checkpoints[fin_c], tok, ds_math, k_tail=0.2)
    fpr_b, tpr_b, _ = roc_curve(y_true, base_scores)
    roc_auc_b = auc(fpr_b, tpr_b)
    print(f"[Plan1][Baseline MIN-K] ROC AUC={roc_auc_b:.3f}")

    plt.figure(figsize=(4, 3))
    plt.plot(fpr_b, tpr_b, label=f"MIN-K% (AUC={roc_auc_b:.2f})", color="tab:orange")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_baseline.pdf"), bbox_inches="tight")
    plt.close()

    print("[Plan1] Completed. Figures saved as .pdf in:", os.path.abspath(outdir))


def run_plan2_robustness(seed: int, outdir: str) -> None:
    """Plan 2: Robustness to injection timing and checkpoint sparsity (K=3)."""
    outdir = _ensure_outdir(outdir)
    print("[Plan2] Generating dataset...")
    ds_math, _, ds_ood = generate_dataset(n_math=40, n_arc=0, n_ood=40, seed=seed)
    tok = TinyCharTokenizer([canonical_text(it) for it in (ds_math + ds_ood)])
    base_model = TinyLM(vocab_size=len(tok.itos), hidden=64)

    settings = {"early": (0.0, 0.1), "mid": (0.45, 0.55), "late": (0.9, 1.0)}
    results: Dict[str, Dict[str, float]] = {}

    for name, phase in settings.items():
        print(f"[Plan2] Injection timing: {name} phase={phase}")
        run = build_training_run(tok, base_model.clone(), ds_math, K=3, leak_phase=phase, include_paraphrases_in_leak=True, seed=seed)
        mid, fin = run.mid_idx, run.final_idx
        # SEDP
        A_mid = eval_paraphrase_accuracy(run.checkpoints[mid], tok, ds_math, use_paraphrase=True)
        A_fin = eval_paraphrase_accuracy(run.checkpoints[fin], tok, ds_math, use_paraphrase=True)
        # Fit mapping from OOD clean (simulate)
        run_ood = build_training_run(tok, base_model.clone(), ds_ood, K=3, leak_phase=(0, 0), include_paraphrases_in_leak=False, seed=seed)
        Am_null = eval_paraphrase_accuracy(run_ood.checkpoints[mid], tok, ds_ood, use_paraphrase=True)
        Af_null = eval_paraphrase_accuracy(run_ood.checkpoints[fin], tok, ds_ood, use_paraphrase=True)
        ir = fit_sedp_mapping(np.array([Am_null]), np.array([Af_null]))
        delta, p, z = sedp_stat(A_mid, A_fin, ir, np.array([Af_null - ir.predict([Am_null])[0]]))
        # SEL
        p_sel, _ = sel_dataset(run.checkpoints[fin], tok, ds_math, shard_size=20, R=5, seed=seed)
        results[name] = {"A_mid": A_mid, "A_fin": A_fin, "delta": delta, "p_sedp": p, "z": z, "p_sel": p_sel}
        print(
            f"  SEDP: A_mid={A_mid:.3f} A_fin={A_fin:.3f} Δ={delta:.3f} p={p:.3f} | SEL p_dataset={p_sel:.6f}"
        )

    plt.figure(figsize=(4, 3))
    xs = ["early", "mid", "late"]
    ys = [results[k]["z"] for k in xs]
    plt.bar(xs, ys, color=["tab:green", "tab:purple", "tab:red"])
    plt.ylabel("SEDP z-score")
    plt.title("SEDP vs injection timing")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sedp_timing.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 3))
    ys2 = [-math.log(max(results[k]["p_sel"], 1e-12)) for k in xs]
    plt.bar(xs, ys2, color=["tab:green", "tab:purple", "tab:red"])
    plt.ylabel("-log SEL p-value")
    plt.title("SEL robustness vs timing")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sel_timing.pdf"), bbox_inches="tight")
    plt.close()

    print("[Plan2] Completed. Figures saved as .pdf in:", os.path.abspath(outdir))


def run_plan3_blackbox(seed: int, outdir: str) -> None:
    """Plan 3: Black-box mode simulation (SEL + GRO only) on ARC-like set with partial contamination."""
    outdir = _ensure_outdir(outdir)
    print("[Plan3] Generating datasets...")
    ds_math, ds_arc, ds_ood = generate_dataset(n_math=40, n_arc=40, n_ood=40, seed=seed)
    tok = TinyCharTokenizer([canonical_text(it) for it in (ds_math + ds_arc + ds_ood)])
    base_model = TinyLM(vocab_size=len(tok.itos), hidden=64)

    print("[Plan3] No-checkpoint mode: SEL + GRO only...")
    model_clean = base_model.clone()  # no memory
    model_cont = base_model.clone()

    # Light contamination: store memory for 30% items
    idxs = np.arange(len(ds_arc))
    np.random.default_rng(seed).shuffle(idxs)
    contam_set = set([ds_arc[i].qid for i in idxs[: int(0.3 * len(ds_arc))]])
    for it in ds_arc:
        if it.qid in contam_set:
            model_cont.memory[it.qid] = torch.tensor(tok.encode(canonical_text(it)), dtype=torch.long)

    # Compute SEL dataset p-values
    p_sel_clean, _ = sel_dataset(model_clean, tok, ds_arc, shard_size=20, R=5, seed=seed)
    p_sel_cont, _ = sel_dataset(model_cont, tok, ds_arc, shard_size=20, R=5, seed=seed)
    print(f"[Plan3][SEL] Clean p_dataset={p_sel_clean:.4f} | Contam p_dataset={p_sel_cont:.6f}")

    # GRO per-item and simple fusion without SEDP
    gro_clean = [gro_item_score(model_clean, tok, it, k_tail=0.2, max_new_tokens=64) for it in ds_arc]
    gro_cont = [gro_item_score(model_cont, tok, it, k_tail=0.2, max_new_tokens=64) for it in ds_arc]
    r_clean = np.array([x[0] for x in gro_clean])
    mk_clean = np.array([x[1] for x in gro_clean])
    r_cont = np.array([x[0] for x in gro_cont])
    mk_cont = np.array([x[1] for x in gro_cont])

    # Build labels: 1 if in contam_set
    y_true = np.array([1 if it.qid in contam_set else 0 for it in ds_arc])
    X = np.column_stack([r_cont, mk_cont])
    w_map, b_map = fit_logistic_MAP(X, y_true, l2=1e-3, steps=600, lr=0.05)
    probs = predict_proba_logistic(X, w_map, b_map)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    print(f"[Plan3][Fusion(SEL+GRO)] ROC AUC={roc_auc:.3f}")

    plt.figure(figsize=(4, 3))
    plt.bar(["clean", "contam"], [-math.log(max(p_sel_clean, 1e-12)), -math.log(max(p_sel_cont, 1e-12))], color=["tab:blue", "tab:red"])
    plt.ylabel("-log SEL p_dataset")
    plt.title("SEL in black-box mode")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sel_blackbox.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"SEL+GRO Fusion (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (black-box)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_blackbox.pdf"), bbox_inches="tight")
    plt.close()

    print("[Plan3] Completed. Figures saved as .pdf in:", os.path.abspath(outdir))
