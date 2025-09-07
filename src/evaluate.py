"""src/evaluate.py
Evaluation / metric helpers (token-level F1, Levenshtein-tolerant EM, …).
"""
from __future__ import annotations

from typing import List

import torch
from sklearn.metrics import f1_score


# -----------------------------------------------------------------------------
#  Token-level micro-F1 (used in EXP-1)
# -----------------------------------------------------------------------------

def token_micro_f1(pred: torch.LongTensor, gold: torch.LongTensor) -> float:
    mask = gold != -100
    if mask.sum() == 0:
        return 0.0
    return f1_score(gold[mask].cpu(), pred[mask].cpu(), average="micro")


# -----------------------------------------------------------------------------
#  Sequence-level exact match with Levenshtein tolerance ≤ 1 (EXP-2/3)
# -----------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def tolerant_exact_match(preds: List[str], golds: List[str]) -> float:
    hit = 0
    for p, g in zip(preds, golds):
        hit += _levenshtein(p.strip(), g.strip()) <= 1
    return hit / max(1, len(preds))
