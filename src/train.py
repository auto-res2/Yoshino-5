#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities for MuViC synthetic experiments.
- TinyLM: minimal bigram-like LM with a contamination memory mechanism.
- build_training_run: simulates checkpoints and late-phase leakage.
- eval_paraphrase_accuracy: helper for SEDP component.

All imports within src use relative paths as required.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .preprocess import (
        TinyCharTokenizer,
        QAItem,
        canonical_text,
        prompt_prefix_for_answer,
    )
except ImportError:  # Fallback when running as a script without package context
    from preprocess import (  # type: ignore
        TinyCharTokenizer,
        QAItem,
        canonical_text,
        prompt_prefix_for_answer,
    )


class TinyLM(nn.Module):
    """A tiny bigram-like language model with a contamination memory.

    - Next-token logits depend only on the current token (bigram) via Embedding->Linear.
    - Memory: dict[id] = canonical tokens; acts as both a logit boost during logprob and a strict copy for generation
      when the prefix exactly matches the memorized prefix.
    """

    def __init__(self, vocab_size: int, hidden: int = 64, device: Optional[torch.device] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, hidden)
        self.lin = nn.Linear(hidden, vocab_size)
        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)
        # contamination memory: id -> torch.LongTensor tokens of canonical (or paraphrase) full text
        self.memory: Dict[str, torch.LongTensor] = {}
        self.memory_boost = 3.0  # logit boost for matching memorized next-token

    def forward(self, x: torch.LongTensor):
        # x: [B, T] current tokens; returns next-token logits for each position [B, T, V]
        h = self.emb(x)
        logits = self.lin(h)
        return logits

    def clone(self) -> "TinyLM":
        new = TinyLM(self.vocab_size, self.emb.embedding_dim, self.device)
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        new.memory = {k: v.clone() for k, v in self.memory.items()}
        new.memory_boost = self.memory_boost
        return new

    @torch.no_grad()
    def avg_logprob(self, tokenizer: TinyCharTokenizer, text: str, qid: Optional[str] = None) -> float:
        """Average per-token logprob under the model with optional memory boost.
        If qid is in memory and prefix alignment holds, boost the logits for memorized next tokens.
        """
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=self.device)
        if ids.numel() <= 1:
            return 0.0
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        logits = self.forward(x)
        # memory boost on matching memorized sequence
        if qid is not None and qid in self.memory:
            mem = self.memory[qid]
            L = min(mem.numel(), ids.numel())
            match_mask = torch.zeros_like(y)
            if L > 1:
                # If prefix exactly matches, apply boost along the aligned region
                eq_prefix = torch.all(ids[:L] == mem[:L])
                if eq_prefix:
                    target_next = mem[1:L].unsqueeze(0)
                    match_mask[:, : L - 1] = (y[:, : L - 1] == target_next).long()
                else:
                    # Partial alignment: boost where y matches mem next tokens and prefix matches
                    Lm = min(L - 1, x.shape[1])
                    if Lm > 0:
                        prefix_equal = torch.all(x[:, :Lm] == mem[:Lm])
                        if prefix_equal:
                            target_next = mem[1 : Lm + 1].unsqueeze(0)
                            match_mask[:, :Lm] = (y[:, :Lm] == target_next).long()
            for t in range(y.shape[1]):
                if match_mask[0, t].item() == 1:
                    logits[0, t, y[0, t]] += self.memory_boost
        logprobs = F.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        avg_lp = (tok_lp.sum() / max(1, tok_lp.numel())).item()
        return float(avg_lp)

    @torch.no_grad()
    def generate(
        self,
        tokenizer: TinyCharTokenizer,
        prefix: str,
        max_new_tokens: int = 64,
        qid: Optional[str] = None,
    ) -> str:
        """Greedy generation with hard copy from memory if the prefix matches a memorized prefix.
        """
        pref_ids = torch.tensor(tokenizer.encode(prefix), dtype=torch.long, device=self.device)
        if qid is not None and qid in self.memory:
            mem = self.memory[qid]
            Lp = pref_ids.numel()
            if Lp <= mem.numel() and torch.all(pref_ids == mem[:Lp]):
                suffix = mem[Lp:].tolist()
                return tokenizer.decode(suffix[:max_new_tokens])
        # Else, bigram-greedy
        cur = pref_ids.clone()
        out_ids = []
        for _ in range(max_new_tokens):
            if cur.numel() == 0:
                x = torch.tensor([[0]], dtype=torch.long, device=self.device)
            else:
                x = cur[-1:].unsqueeze(0)
            logits = self.forward(x)[:, -1, :]
            next_id = torch.argmax(logits, dim=-1).item()
            out_ids.append(next_id)
            cur = torch.cat([cur, torch.tensor([next_id], device=self.device)])
        return tokenizer.decode(out_ids)


@dataclass
class TrainingRun:
    checkpoints: List[TinyLM]
    mid_idx: int
    final_idx: int
    leak_ids: Set[str]


def build_training_run(
    tokenizer: TinyCharTokenizer,
    model: TinyLM,
    dataset: List[QAItem],
    K: int = 6,
    leak_phase: Tuple[float, float] = (0.9, 1.0),
    include_paraphrases_in_leak: bool = True,
    seed: int = 123,
) -> TrainingRun:
    """Simulates K evenly spaced checkpoints; injects leakage memory during the specified final phase.
    include_paraphrases_in_leak=True simulates paraphrase exposure, affecting SEDP.
    """
    import numpy as np

    steps = 100
    leak_mask = np.zeros(steps, dtype=bool)
    start, end = leak_phase
    leak_mask[int(start * steps) : int(end * steps)] = True
    cps: List[TinyLM] = []
    leak_ids: Set[str] = set()

    # Precompute canonical and optionally paraphrase texts for memory
    mem_texts: Dict[str, torch.LongTensor] = {}
    para_texts: Dict[str, torch.LongTensor] = {}
    for it in dataset:
        full = canonical_text(it)
        mem_texts[it.qid] = torch.tensor(tokenizer.encode(full), dtype=torch.long, device=model.device)
        if include_paraphrases_in_leak:
            pfx = prompt_prefix_for_answer(it, paraphrase=True) + it.a
            para_texts[it.qid] = torch.tensor(tokenizer.encode(pfx), dtype=torch.long, device=model.device)

    mid_idx = K // 2
    final_idx = K - 1
    for k in range(K):
        step = int((k + 1) * steps / K)
        # Simulate a tiny SGD step (noise) to break exact ties; no real training needed
        for p in model.parameters():
            p.data.add_(0.001 * torch.randn_like(p))
        ck = model.clone()
        # Inject memory if within leak phase
        if leak_mask[step - 1]:
            for it in dataset:
                ck.memory[it.qid] = mem_texts[it.qid]
                leak_ids.add(it.qid)
                if include_paraphrases_in_leak and it.qid in para_texts:
                    ck.memory[it.qid + "_para"] = para_texts[it.qid]
        cps.append(ck)
    return TrainingRun(checkpoints=cps, mid_idx=mid_idx, final_idx=final_idx, leak_ids=leak_ids)


def eval_paraphrase_accuracy(
    model: TinyLM,
    tokenizer: TinyCharTokenizer,
    dataset: List[QAItem],
    use_paraphrase: bool = True,
    max_new_tokens: int = 16,
) -> float:
    """Simple answer accuracy on (optionally) paraphrased questions.
    If use_paraphrase=True, we prompt with paraphrases and allow memory via qid+"_para".
    """
    correct = 0
    for it in dataset:
        pfx = prompt_prefix_for_answer(it, paraphrase=use_paraphrase)
        qid_for_mem = it.qid + ("_para" if use_paraphrase else "")
        gen = model.generate(tokenizer, pfx, max_new_tokens=max_new_tokens, qid=qid_for_mem)
        pred = gen.strip().split()[0] if len(gen.strip()) > 0 else ""
        gold = it.a.strip().split()[0]
        correct += int(pred == gold)
    acc = correct / max(1, len(dataset))
    return acc
