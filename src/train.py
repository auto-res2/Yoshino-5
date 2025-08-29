import math
import time
import json
import random
import string
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------
# Tokenizer (Toy)
# ------------------------------
class ToyTokenizer:
    def __init__(self):
        # Character-level digits + basic symbols + words needed
        self.special_tokens = ["<BOS>", "<EOS>"]
        digits = list(string.digits)
        base_vocab = [
            "Q:", "A:", "add", "and", "What", "is", "the", "sum", "?", "+",
            "Please", "solve", "Use", "steps", ",", ".",
            # distractors and retrieval tokens
            "Note:", "Fact:", "Context:", "Irrelevant:", "Relevant:", "Poison:",
        ]
        self.vocab = self.special_tokens + base_vocab + digits
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self.eos_token_id = self.stoi["<EOS>"]
        self.bos_token_id = self.stoi["<BOS>"]

    def encode(self, text: str) -> List[int]:
        tokens = []
        parts = text.strip().split()
        for w in parts:
            if w.isdigit() or (w.startswith("-") and w[1:].isdigit()):
                for ch in w:
                    if ch in self.stoi:
                        tokens.append(self.stoi[ch])
            else:
                if w in self.stoi:
                    tokens.append(self.stoi[w])
        return tokens

    def decode(self, ids: List[int], skip_special_tokens=True) -> str:
        toks = []
        for i in ids:
            if i in self.itos:
                t = self.itos[i]
                if skip_special_tokens and t in self.special_tokens:
                    continue
                toks.append(t)
        out = []
        for t in toks:
            if t in list(string.digits):
                if out and (out[-1] and out[-1][-1].isdigit()):
                    out[-1] = out[-1] + t
                else:
                    out.append(t)
            else:
                out.append(t)
        return " ".join(out)

    def __call__(self, text: str):
        ids = [self.bos_token_id] + self.encode(text)
        attn = [1] * len(ids)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }


# ------------------------------
# Utilities
# ------------------------------

def parse_addition_from_prompt(prompt: str) -> Tuple[int, int]:
    toks = prompt.replace("?", " ").replace(",", " ").replace("+", " ").split()
    ints = [int(x) for x in toks if x.lstrip("-").isdigit()]
    if len(ints) >= 2:
        return ints[0], ints[1]
    return 0, 0


def normalize_answer(ans_str: str) -> str:
    s = ans_str.strip().lower()
    s = s.replace(",", "").replace("$", "")
    return s


def extract_final_answer(text: str) -> str:
    import re
    m = re.findall(r"-?\d+", text)
    return m[-1] if m else text.strip()


# ------------------------------
# Toy Generator (PyTorch Module)
# ------------------------------
class ToyGenerator(nn.Module):
    """
    A toy conditional next-token model that 'knows' how to add two integers and emits
    the digits of the sum after 'A:'. It simulates log-prob gaps as a function of
    difficulty and perturbations.
    """
    def __init__(self, vocab_size: int, eos_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        # Tiny embedding and MLP placeholders
        self.emb = nn.Embedding(vocab_size, 16)
        self.mlp = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, vocab_size)
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Not used in the toy; placeholder to satisfy nn.Module API
        B, T = input_ids.shape
        return torch.zeros((B, T, self.vocab_size))

    def next_token_logits(self, tokenizer: ToyTokenizer, full_prompt: str, generated_ids: List[int], perturbation: Optional[Dict] = None) -> torch.Tensor:
        a, b = parse_addition_from_prompt(full_prompt)
        target = str(a + b)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ans_so_far = extract_final_answer(generated_text) if any(ch.isdigit() for ch in generated_text) else ""
        if len(ans_so_far) >= len(target):
            correct_token = tokenizer.eos_token_id
            correct_digit = None
        else:
            correct_digit = target[len(ans_so_far)]
            correct_token = tokenizer.stoi.get(correct_digit, tokenizer.eos_token_id)
        V = self.vocab_size
        logits = torch.full((V,), -5.0)
        # Difficulty heuristic
        difficulty = len(target)
        base_gap = 1.2 if difficulty <= 2 else (0.6 if difficulty == 3 else 0.3)
        if perturbation is not None:
            if perturbation.get("distractors", 0) > 0:
                base_gap *= 0.9
            if perturbation.get("long_input", False):
                base_gap *= 0.85
            if perturbation.get("poison_rate", 0.0) > 0:
                base_gap *= max(0.7, 1.0 - 2.0 * perturbation["poison_rate"])
        logits[correct_token] = 2.0 + base_gap
        incorrect_digits = [d for d in string.digits if d != (correct_digit if correct_digit is not None else "-1")]
        if incorrect_digits:
            second_digit = random.choice(incorrect_digits)
            second_token = tokenizer.stoi[second_digit]
            logits[second_token] = 2.0
        return logits


# ------------------------------
# Toy PRMs (Shared-Adapter and MoE)
# ------------------------------
class SharedAdapterPRM:
    """
    Simulates a PRM that outputs a calibrated probability p that the current partial
    trace is on track. Uses simple heuristics.
    """
    def __init__(self, d_model: int = 8):
        self.feat_proj = nn.Sequential(nn.Linear(4, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        for p in self.feat_proj.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score(self, prompt: str, tokenizer: ToyTokenizer, generated_ids: List[int], perturbation: Optional[Dict] = None) -> float:
        a, b = parse_addition_from_prompt(prompt)
        target = str(a + b)
        gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ans_len = len(extract_final_answer(gen_text)) if any(ch.isdigit() for ch in gen_text) else 0
        difficulty = len(target)
        if difficulty <= 2:
            base_p = 0.97
        elif difficulty == 3:
            base_p = 0.80
        else:
            base_p = 0.60
        if perturbation is not None:
            base_p -= 0.02 * perturbation.get("distractors", 0)
            if perturbation.get("long_input", False):
                base_p -= 0.03
            base_p -= 0.10 * perturbation.get("poison_rate", 0.0)
        x = torch.tensor([[float(difficulty), float(ans_len), float(a % 10), float(b % 10)]], dtype=torch.float32)
        z = torch.sigmoid(self.feat_proj(x)).item()
        p = max(0.0, min(1.0, 0.7 * base_p + 0.3 * z))
        return p


class MoEPRM:
    """
    Simulates a Sparse-MoE PRM with top-2 expert routing and lazy-load overhead.
    Scores whether the candidate's first new token (after prefix_len) matches the
    correct next digit.
    """
    def __init__(self, tokenizer: ToyTokenizer, lazy_overhead_ms: float = 3.0):
        self.tokenizer = tokenizer
        self.lazy_overhead_ms = lazy_overhead_ms
        self.router = nn.Linear(4, 12, bias=False)
        for p in self.router.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score_batch(self, prompt: str, candidate_ids: List[List[int]], prefix_len: int, perturbation: Optional[Dict] = None) -> Tuple[torch.Tensor, float]:
        a, b = parse_addition_from_prompt(prompt)
        target = str(a + b)
        t0 = time.perf_counter()
        _ = torch.randn(16, 16) @ torch.randn(16, 16)
        load_ms = (time.perf_counter() - t0) * 1000.0 + self.lazy_overhead_ms
        scores = []
        for ids in candidate_ids:
            # Determine the candidate's proposed next token after prefix
            if len(ids) <= prefix_len:
                # no new token; neutral low score
                p = 0.2
            else:
                cand_tok = ids[prefix_len]
                cand_sym = self.tokenizer.itos.get(cand_tok, "")
                gen_prefix = ids[:prefix_len]
                gen_text = self.tokenizer.decode(gen_prefix, skip_special_tokens=True)
                ans = extract_final_answer(gen_text) if any(ch.isdigit() for ch in gen_text) else ""
                if len(ans) >= len(target):
                    # Should emit EOS
                    p = 0.98 if cand_tok == self.tokenizer.eos_token_id else 0.10
                else:
                    true_digit = target[len(ans)]
                    p = 0.95 if cand_sym == true_digit else 0.10
            if perturbation is not None:
                p -= 0.05 * perturbation.get("poison_rate", 0.0)
            scores.append(max(0.0, min(1.0, p)))
        return torch.tensor(scores, dtype=torch.float32), load_ms


# ------------------------------
# Cascade Controller
# ------------------------------
class TinyPRMCascade:
    def __init__(
        self,
        generator: ToyGenerator,
        tokenizer: ToyTokenizer,
        shared_prm: SharedAdapterPRM,
        moe_prm: Optional[MoEPRM],
        pt_high: float = 0.95,
        pt_low: float = 0.65,
        dt_high: float = 0.7,
        beam_k: int = 5,
        horizon: int = 4,
        max_new_tokens: int = 16,
    ):
        self.gen = generator
        self.tok = tokenizer
        self.shared = shared_prm
        self.moe = moe_prm
        self.pt_high = pt_high
        self.pt_low = pt_low
        self.dt_high = dt_high
        self.beam_k = beam_k
        self.horizon = horizon
        self.max_new_tokens = max_new_tokens

    def decode_one(
        self,
        prompt: str,
        perturbation: Optional[Dict] = None,
        greedy_time_estimate: Optional[float] = None,
        budget_multiplier: float = 1.5,
    ) -> Dict:
        input_ids = self.tok(prompt)["input_ids"][0].tolist()
        generated: List[int] = []
        prm_calls = 0
        moe_calls = 0
        lazy_overheads: List[float] = []
        t0 = time.perf_counter()
        B = budget_multiplier * (greedy_time_estimate if greedy_time_estimate is not None else 0.01)

        for _ in range(self.max_new_tokens):
            logits = self.gen.next_token_logits(self.tok, prompt, generated, perturbation)
            with torch.no_grad():
                logp = F.log_softmax(logits, dim=-1)
                top2_vals, _ = torch.topk(logp, k=2)
                gap = (top2_vals[0] - top2_vals[1]).item()
            p_shared = self.shared.score(prompt, self.tok, generated, perturbation)
            prm_calls += 1
            elapsed = time.perf_counter() - t0
            budget_ok = (elapsed < B) if greedy_time_estimate is not None else True

            if (p_shared > self.pt_high) or (gap > self.dt_high) or (not budget_ok):
                next_id = torch.argmax(logits).item()
            elif (p_shared > self.pt_low) and (self.moe is not None) and budget_ok:
                beams: List[List[int]] = []
                for _k in range(self.beam_k):
                    cand = generated.copy()
                    for _h in range(self.horizon):
                        logits_k = self.gen.next_token_logits(self.tok, prompt, cand, perturbation)
                        probs = F.softmax(logits_k / 0.7, dim=-1)
                        tok_id = torch.multinomial(probs, 1).item()
                        cand.append(tok_id)
                        if tok_id == self.tok.eos_token_id:
                            break
                    beams.append(cand)
                scores, overhead_ms = self.moe.score_batch(prompt, beams, prefix_len=len(generated), perturbation=perturbation)
                best = torch.argmax(scores).item()
                lazy_overheads.append(overhead_ms)
                moe_calls += 1
                if len(beams[best]) <= len(generated):
                    next_id = torch.argmax(logits).item()
                else:
                    next_id = beams[best][len(generated)]
            else:
                next_id = torch.argmax(logits).item()

            generated.append(next_id)
            if next_id == self.tok.eos_token_id:
                break

        text = self.tok.decode(generated, skip_special_tokens=True)
        total_time = time.perf_counter() - t0
        return {
            "text": text,
            "wall_seconds": total_time,
            "tokens": len(generated),
            "prm_calls": prm_calls,
            "moe_calls": moe_calls,
            "lazy_overhead_ms": lazy_overheads,
        }


# ------------------------------
# Baselines
# ------------------------------

def run_greedy(generator: ToyGenerator, tokenizer: ToyTokenizer, prompt: str, max_new_tokens: int = 16, perturbation: Optional[Dict] = None) -> Dict:
    generated: List[int] = []
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = generator.next_token_logits(tokenizer, prompt, generated, perturbation)
        next_id = torch.argmax(logits).item()
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"text": text, "wall_seconds": time.perf_counter() - t0}


def run_self_consistency(generator: ToyGenerator, tokenizer: ToyTokenizer, prompt: str, n: int = 5, max_new_tokens: int = 16, perturbation: Optional[Dict] = None) -> Dict:
    answers: List[str] = []
    t0 = time.perf_counter()
    for _ in range(n):
        generated: List[int] = []
        for _ in range(max_new_tokens):
            logits = generator.next_token_logits(tokenizer, prompt, generated, perturbation)
            probs = F.softmax(logits / 0.9, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
        answers.append(tokenizer.decode(generated, skip_special_tokens=True))
    return {"answers": answers, "wall_seconds": time.perf_counter() - t0}


# ------------------------------
# Public factory (used by src.main)
# ------------------------------

def build_models(config: Optional[Dict] = None) -> Dict:
    tok = ToyTokenizer()
    gen = ToyGenerator(vocab_size=len(tok.vocab), eos_token_id=tok.eos_token_id)
    shared = SharedAdapterPRM()
    moe = MoEPRM(tok)
    cascade = TinyPRMCascade(
        gen, tok, shared, moe,
        pt_high=float(config.get("pt_high", 0.95)) if config else 0.95,
        pt_low=float(config.get("pt_low", 0.65)) if config else 0.65,
        dt_high=float(config.get("dt_high", 0.7)) if config else 0.7,
        beam_k=int(config.get("beam_k", 5)) if config else 5,
        horizon=int(config.get("horizon", 4)) if config else 4,
        max_new_tokens=int(config.get("max_new_tokens", 8)) if config else 8,
    )
    return {
        "tokenizer": tok,
        "generator": gen,
        "shared_prm": shared,
        "moe_prm": moe,
        "cascade": cascade,
    }
