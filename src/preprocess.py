# -*- coding: utf-8 -*-
"""
Preprocessing, verification and dataset building for EXACT experiments.
Contains:
  - Synthetic data generators (math, commonsense, MGSM-style)
  - Step-level verification: math via SymPy/Python eval; commonsense via toy RAG+entailment
  - Minimal core extraction via ILP (PuLP) or greedy fallback
  - Dataset assembly with curriculum masking
"""
import os
import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application
    )
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# Regex helpers
STEP_SPLIT_RE = re.compile(r"(?:^|\n)\s*\d+[).]\s+")
NUM_PAT = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
ASSIGN_PAT = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]+)")
EQUAL_PAT = re.compile(r"(.+?)\s*=\s*(.+)")

TRANSFORMATIONS = []
if SYMPY_AVAILABLE:
    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str = "auto") -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------- Step utilities -------------

def split_steps(text: str) -> List[str]:
    parts = STEP_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [s.strip() for s in text.split("\n") if s.strip()]
    parts = [p for p in parts if not p.lower().startswith("final answer")] 
    return parts


# ------------- Math verification -------------

class MathState:
    def __init__(self):
        self.env = {}

    def eval_expr(self, expr_str: str):
        if not SYMPY_AVAILABLE:
            expr_str = expr_str.replace(",", "")
            val = eval(expr_str, {"__builtins__": {}}, dict(self.env))
            return float(val)
        expr_str = expr_str.replace(",", "")
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, evaluate=True)

    def eval_assignment(self, var: str, rhs: str):
        try:
            rhs_expr = self.eval_expr(rhs)
            if SYMPY_AVAILABLE:
                self.env[var] = sp.N(rhs_expr)
            else:
                self.env[var] = rhs_expr
            return True
        except Exception:
            return False

    def substitute(self, expr: str):
        if SYMPY_AVAILABLE:
            subs = {sp.Symbol(k): v for k, v in self.env.items()}
            try:
                return sp.N(sp.sympify(expr).subs(subs))
            except Exception:
                expr2 = parse_expr(expr, transformations=TRANSFORMATIONS, evaluate=True)
                return sp.N(expr2.subs(subs))
        for k, v in self.env.items():
            expr = re.sub(rf"\b{k}\b", str(v), expr)
        return float(eval(expr, {"__builtins__": {}}, {}))

    def check_equality(self, left: str, right: str, rtol=1e-6, atol=1e-8):
        try:
            L = self.substitute(left)
            R = self.substitute(right)
            if SYMPY_AVAILABLE:
                return sp.Abs(L - R) <= max(atol, rtol*sp.Max(sp.Abs(L), sp.Abs(R)))
            return abs(L - R) <= max(atol, rtol*max(abs(L), abs(R)))
        except Exception:
            return False


def verify_step_math(step: str, state: MathState) -> Dict:
    meta = {"assigns": [], "equalities": []}
    try:
        any_verified = False
        any_rejected = False
        for m in ASSIGN_PAT.finditer(step):
            var, rhs = m.group(1).strip(), m.group(2).strip()
            ok = state.eval_assignment(var, rhs)
            meta["assigns"].append((var, rhs, bool(ok)))
            any_verified = any_verified or ok
        eqls = EQUAL_PAT.findall(step)
        for (L, R) in eqls:
            ok = state.check_equality(L.strip(), R.strip())
            meta["equalities"].append(((L.strip(), R.strip()), bool(ok)))
            any_verified = any_verified or ok
            any_rejected = any_rejected or (ok is False)
        if any_verified and not any_rejected:
            return {"verified": True, "rejected": False, "meta": meta}
        if any_rejected and not any_verified:
            return {"verified": False, "rejected": True, "meta": meta}
        return {"verified": False, "rejected": False, "meta": meta}
    except Exception as e:
        meta["error"] = str(e)
        return {"verified": False, "rejected": False, "meta": meta}


def verify_trace_math(teacher_cot: str) -> Tuple[List[str], List[int], List[int], List[Dict]]:
    steps = split_steps(teacher_cot)
    state = MathState()
    vmask, rmask, metas = [], [], []
    for s in steps:
        out = verify_step_math(s, state)
        vmask.append(1 if out["verified"] else 0)
        rmask.append(1 if out["rejected"] else 0)
        metas.append(out["meta"])
    return steps, vmask, rmask, metas


# ------------- Minimal core extraction -------------

def token_len(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


def extract_symbols(step: str) -> set:
    vars_ = [m.group(1) for m in ASSIGN_PAT.finditer(step)]
    nums_ = NUM_PAT.findall(step)
    return set(vars_ + nums_)


def build_dependency(steps: List[str], verified_mask: List[int]) -> Tuple[List[int], Dict[int, set]]:
    verified_idx = [i for i, v in enumerate(verified_mask) if v == 1]
    covers = {}
    for i in verified_idx:
        covers[i] = extract_symbols(steps[i])
    return verified_idx, covers


def answer_symbols(answer_text: str) -> set:
    return set(NUM_PAT.findall(answer_text or ""))


def minimal_core_ilp(steps: List[str], verified_mask: List[int], final_answer_syms: set, tokenizer=None) -> List[int]:
    idxs, covers = build_dependency(steps, verified_mask)
    if not idxs or not final_answer_syms:
        return idxs
    selected = []
    if PULP_AVAILABLE:
        prob = pulp.LpProblem("min_core", pulp.LpMinimize)
        x = {i: pulp.LpVariable(f"x_{i}", 0, 1, cat=pulp.LpBinary) for i in idxs}
        prob += pulp.lpSum([token_len(steps[i], tokenizer) * x[i] for i in idxs])
        U = set(final_answer_syms)
        for u in U:
            prob += pulp.lpSum([x[i] for i in idxs if u in covers[i]]) >= 1
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status == 1:
            selected = [i for i in idxs if pulp.value(x[i]) > 0.5]
    if not selected:
        # Greedy fallback
        U = set(final_answer_syms)
        uncovered = set(U)
        avail = set(idxs)
        chosen = []
        while uncovered and avail:
            best = max(
                avail,
                key=lambda i: (len(covers[i] & uncovered) + 1e-6) / (1 + token_len(steps[i], tokenizer))
            )
            gain = len(covers[best] & uncovered)
            if gain == 0:
                break
            chosen.append(best)
            uncovered -= covers[best]
            avail.remove(best)
        selected = sorted(chosen)
    return selected


# ------------- Synthetic data (Math) -------------

def make_synthetic_math_teacher_data(n: int = 10) -> List[Dict]:
    problems = []
    templates = [
        ("Alice has {a} apples and buys {b} more. How many apples does she have now?",
         lambda a,b: a+b),
        ("If a book costs {a} dollars and a pen costs {b} dollars, what is the total cost?",
         lambda a,b: a+b),
        ("Tom walked {a} km in the morning and {b} km in the evening. Total distance?",
         lambda a,b: a+b),
        ("A box has {a} red balls and {b} blue balls. How many balls in total?",
         lambda a,b: a+b),
    ]
    for i in range(n):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        fmt, fn = templates[i % len(templates)]
        q = fmt.format(a=a, b=b)
        ans = fn(a,b)
        steps = []
        steps.append(f"Let x = {a}")
        steps.append(f"Let y = {b}")
        if i % 3 == 0:
            wrong = a + b + 1
            steps.append(f"Compute s = x + y = {wrong}")
            steps.append(f"But re-check: s = x + y = {a+b}")
        else:
            steps.append(f"Compute s = x + y = {a+b}")
        steps.append(f"Therefore, the total is s = {a+b}")
        cot = "\n".join([f"{j+1}) {st}" for j, st in enumerate(steps)])
        cot += f"\nFinal Answer: {ans}"
        problems.append({
            "id": f"math_{i}",
            "problem": q,
            "teacher_cot": cot,
            "teacher_answer": str(ans),
        })
    return problems


# ------------- Build EXACT examples (Math) -------------

def build_exact_examples_from_teacher_math(teacher_items: List[Dict], tokenizer) -> List[Dict]:
    examples = []
    for ex in teacher_items:
        steps, vmask, rmask, _ = verify_trace_math(ex["teacher_cot"])  
        U = answer_symbols(ex.get("teacher_answer", ""))
        core_idx = minimal_core_ilp(steps, vmask, U, tokenizer)
        examples.append({
            "id": ex["id"],
            "problem": ex["problem"],
            "teacher_cot": ex["teacher_cot"],
            "teacher_answer": ex.get("teacher_answer", ""),
            "steps": steps,
            "verified_mask": vmask,
            "rejected_mask": rmask,
            "core_indices": core_idx,
        })
    return examples


# ------------- Commonsense toy RAG/entailment -------------
class ToyRetriever:
    def __init__(self, passages: List[str]):
        self.passages = passages
        self.index = [set(re.findall(r"\w+", p.lower())) for p in passages]
    def search(self, query: str, k: int = 5) -> List[str]:
        qset = set(re.findall(r"\w+", query.lower()))
        scores = []
        for i, pset in enumerate(self.index):
            score = len(qset & pset) / max(1, len(qset))
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self.passages[i] for _, i in scores[:k]]

def toy_entailment(premise: str, hypothesis: str) -> Tuple[float, float]:
    p = set(re.findall(r"\w+", premise.lower()))
    h = set(re.findall(r"\w+", hypothesis.lower()))
    overlap = len(p & h) / max(1, len(h))
    contra = 1.0 - overlap if any(w in premise.lower() for w in ["not", "no", "never"]) else 0.2 * (1.0 - overlap)
    entail = min(1.0, max(0.0, overlap + 0.1))
    return entail, contra

def verify_step_rag(question: str, step: str, retriever: ToyRetriever, entail_thr: float = 0.7, contra_thr: float = 0.6) -> Dict:
    ctxs = retriever.search(question + "\n" + step, k=5)
    entails, contras = 0.0, 0.0
    for c in ctxs:
        e, cprob = toy_entailment(c, step)
        entails = max(entails, e)
        contras = max(contras, cprob)
    if entails >= entail_thr and entails > contras:
        return {"verified": True, "rejected": False, "e": entails, "c": contras}
    if contras >= contra_thr and contras > entails:
        return {"verified": False, "rejected": True, "e": entails, "c": contras}
    return {"verified": False, "rejected": False, "e": entails, "c": contras}


def make_synthetic_commonsense_teacher_data(n: int = 9) -> List[Dict]:
    questions = []
    items = [
        {
            "q": "Which animal is a mammal?",
            "options": ["Shark", "Dolphin", "Eagle", "Lizard"],
            "gold": "B",
            "facts": [
                "Dolphins are mammals and breathe air.",
                "Sharks are fish, not mammals.",
                "Eagles are birds.",
            ],
            "rationale": [
                "Sharks are fish, so not mammals.",
                "Dolphins nurse their young and are warm-blooded, so they are mammals.",
                "Thus, choose B.",
            ],
        },
        {
            "q": "What do bees collect to make honey?",
            "options": ["Leaves", "Nectar", "Seeds", "Bark"],
            "gold": "B",
            "facts": ["Bees collect nectar from flowers to make honey."],
            "rationale": [
                "Leaves are not used to make honey.",
                "Bees collect nectar from flowers.",
                "So answer is B.",
            ],
        },
        {
            "q": "Which season typically has the coldest weather?",
            "options": ["Spring", "Summer", "Autumn", "Winter"],
            "gold": "D",
            "facts": ["Winter is usually the coldest season."],
            "rationale": [
                "Summer is the hottest season.",
                "Winter is the coldest season.",
                "Therefore choose D.",
            ],
        },
    ]
    idx = 0
    while len(questions) < n:
        it = items[idx % len(items)]
        idx += 1
        steps = []
        if len(questions) % 3 == 0:
            steps.append("Sharks are mammals.")
        steps.extend(it["rationale"])
        cot = "\n".join([f"{j+1}) {st}" for j, st in enumerate(steps)])
        cot += f"\nFinal Answer: {it['gold']}"
        questions.append({
            "id": f"csqa_{len(questions)}",
            "problem": it["q"] + "\nOptions: " + ", ".join([f"{chr(65+i)}) {opt}" for i,opt in enumerate(it["options"])]) + "\n",
            "teacher_cot": cot,
            "teacher_answer": it["gold"],
            "facts": it["facts"],
        })
    return questions


def build_exact_examples_from_teacher_csqa(teacher_items: List[Dict], retriever: ToyRetriever) -> List[Dict]:
    examples = []
    for ex in teacher_items:
        steps = split_steps(ex["teacher_cot"])  
        vmask, rmask, metas = [], [], []
        for s in steps:
            out = verify_step_rag(ex["problem"], s, retriever)
            vmask.append(1 if out["verified"] else 0)
            rmask.append(1 if out["rejected"] else 0)
            metas.append(out)
        core_idx = [i for i, v in enumerate(vmask) if v == 1]
        examples.append({
            "id": ex["id"],
            "problem": ex["problem"],
            "teacher_cot": ex["teacher_cot"],
            "teacher_answer": ex.get("teacher_answer", ""),
            "steps": steps,
            "verified_mask": vmask,
            "rejected_mask": rmask,
            "core_indices": core_idx,
        })
    return examples


# ------------- MGSM style synthetic -------------

def make_synthetic_mgsm_items() -> Dict[str, List[Dict]]:
    items = {
        "en": [
            {"q": "John has 3 apples and buys 4 more. How many apples now?", "a": "7"},
            {"q": "Sara walked 2 km and then 5 km. What is the total distance?", "a": "7"},
        ],
        "es": [
            {"q": "Juan tiene 3 manzanas y compra 4 más. ¿Cuántas tiene ahora?", "a": "7"},
            {"q": "Sara caminó 2 km y luego 5 km. ¿Cuál es la distancia total?", "a": "7"},
        ],
        "fr": [
            {"q": "Jean a 3 pommes et en achète 4 de plus. Combien en a-t-il maintenant?", "a": "7"},
        ],
        "de": [
            {"q": "Hans hat 3 Äpfel und kauft 4 weitere. Wie viele hat er jetzt?", "a": "7"},
        ],
    }
    return items


# ------------- Dataset for EXACT -------------
class ExactDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, curriculum_ratio: float = 0.0):
        self.examples = examples
        self.tokenizer = tokenizer
        self.curriculum_ratio = curriculum_ratio

    def set_curriculum(self, ratio: float):
        self.curriculum_ratio = ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tok = self.tokenizer
        problem = ex["problem"].strip()
        steps = ex["steps"]
        vmask = ex["verified_mask"]
        rmask = ex["rejected_mask"]
        core_idx = set(ex.get("core_indices", []))

        verified_idxs = [i for i, v in enumerate(vmask) if v == 1]
        interm = [i for i in verified_idxs if i not in core_idx]
        hide_k = int(self.curriculum_ratio * len(interm))
        hide_set = set(random.sample(interm, hide_k)) if hide_k > 0 else set()

        prompt = f"Problem: {problem}\nReasoning:\n"
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        input_ids = list(prompt_ids)
        labels = [-100] * len(prompt_ids)
        core_mask = [0] * len(prompt_ids)
        rej_mask_toks = [0] * len(prompt_ids)

        for i, s in enumerate(steps):
            prefix = f"{i+1}) "
            line = prefix + s.strip() + "\n"
            line_ids = tok(line, add_special_tokens=False).input_ids
            teach = True
            tag_core = (vmask[i] == 1 and i in core_idx and i not in hide_set)
            tag_rej = (rmask[i] == 1)
            if vmask[i] == 1 and i in hide_set:
                teach = False
            input_ids.extend(line_ids)
            labels.extend(line_ids if teach else [-100] * len(line_ids))
            core_mask.extend([1 if tag_core else 0] * len(line_ids))
            rej_mask_toks.extend([1 if tag_rej else 0] * len(line_ids))

        answer = ex.get("teacher_answer", "")
        tail = f"Final Answer: {answer}\n"
        tail_ids = tok(tail, add_special_tokens=False).input_ids
        input_ids.extend(tail_ids)
        labels.extend(tail_ids)
        core_mask.extend([0] * len(tail_ids))
        rej_mask_toks.extend([0] * len(tail_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "core_mask": torch.tensor(core_mask, dtype=torch.float),
            "rej_mask": torch.tensor(rej_mask_toks, dtype=torch.float),
        }


def collate_exact(batch: List[Dict], pad_id: int):
    max_len = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    core_mask = torch.zeros((B, max_len), dtype=torch.float)
    rej_mask = torch.zeros((B, max_len), dtype=torch.float)
    for i, it in enumerate(batch):
        L = it["input_ids"].size(0)
        input_ids[i, :L] = it["input_ids"]
        labels[i, :L] = it["labels"]
        core_mask[i, :L] = it["core_mask"]
        rej_mask[i, :L] = it["rej_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "core_mask": core_mask,
        "rej_mask": rej_mask,
    }
