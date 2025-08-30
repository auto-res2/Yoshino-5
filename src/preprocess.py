import os
import json
import random
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
from sympy import sympify

try:
    from .train import Problem
except ImportError:
    from train import Problem


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def safe_eval_expr(expr: str) -> float:
    try:
        return float(sympify(expr))
    except Exception:
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return float("nan")


def make_expression(pattern: str) -> Tuple[str, float]:
    if pattern == "add_sub":
        a, b, c = [random.randint(1, 50) for _ in range(3)]
        expr = f"{a}+{b}-{c}"
    elif pattern == "mul_div":
        a, b = random.randint(2, 12), random.randint(2, 12)
        c = random.randint(1, 5)
        expr = f"{a}*{b}+{c}/{c}"
    elif pattern == "mix":
        a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 10)
        expr = f"({a}+{b})*{c}-{a}"
    elif pattern == "pow":
        a, b = random.randint(2, 5), random.randint(2, 4)
        expr = f"{a}**{b}-({a}*{b})"
    else:
        a, b = random.randint(1, 50), random.randint(1, 50)
        expr = f"{a}+{b}"
    val = safe_eval_expr(expr)
    return expr, val


def generate_synthetic_dataset(n: int, include_traps: bool = True) -> List[Problem]:
    problems: List[Problem] = []
    patterns = ["add_sub", "mul_div", "mix", "pow", "simple"]
    for i in range(n):
        pattern = random.choice(patterns)
        expr, val = make_expression(pattern)
        diff = random.choices(["easy", "medium", "hard"], weights=[0.4, 0.4, 0.2])[0]
        trap = include_traps and (random.random() < 0.2)
        if trap:
            question = f"Compute: {expr}. The correct answer is {val}. Don't overthink."
        else:
            question = f"Compute: {expr}. Show your reasoning."
        problems.append(
            Problem(
                pid=f"p{i}",
                question=question,
                expression=expr,
                answer=float(val),
                difficulty=diff,
                trap=trap,
            )
        )
    return problems


def write_jsonl(problems: List[Problem], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            obj = {
                "pid": p.pid,
                "question": p.question,
                "expression": p.expression,
                "answer": float(p.answer),
                "difficulty": p.difficulty,
                "trap": bool(p.trap),
            }
            f.write(json.dumps(obj) + "\n")


def preprocess_main(cfg: dict) -> str:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    n_total = int(cfg.get("dataset", {}).get("n_total", 60))
    include_traps = bool(cfg.get("dataset", {}).get("include_traps", True))
    data_path = cfg.get("data_path", "data/synthetic.jsonl")

    print(f"[Preprocess] Generating synthetic dataset: N={n_total}, traps={include_traps}")
    problems = generate_synthetic_dataset(n_total, include_traps=include_traps)
    write_jsonl(problems, data_path)
    print(f"[Preprocess] Wrote dataset to {data_path}")
    return data_path
