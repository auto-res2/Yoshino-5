#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing and synthetic data generation for MuViC experiments.
Includes a tiny character tokenizer and simple QA item generators.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ------------------------------
# Tiny tokenizer (character-level)
# ------------------------------
class TinyCharTokenizer:
    def __init__(self, texts: List[str]):
        base_chars = list(
            "\n\t .,;:!?-_=+*/()[]{}<>'\"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        vocab_set = set(base_chars)
        for t in texts:
            for ch in t:
                vocab_set.add(ch)
        self.itos = ["<pad>"] + sorted(list(vocab_set))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_token_id = 0

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.pad_token_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)


# ------------------------------
# Synthetic dataset generation
# ------------------------------
@dataclass
class QAItem:
    qid: str
    q: str
    r: str
    a: str


def make_math_item(idx: int, rng: np.random.Generator) -> QAItem:
    x = int(rng.integers(2, 50))
    y = int(rng.integers(2, 50))
    q = f"What is {x} + {y}?"
    r = f"Add the two numbers: {x} + {y} = {x+y}."
    a = str(x + y)
    return QAItem(qid=f"math_{idx}", q=q, r=r, a=a)


def make_arc_item(idx: int, rng: np.random.Generator) -> QAItem:
    concepts = [
        ("Which state of matter has a definite volume but no definite shape?", "B"),
        ("What gas do plants primarily take in for photosynthesis?", "C"),
        ("Which force keeps planets in orbit around the Sun?", "A"),
        ("What is the primary function of white blood cells?", "D"),
    ]
    options = {
        0: ["A) Solid", "B) Liquid", "C) Gas", "D) Plasma"],
        1: ["A) Oxygen", "B) Nitrogen", "C) Carbon dioxide", "D) Hydrogen"],
        2: ["A) Gravity", "B) Magnetism", "C) Friction", "D) Electricity"],
        3: ["A) Carry oxygen", "B) Carry nutrients", "C) Clot blood", "D) Fight infection"],
    }
    k = int(rng.integers(0, len(concepts)))
    stem, ans = concepts[k]
    opt_str = " ".join(options[k])
    q = f"{stem} Choose the best answer. {opt_str}"
    r = "We recall basic science facts and eliminate distractors."
    a = ans
    return QAItem(qid=f"arc_{idx}", q=q, r=r, a=a)


def make_ood_item(idx: int, rng: np.random.Generator) -> QAItem:
    facts = [
        ("In which continent is the Sahara Desert located?", "Africa"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Which ocean is the largest on Earth?", "Pacific"),
        ("How many continents are there?", "7"),
    ]
    k = int(rng.integers(0, len(facts)))
    q, a = facts[k]
    r = "Use general world knowledge to answer succinctly."
    return QAItem(qid=f"ood_{idx}", q=q, r=r, a=a)


def generate_dataset(n_math: int = 40, n_arc: int = 40, n_ood: int = 40, seed: int = 123) -> Tuple[List[QAItem], List[QAItem], List[QAItem]]:
    rng = np.random.default_rng(seed)
    ds_math = [make_math_item(i, rng) for i in range(n_math)]
    ds_arc = [make_arc_item(i, rng) for i in range(n_arc)]
    ds_ood = [make_ood_item(i, rng) for i in range(n_ood)]
    return ds_math, ds_arc, ds_ood


def canonical_text(item: QAItem) -> str:
    r = item.r if item.r is not None else ""
    return f"You are a helpful assistant.\nQuestion: {item.q}\nReasoning: {r}\nAnswer: {item.a}"


def paraphrase_question(q: str) -> str:
    repl = [
        ("What is", "Compute"),
        ("Which", "Identify which"),
        ("Choose the best answer.", "Pick the most appropriate option."),
        ("How many", "Determine the number of"),
    ]
    q2 = q
    for a, b in repl:
        q2 = q2.replace(a, b)
    if q2 == q:
        q2 = "Rephrase: " + q
    return q2


def prompt_prefix_for_answer(item: QAItem, paraphrase: bool = False) -> str:
    if paraphrase:
        q = paraphrase_question(item.q)
    else:
        q = item.q
    r = item.r if item.r is not None else ""
    return f"You are a helpful assistant.\nQuestion: {q}\nReasoning: {r}\nAnswer: "
