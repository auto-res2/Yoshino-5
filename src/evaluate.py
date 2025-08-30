# -*- coding: utf-8 -*-
"""
Evaluation and plotting utilities for EXACT experiments.
Saves figures in high-quality PDF format suitable for academic papers.
"""
from typing import Dict, List, Optional, Tuple
import os
import time
import re
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from rapidfuzz.distance.LCSseq import distance as lcs_distance
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# Support running as a module or as a script
try:
    from .preprocess import (
        split_steps,
        build_exact_examples_from_teacher_math,
        make_synthetic_mgsm_items,
    )
except Exception:  # noqa: E722
    from preprocess import (  # type: ignore
        split_steps,
        build_exact_examples_from_teacher_math,
        make_synthetic_mgsm_items,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens=128, temperature=0.0, device="cpu"):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        start = time.time()
        if temperature and temperature > 0.0:
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        end = time.time()
        gen = tokenizer.batch_decode(out, skip_special_tokens=True)
        gen_tokens = (out.shape[1] - inputs.input_ids.shape[1]) * out.shape[0]
        toks_per_sec = gen_tokens / (end - start + 1e-9)
    return gen, toks_per_sec


def extract_final_answer(text: str) -> Optional[str]:
    m = re.search(r"Final Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().replace(",", "")
    return None


def lcs_token_f1(student_text: str, core_text: str) -> float:
    if not RAPIDFUZZ_AVAILABLE:
        s = student_text.split()
        c = core_text.split()
        s_set = set(s)
        c_set = set(c)
        overlap = len(s_set & c_set)
        prec = overlap / max(1, len(s_set))
        rec = overlap / max(1, len(c_set))
        return 2*prec*rec / max(1e-9, (prec+rec)) if (prec+rec) > 0 else 0.0
    s = student_text
    c = core_text
    dist = lcs_distance(s, c)
    overlap = (len(s) + len(c) - dist) / 2.0
    precision = overlap / max(1.0, float(len(s)))
    recall = overlap / max(1.0, float(len(c)))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _plot_line(values: List[float], title: str, filename: str):
    plt.figure(figsize=(5,3))
    sns.lineplot(x=np.arange(len(values)), y=values)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Training loss")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _plot_bar(values: Dict[str, float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(5,3))
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    sns.barplot(x=keys, y=vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _plot_conf_mat(y_true: List[str], y_pred: List[str], labels: List[str], title: str, filename: str):
    if not SKLEARN_AVAILABLE:
        print(f"[Plot] sklearn unavailable; skipping {filename}")
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ========== Experiment 1 (Math) evaluation ==========

def evaluate_experiment1_math(models: Dict[str, AutoModelForCausalLM], tokenizer: AutoTokenizer, val_items: List[Dict], images_dir: str, device: str = "cpu") -> Dict:
    def make_prompt(p: str):
        return f"Solve the problem with steps and finish with 'Final Answer: <number>'.\n{p}\n"

    prompts = [make_prompt(ex["problem"]) for ex in val_items]
    gold = [ex["teacher_answer"] for ex in val_items]

    # Build target core and rejected texts
    core_texts, rejected_texts = [], []
    for ex in val_items:
        core, rej = [], []
        core_idx = set(ex.get("core_indices", []))
        for i, s in enumerate(ex["steps"]):
            if i in core_idx:
                core.append(s)
            if ex["rejected_mask"][i] == 1:
                rej.append(s)
        core_texts.append("\n".join(core))
        rejected_texts.append("\n".join(rej))

    outs = {}
    lats = {}
    for name, model in models.items():
        gen, lat = generate_texts(model, tokenizer, prompts, device=device)
        outs[name] = gen
        lats[name] = lat

    def eval_math(outputs: List[str]) -> Tuple[float, float, float, float]:
        acc, f1s, eprs, lengths = [], [], [], []
        for i, out in enumerate(outputs):
            pred = extract_final_answer(out)
            acc.append(1 if pred == gold[i] else 0)
            f1s.append(lcs_token_f1(out, core_texts[i]))
            s_toks = out.split()
            rej_set = set(rejected_texts[i].split())
            overlap = sum(1 for t in s_toks if t in rej_set)
            eprs.append(overlap / max(1, len(s_toks)))
            lengths.append(len(tokenizer(out).input_ids))
        return float(np.mean(acc)), float(np.mean(f1s)), float(np.mean(eprs)), float(np.mean(lengths))

    results = {}
    for name in ["exact", "cot", "ans"]:
        acc, f1, epr, length = eval_math(outs[name])
        results[name] = {"acc": acc, "f1": f1, "epr": epr, "len": length, "lat": lats[name]}
        print(f"[Eval:Math] {name.upper():6s} => Acc={acc:.3f} FaithF1={f1:.3f} EPR={epr:.3f} AvgLen={int(length)} Lat(toks/s)={lats[name]:.1f}")

    # Plots
    ensure_dir(images_dir)
    _plot_bar({k.upper(): results[k]["acc"] for k in results}, "Accuracy (Exp1)", "Accuracy", os.path.join(images_dir, "accuracy_exp1.pdf"))
    _plot_bar({k.upper(): results[k]["f1"] for k in results}, "Faithfulness F1 (Exp1)", "F1", os.path.join(images_dir, "faithfulness_exp1.pdf"))
    _plot_bar({k.upper(): results[k]["epr"] for k in results}, "Error Propagation Rate (Exp1)", "EPR", os.path.join(images_dir, "epr_exp1.pdf"))
    _plot_bar({k.upper(): results[k]["len"] for k in results}, "Avg CoT length (tokens) (Exp1)", "Tokens", os.path.join(images_dir, "cot_length_exp1.pdf"))
    _plot_bar({k.upper(): results[k]["lat"] for k in results}, "Decoding latency (toks/s) (Exp1)", "toks/s", os.path.join(images_dir, "inference_latency_exp1.pdf"))

    return results


# ========== Experiment 2 (Commonsense) evaluation ==========

def evaluate_experiment2_commonsense(models: Dict[str, AutoModelForCausalLM], tokenizer: AutoTokenizer, val_items: List[Dict], verify_fn, images_dir: str, device: str = "cpu") -> Dict:
    def make_prompt_cs(p: str):
        return f"Answer the question by reasoning in steps and then state 'Final Answer: <A/B/C/D/E>'.\n{p}\n"

    prompts = [make_prompt_cs(ex["problem"]) for ex in val_items]
    gold = [ex["teacher_answer"] for ex in val_items]

    outs = {}
    lats = {}
    for name, model in models.items():
        gen, lat = generate_texts(model, tokenizer, prompts, device=device)
        outs[name] = gen
        lats[name] = lat

    def parse_letter(ans: Optional[str]) -> str:
        if not ans:
            return "?"
        m = re.search(r"([A-E])", ans.upper())
        return m.group(1) if m else "?"

    def eval_cs(outputs: List[str]) -> Tuple[float, List[str], List[str], float, float]:
        y_true, y_pred = [], []
        hallucination_rates, lengths = [], []
        for i, out in enumerate(outputs):
            pred = parse_letter(extract_final_answer(out))
            y_pred.append(pred)
            y_true.append(parse_letter(gold[i]))
            # Verify each generated step via verify_fn
            gen_steps = split_steps(out)
            not_ent = 0
            for s in gen_steps:
                res = verify_fn(val_items[i]["problem"], s)
                if not res.get("verified", False):
                    not_ent += 1
            denom = max(1, len(gen_steps))
            hallucination_rates.append(not_ent/denom)
            lengths.append(len(tokenizer(out).input_ids))
        acc = float(np.mean([1 if a==b else 0 for a,b in zip(y_true, y_pred)]))
        return acc, y_true, y_pred, float(np.mean(hallucination_rates)), float(np.mean(lengths))

    results = {}
    y_true_map = {}
    y_pred_map = {}
    for name in ["exact", "cot", "ans"]:
        acc, y_true, y_pred, hall, length = eval_cs(outs[name])
        results[name] = {"acc": acc, "hall": hall, "len": length, "lat": lats[name]}
        y_true_map[name] = y_true
        y_pred_map[name] = y_pred
        print(f"[Eval:CSQA] {name.upper():6s} => Acc={acc:.3f} Hallucination={hall:.3f} AvgLen={int(length)} Lat(toks/s)={lats[name]:.1f}")

    # Plots
    ensure_dir(images_dir)
    _plot_bar({k.upper(): results[k]["acc"] for k in results}, "Accuracy (Exp2)", "Accuracy", os.path.join(images_dir, "accuracy_exp2.pdf"))
    _plot_bar({k.upper(): results[k]["hall"] for k in results}, "Hallucination rate (Exp2)", "Rate", os.path.join(images_dir, "hallucination_exp2.pdf"))
    _plot_bar({k.upper(): results[k]["len"] for k in results}, "Avg CoT length (tokens) (Exp2)", "Tokens", os.path.join(images_dir, "cot_length_exp2.pdf"))
    labels = ["A","B","C","D","E","?"]
    _plot_conf_mat(y_true_map["exact"], y_pred_map["exact"], labels, "Confusion Matrix (EXACT, Exp2)", os.path.join(images_dir, "confusion_matrix_exp2_exact.pdf"))
    _plot_conf_mat(y_true_map["cot"], y_pred_map["cot"], labels, "Confusion Matrix (CoT, Exp2)", os.path.join(images_dir, "confusion_matrix_exp2_baseline.pdf"))

    return results


# ========== Experiment 3 (Cross-lingual) evaluation ==========

def verify_numeric_ops_in_text(text: str) -> float:
    eqs = re.findall(r"(.+?)\s*=\s*(.+)", text)
    if not eqs:
        return 0.0
    ok = 0
    total = 0
    for (L, R) in eqs:
        total += 1
        try:
            # naive numeric compare after eval; safe empty globals
            Lv = eval(L, {"__builtins__": {}}, {})
            Rv = eval(R, {"__builtins__": {}}, {})
            if abs(float(Lv) - float(Rv)) < 1e-6:
                ok += 1
        except Exception:
            pass
    return ok / max(1, total)


def evaluate_experiment3_mgsm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, images_dir: str, device: str = "cpu") -> Dict:
    items = make_synthetic_mgsm_items()

    def make_prompt(problem: str, minimal: bool = False):
        ctrl = "Use minimal steps.\n" if minimal else ""
        return ctrl + "Solve step by step and finish with 'Final Answer: <number>'.\n" + problem + "\n"

    acc_per_lang = {}
    len_per_lang = {}
    acc_min_per_lang = {}
    len_min_per_lang = {}
    faith_per_lang = {}

    for lang, lst in items.items():
        prompts = [make_prompt(x["q"], minimal=False) for x in lst]
        outs, _ = generate_texts(model, tokenizer, prompts, device=device)
        prompts_min = [make_prompt(x["q"], minimal=True) for x in lst]
        outs_min, _ = generate_texts(model, tokenizer, prompts_min, device=device)
        acc = []
        acc_min = []
        lens = []
        lens_min = []
        faiths = []
        for x, o, o2 in zip(lst, outs, outs_min):
            pa = extract_final_answer(o)
            pa2 = extract_final_answer(o2)
            acc.append(1 if (pa == x["a"]) else 0)
            acc_min.append(1 if (pa2 == x["a"]) else 0)
            lens.append(len(tokenizer(o).input_ids))
            lens_min.append(len(tokenizer(o2).input_ids))
            faiths.append(verify_numeric_ops_in_text(o))
        acc_per_lang[lang] = float(np.mean(acc))
        len_per_lang[lang] = float(np.mean(lens))
        acc_min_per_lang[lang] = float(np.mean(acc_min))
        len_min_per_lang[lang] = float(np.mean(lens_min))
        faith_per_lang[lang] = float(np.mean(faiths))
        print(f"[Eval:MGSM] {lang} => Acc={acc_per_lang[lang]:.3f} Acc(min)={acc_min_per_lang[lang]:.3f} Len={len_per_lang[lang]:.1f} Len(min)={len_min_per_lang[lang]:.1f} FaithNum={faith_per_lang[lang]:.2f}")

    # Plots
    ensure_dir(images_dir)
    _plot_bar(acc_per_lang, "Accuracy by language (Exp3)", "Accuracy", os.path.join(images_dir, "accuracy_exp3.pdf"))
    # Grouped length bars
    plt.figure(figsize=(6,3))
    langs = list(acc_per_lang.keys())
    x = np.arange(len(langs))
    w = 0.35
    plt.bar(x - w/2, [len_per_lang[l] for l in langs], width=w, label="rationale")
    plt.bar(x + w/2, [len_min_per_lang[l] for l in langs], width=w, label="compressed")
    plt.xticks(x, langs)
    plt.ylabel("Avg tokens")
    plt.title("Output length by language (Exp3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "cot_length_exp3.pdf"), bbox_inches="tight")
    plt.close()

    return {
        "acc": acc_per_lang,
        "acc_min": acc_min_per_lang,
        "len": len_per_lang,
        "len_min": len_min_per_lang,
        "faith_num": faith_per_lang,
    }
