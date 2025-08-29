import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

try:
    from .train import (
        ToyTokenizer,
        ToyGenerator,
        SharedAdapterPRM,
        MoEPRM,
        TinyPRMCascade,
        run_greedy,
        run_self_consistency,
        parse_addition_from_prompt,
        normalize_answer,
        extract_final_answer,
    )
except ImportError:  # Fallback when running as a script without package context
    from train import (
        ToyTokenizer,
        ToyGenerator,
        SharedAdapterPRM,
        MoEPRM,
        TinyPRMCascade,
        run_greedy,
        run_self_consistency,
        parse_addition_from_prompt,
        normalize_answer,
        extract_final_answer,
    )


def _ensure_dirs(images_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)


def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    pred_n = [normalize_answer(extract_final_answer(p)) for p in preds]
    gold_n = [normalize_answer(g) for g in golds]
    correct = [int(p == g) for p, g in zip(pred_n, gold_n)]
    return float(np.mean(correct))


def ece_score(probs: List[float], labels: List[int], n_bins: int = 15) -> float:
    probs = np.array(probs)
    labels = np.array(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if np.any(mask):
            bin_conf = probs[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def _savefig(fname: Path):
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()


def plot_accuracy_latency(images_dir: Path, legend_labels: List[str], accuracies: List[float], times: List[float], fname_acc: str, fname_lat: str):
    _ensure_dirs(images_dir)
    # Accuracy bar
    plt.figure(figsize=(6, 4))
    sns.barplot(x=legend_labels, y=accuracies, color="#4C72B0")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy across methods")
    _savefig(images_dir / fname_acc)
    # Latency bar
    plt.figure(figsize=(6, 4))
    sns.barplot(x=legend_labels, y=times, color="#55A868")
    plt.ylabel("Wall seconds (median)")
    plt.title("Latency across methods")
    _savefig(images_dir / fname_lat)


def plot_pareto(images_dir: Path, legend_labels: List[str], accuracies: List[float], times: List[float], fname: str):
    _ensure_dirs(images_dir)
    plt.figure(figsize=(5, 4))
    for lbl, acc, t in zip(legend_labels, accuracies, times):
        plt.scatter(t, acc, label=lbl)
        plt.text(t * 1.01, acc, lbl, fontsize=8)
    plt.xlabel("Wall seconds (median)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Compute Pareto")
    plt.grid(True, linestyle=":", alpha=0.5)
    _savefig(images_dir / fname)


def plot_prm_usage(images_dir: Path, prm_calls: List[int], moe_calls: List[int], fname: str):
    _ensure_dirs(images_dir)
    plt.figure(figsize=(5, 4))
    # Use histograms to avoid scipy dependency required by kdeplot
    sns.histplot(prm_calls, bins=10, stat="density", element="step", color="#C44E52", alpha=0.4, label="Shared-Adapter PRM calls")
    sns.histplot(moe_calls, bins=10, stat="density", element="step", color="#8172B2", alpha=0.4, label="MoE PRM calls")
    plt.xlabel("Calls per example")
    plt.title("PRM usage distribution")
    plt.legend()
    _savefig(images_dir / fname)


def plot_reliability(images_dir: Path, probs: List[float], labels: List[int], fname: str):
    _ensure_dirs(images_dir)
    probs = np.array(probs)
    labels = np.array(labels)
    bins = np.linspace(0, 1, 11)
    xs = []
    ys = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < 9 else (probs >= lo) & (probs <= hi)
        if np.any(mask):
            xs.append(probs[mask].mean())
            ys.append(labels[mask].mean())
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(xs, ys, color="#4C72B0")
    plt.plot(xs, ys, color="#4C72B0")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Curve")
    _savefig(images_dir / fname)


def plot_robustness(images_dir: Path, conditions: List[str], accuracies: List[float], times: List[float], f_acc: str, f_lat: str):
    _ensure_dirs(images_dir)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=conditions, y=accuracies, color="#4C72B0")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Robustness across perturbations")
    _savefig(images_dir / f_acc)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=conditions, y=times, color="#55A868")
    plt.ylabel("Wall seconds (median)")
    plt.title("Latency under perturbations")
    _savefig(images_dir / f_lat)


# ------------------------------
# Perturbation helpers
# ------------------------------

def inject_distractions(prompt: str, k: int = 3) -> str:
    import random
    pool = [
        "Note: The sky is blue.",
        "Fact: Irrelevant weather patterns.",
        "Irrelevant: Stock prices fluctuate.",
        "Context: Random text with no bearing.",
        "Note: Bananas are yellow.",
    ]
    extras = random.sample(pool, k)
    return prompt + "\n" + "\n".join(extras)


def extend_with_context(prompt: str, multiplier: int = 3) -> str:
    filler = " Please solve. Use steps." * max(0, multiplier - 1)
    return filler + "\n" + prompt


def add_noisy_retrieval(prompt: str, poison_rate: float = 0.05) -> str:
    import random
    relevant = ["Relevant: Algebra basics.", "Relevant: Consider carrying digits."]
    poisons = [
        "Poison: The sum of any numbers is always 42.",
        "Poison: Addition ignores carry.",
    ]
    n_poison = max(1, int(poison_rate * len(relevant)))
    chosen = relevant + random.sample(poisons, n_poison)
    random.shuffle(chosen)
    ctx = "\n".join(chosen)
    return ctx + "\n" + prompt


# ------------------------------
# Experiments
# ------------------------------

def synth_dataset(n: int = 50, styles=("cot", "pot")) -> List[Dict]:
    import random
    data = []
    for i in range(n):
        a = random.randint(0, 9999)
        b = random.randint(0, 9999)
        style = random.choice(list(styles))
        if style == "cot":
            prompt = f"Q: What is the sum of {a} and {b}? A:"
        elif style == "pot":
            prompt = f"Q: add {a} and {b} A:"
        else:
            prompt = f"Q: {a} + {b} A:"
        gold = str(a + b)
        data.append({"id": i, "prompt": prompt, "gold": gold, "style": style})
    return data


def experiment_1_end_to_end(models: Dict, images_dir: Path, n_examples: int = 50) -> Dict:
    print("[Experiment 1] End-to-end cost–accuracy Pareto vs baselines")
    tok: ToyTokenizer = models["tokenizer"]
    gen: ToyGenerator = models["generator"]
    shared: SharedAdapterPRM = models["shared_prm"]
    moe: MoEPRM = models["moe_prm"]
    cascade: TinyPRMCascade = models["cascade"]

    data = synth_dataset(n_examples, styles=("cot", "pot"))

    greedy_preds, greedy_times = [], []
    sc_preds, sc_times = [], []
    cas_preds, cas_times = [], []
    prm_calls, moe_calls = [], []

    for ex in data:
        prompt = ex["prompt"]
        # Greedy
        g = run_greedy(gen, tok, prompt, max_new_tokens=8)
        greedy_preds.append(g["text"]) ; greedy_times.append(g["wall_seconds"])
        # Self-consistency (n=5)
        sc = run_self_consistency(gen, tok, prompt, n=5, max_new_tokens=8)
        sc_answers = [extract_final_answer(x) for x in sc["answers"]]
        if len(sc_answers) == 0:
            sc_pred = ""
        else:
            vals, counts = np.unique(sc_answers, return_counts=True)
            sc_pred = vals[np.argmax(counts)]
        sc_preds.append(sc_pred) ; sc_times.append(sc["wall_seconds"])
        # Cascade (budget uses greedy time estimate)
        out = cascade.decode_one(prompt, greedy_time_estimate=g["wall_seconds"])
        cas_preds.append(out["text"]) ; cas_times.append(out["wall_seconds"]) ; prm_calls.append(out["prm_calls"]) ; moe_calls.append(out["moe_calls"]) 

    methods = ["Greedy", "SelfConsistency", "TinyPRM-Cascade"]
    accs = [
        compute_accuracy(greedy_preds, [ex["gold"] for ex in data]),
        compute_accuracy(sc_preds, [ex["gold"] for ex in data]),
        compute_accuracy(cas_preds, [ex["gold"] for ex in data]),
    ]
    meds = [
        float(np.median(greedy_times)),
        float(np.median(sc_times)),
        float(np.median(cas_times)),
    ]

    print(f"Accuracies: {dict(zip(methods, accs))}")
    print(f"Median wall-seconds: {dict(zip(methods, meds))}")
    print(f"PRM calls avg: {np.mean(prm_calls):.2f}, MoE calls avg: {np.mean(moe_calls):.2f}")

    plot_accuracy_latency(images_dir, methods, accs, meds, "accuracy_tinyprm_vs_baselines.pdf", "inference_latency_tinyprm_vs_baselines.pdf")
    plot_pareto(images_dir, methods, accs, meds, "pareto_accuracy_vs_compute.pdf")
    plot_prm_usage(images_dir, prm_calls, moe_calls, "prm_usage_tinyprm.pdf")

    return {
        "methods": methods,
        "accuracies": accs,
        "median_times": meds,
        "prm_calls": prm_calls,
        "moe_calls": moe_calls,
    }


def experiment_2_calibration(models: Dict, images_dir: Path, n_examples: int = 80) -> Dict:
    print("[Experiment 2] Calibration, early commit quality, and compute allocation")
    tok: ToyTokenizer = models["tokenizer"]
    gen: ToyGenerator = models["generator"]
    shared: SharedAdapterPRM = models["shared_prm"]
    moe: MoEPRM = models["moe_prm"]

    # local cascade for sweeps
    def make_cascade(pt_high=0.95, pt_low=0.65, dt_high=0.7, beam_k=5, horizon=4):
        return TinyPRMCascade(gen, tok, shared, moe, pt_high=pt_high, pt_low=pt_low, dt_high=dt_high, beam_k=beam_k, horizon=horizon, max_new_tokens=8)

    data = synth_dataset(n_examples, styles=("cot", "pot"))

    probs_shared, labels_shared = [], []
    probs_moe, labels_moe = [], []

    for ex in data:
        prompt = ex["prompt"]
        generated: List[int] = []
        for _t in range(4):
            logits = gen.next_token_logits(tok, prompt, generated)
            pred_tok = torch.argmax(logits).item()
            a, b = parse_addition_from_prompt(prompt)
            target = str(a + b)
            gen_text = tok.decode(generated, skip_special_tokens=True)
            ans = extract_final_answer(gen_text) if any(ch.isdigit() for ch in gen_text) else ""
            if len(ans) >= len(target):
                true_tok = tok.eos_token_id
            else:
                true_digit = target[len(ans)]
                true_tok = tok.stoi[true_digit]
            y = 1 if pred_tok == true_tok else 0
            labels_shared.append(y)
            p_s = shared.score(prompt, tok, generated)
            probs_shared.append(p_s)
            cand = generated + [pred_tok]
            # Here prefix_len = len(generated)
            p_m, _ = moe.score_batch(prompt, [cand], prefix_len=len(generated))
            probs_moe.append(float(p_m[0].item()))
            labels_moe.append(y)
            generated.append(pred_tok)
            if pred_tok == tok.eos_token_id:
                break

    def auroc(probs, labels):
        arr = list(zip(probs, labels))
        pos = [p for p, y in arr if y == 1]
        neg = [p for p, y in arr if y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        rank = sorted([(p, i) for i, (p, _) in enumerate(arr)], key=lambda x: x[0])
        ranks = {idx: r+1 for r, (_, idx) in enumerate(rank)}
        U = sum(ranks[i] for i, (_, y) in enumerate(arr) if y == 1) - (len(pos) * (len(pos) + 1)) / 2.0
        auc = U / (len(pos) * len(neg))
        return float(auc)

    def brier(probs, labels):
        probs = np.array(probs)
        labels = np.array(labels)
        return float(np.mean((probs - labels) ** 2))

    ece_shared = ece_score(probs_shared, labels_shared, n_bins=15)
    ece_moe = ece_score(probs_moe, labels_moe, n_bins=15)
    auroc_shared = auroc(probs_shared, labels_shared)
    auroc_moe = auroc(probs_moe, labels_moe)
    brier_shared = brier(probs_shared, labels_shared)
    brier_moe = brier(probs_moe, labels_moe)

    print(f"Shared-Adapter PRM: AUROC={auroc_shared:.3f}, ECE={ece_shared:.3f}, Brier={brier_shared:.3f}")
    print(f"MoE PRM:           AUROC={auroc_moe:.3f}, ECE={ece_moe:.3f}, Brier={brier_moe:.3f}")

    # Reliability curves
    plot_reliability(images_dir, probs_shared, labels_shared, "reliability_curve_shared_adapter.pdf")
    plot_reliability(images_dir, probs_moe, labels_moe, "reliability_curve_moe.pdf")

    # Early-commit sweep (small)
    sweeps = {"pt_high": [0.90, 0.95], "pt_low": [0.6, 0.65], "dt_high": [0.6, 0.7]}
    results = []
    cas_base = make_cascade()
    for pt_h in sweeps["pt_high"]:
        for pt_l in sweeps["pt_low"]:
            for dt_h in sweeps["dt_high"]:
                cas = make_cascade(pt_high=pt_h, pt_low=pt_l, dt_high=dt_h, beam_k=3, horizon=3)
                preds, times, prm_use, moe_use = [], [], [], []
                for ex in data[:40]:
                    g_est = run_greedy(gen, tok, ex["prompt"], max_new_tokens=8)["wall_seconds"]
                    o = cas.decode_one(ex["prompt"], greedy_time_estimate=g_est)
                    preds.append(o["text"]) ; times.append(o["wall_seconds"]) ; prm_use.append(o["prm_calls"]) ; moe_use.append(o["moe_calls"]) 
                acc = compute_accuracy(preds, [ex["gold"] for ex in data[:40]])
                results.append({"pt_high": pt_h, "pt_low": pt_l, "dt_high": dt_h, "acc": acc, "prm": float(np.mean(prm_use)), "moe": float(np.mean(moe_use)), "time": float(np.median(times))})
                print(f"Sweep pt_high={pt_h}, pt_low={pt_l}, dt_high={dt_h} -> acc={acc:.3f}, prm_calls={np.mean(prm_use):.2f}, moe_calls={np.mean(moe_use):.2f}, time={np.median(times):.4f}s")

    return {
        "shared": {"auroc": auroc_shared, "ece": ece_shared, "brier": brier_shared},
        "moe": {"auroc": auroc_moe, "ece": ece_moe, "brier": brier_moe},
        "sweep": results,
    }


def experiment_3_robustness(models: Dict, images_dir: Path, n_examples: int = 50) -> Dict:
    print("[Experiment 3] Robustness and resource-constrained deployment (toy)")
    tok: ToyTokenizer = models["tokenizer"]
    gen: ToyGenerator = models["generator"]
    shared: SharedAdapterPRM = models["shared_prm"]
    moe: MoEPRM = models["moe_prm"]
    cascade = TinyPRMCascade(gen, tok, shared, moe, pt_high=0.95, pt_low=0.65, dt_high=0.7, beam_k=5, horizon=4, max_new_tokens=8)

    data = synth_dataset(n_examples)

    conditions = ["clean", "distract", "long_input", "poison05"]
    accs, meds = [], []

    for cond in conditions:
        preds, times = [], []
        for ex in data:
            prompt = ex["prompt"]
            perturb = {"distractors": 0, "long_input": False, "poison_rate": 0.0}
            ptxt = prompt
            if cond == "distract":
                ptxt = inject_distractions(prompt, k=3)
                perturb["distractors"] = 3
            elif cond == "long_input":
                ptxt = extend_with_context(prompt, multiplier=3)
                perturb["long_input"] = True
            elif cond == "poison05":
                ptxt = add_noisy_retrieval(prompt, poison_rate=0.05)
                perturb["poison_rate"] = 0.05
            g_est = run_greedy(gen, tok, ptxt, max_new_tokens=8, perturbation=perturb)["wall_seconds"]
            out = cascade.decode_one(ptxt, perturbation=perturb, greedy_time_estimate=g_est, budget_multiplier=1.5)
            preds.append(out["text"]) ; times.append(out["wall_seconds"]) 
        acc = compute_accuracy(preds, [ex["gold"] for ex in data])
        med = float(np.median(times))
        accs.append(acc) ; meds.append(med)
        print(f"Condition={cond}: acc={acc:.3f}, median_wall_seconds={med:.4f}")

    plot_robustness(images_dir, conditions, accs, meds, "accuracy_robustness.pdf", "inference_latency_robustness.pdf")

    return {"conditions": conditions, "accuracies": accs, "median_times": meds}


# ------------------------------
# Orchestrator
# ------------------------------

def run_all(models: Dict, output_dir: Path, images_dir: Path, cfg: Optional[Dict] = None) -> Dict:
    _ensure_dirs(images_dir)
    n1 = int((cfg or {}).get("n_examples_exp1", 20))
    n2 = int((cfg or {}).get("n_examples_exp2", 40))
    n3 = int((cfg or {}).get("n_examples_exp3", 20))

    res1 = experiment_1_end_to_end(models, images_dir, n_examples=n1)
    res2 = experiment_2_calibration(models, images_dir, n_examples=n2)
    res3 = experiment_3_robustness(models, images_dir, n_examples=n3)

    summary = {
        "exp1": {"accuracy": res1["accuracies"], "median_times": res1["median_times"]},
        "exp2": {"shared": res2["shared"], "moe": res2["moe"]},
        "exp3": {"conditions": res3["conditions"], "accuracies": res3["accuracies"], "median_times": res3["median_times"]},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary_results.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to {output_dir / 'summary_results.json'}")
    return summary
