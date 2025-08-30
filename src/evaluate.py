import os
import json
import time
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import (
    Problem,
    Tools,
    SimplePRM,
    SyntheticBackbone,
    Action,
    DEFAULT_COSTS,
    EpisodeResult,
    StudentController,
    run_episode,
)


def serialize_episode(ep: EpisodeResult) -> Dict[str, Any]:
    return {
        "pid": ep.pid,
        "correct": int(ep.correct),
        "flops": float(ep.flops),
        "latency_ms_sim": float(ep.latency_ms_sim),
        "energy_J": float(ep.energy_J),
        "actions": ep.actions,
        "prm_scores": [float(x) for x in ep.prm_scores],
    }


# ------------------------------
# Experiment 1: End-to-end eval
# ------------------------------

def experiment1_end_to_end(
    problems: List[Problem],
    budgets: List[float],
    student_controller: StudentController,
    images_dir: str,
    save_json_path: str,
) -> Dict[str, Any]:
    sns.set_theme(style="whitegrid")
    tools = Tools()
    backbone = SyntheticBackbone()
    prm = SimplePRM()

    systems = {
        "BACS": student_controller,
        "FixedCascade": _FixedCascadeWrapper(),
        "Beam5": _StaticBeamWrapper(),
        "StaticCoT13B": _StaticCoTWrapper(),
    }

    results = {name: {cap: [] for cap in budgets} for name in systems}

    for cap in budgets:
        print(f"\n-- Budget cap: {cap:.0f} FLOPs --")
        for name, sys in systems.items():
            print(f"Running system: {name}")
            start = time.perf_counter()
            for pb in problems:
                if name == "BACS":
                    decide_fn = lambda p, t, ps, f: student_controller.decide(p, t, ps, f, lam=1.0)
                else:
                    decide_fn = sys.decide
                ep = run_episode(pb, backbone, prm, decide_fn, DEFAULT_COSTS, cap, tools)
                results[name][cap].append(ep)
            dur = time.perf_counter() - start
            avg_acc = np.mean([r.correct for r in results[name][cap]])
            avg_flops = np.mean([r.flops for r in results[name][cap]])
            p95_latency = np.percentile([r.latency_ms_sim for r in results[name][cap]], 95)
            overruns = np.mean([r.flops > cap for r in results[name][cap]])
            print(
                f"{name}: N={len(problems)} acc={avg_acc:.3f} avgFLOPs={avg_flops:.2e} "
                f"P95lat(ms)={p95_latency:.1f} overruns={overruns:.3f} wall_time={dur:.2f}s"
            )

    # Save per-episode results as JSON
    out_json = {}
    for name in systems:
        out_json[name] = {}
        for cap in budgets:
            out_json[name][str(int(cap))] = [serialize_episode(ep) for ep in results[name][cap]]
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f)

    # Plot: Accuracy vs average FLOPs
    fig1 = plt.figure(figsize=(6, 4))
    for name in systems:
        xs, ys = [], []
        for cap in budgets:
            R = results[name][cap]
            xs.append(np.mean([r.flops for r in R]))
            ys.append(np.mean([r.correct for r in R]))
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Average FLOPs per instance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Compute (BACS vs baselines)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "accuracy_budgets_bacs_vs_baselines.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # Plot: P95 latency vs budget
    fig2 = plt.figure(figsize=(6, 4))
    for name in systems:
        xs = budgets
        ys = [np.percentile([r.latency_ms_sim for r in results[name][cap]], 95) for cap in budgets]
        plt.plot(xs, ys, marker="s", label=name)
    plt.xlabel("FLOP cap")
    plt.ylabel("P95 simulated latency (ms)")
    plt.title("P95 latency vs budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "inference_latency_bacs_vs_baselines.pdf"), bbox_inches="tight")
    plt.close(fig2)

    return results


# --------------------------------------------------
# Experiment 2: Causal ablation and compute-matched
# --------------------------------------------------

def experiment2_causal_ablation(
    problems: List[Problem],
    ref_logs: Dict[str, EpisodeResult],
    student_controller: StudentController,
    images_dir: str,
    save_json_path: str,
) -> Dict[str, List[EpisodeResult]]:
    sns.set_theme(style="whitegrid")
    tools = Tools()
    backbone = SyntheticBackbone()
    prm = SimplePRM()

    def run_compute_matched(problem: Problem, reference: EpisodeResult, decide_fn):
        cap = reference.flops
        return run_episode(problem, backbone, prm, decide_fn, DEFAULT_COSTS, cap, tools)

    def decide_full(p, t, ps, f):
        return student_controller.decide(p, t, ps, f, lam=1.0)

    def decide_no_verifier(p, t, ps, f):
        a, conf, logits = student_controller.decide(p, t, ps, f, lam=1.0)
        if a == Action.CALL_VER:
            return Action.ADVANCE, conf * 0.9, logits
        return a, conf, logits

    def decide_no_calculator(p, t, ps, f):
        a, conf, logits = student_controller.decide(p, t, ps, f, lam=1.0)
        if a == Action.CALL_CALC:
            return Action.ADVANCE, conf * 0.9, logits
        return a, conf, logits

    def decide_shuffled_top2(p, t, ps, f):
        a, conf, logits = student_controller.decide(p, t, ps, f, lam=1.0)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        top2 = np.argsort(probs)[-2:]
        choose = int(np.random.choice(top2))
        return ALL_ACTIONS[choose], float(probs[choose]), logits

    def decide_fixed_depth(p, t, ps, f):
        step_idx = f.get("step_idx", 0)
        if step_idx < 4:
            return Action.ADVANCE, 1.0, None
        return Action.TERMINATE, 1.0, None

    conditions = {
        "FullBACS": decide_full,
        "NoVerifier": decide_no_verifier,
        "NoCalculator": decide_no_calculator,
        "ShuffledTop2": decide_shuffled_top2,
        "FixedDepth": decide_fixed_depth,
    }

    outs: Dict[str, List[EpisodeResult]] = {k: [] for k in conditions}

    for pb in problems:
        ref = ref_logs[pb.pid]
        for cname, decide_fn in conditions.items():
            out = run_compute_matched(pb, ref, decide_fn)
            outs[cname].append(out)

    # Summaries
    for cname in conditions:
        acc = np.mean([o.correct for o in outs[cname]])
        p95 = np.percentile([o.latency_ms_sim for o in outs[cname]], 95)
        print(f"{cname}: accuracy={acc:.3f} P95lat(ms)={p95:.1f} (compute-matched to FullBACS)")

    trap_ids = [pb.pid for pb in problems if pb.trap]

    def subset(arr: List[EpisodeResult], ids: List[str]):
        m = {x.pid: x for x in arr}
        return [m[i] for i in ids if i in m]

    full_trap = subset(outs["FullBACS"], trap_ids)
    fixed_trap = subset(outs["FixedDepth"], trap_ids)

    early_term_rate_full = np.mean([1.0 if ("terminate" in " ".join(x.actions).lower()) else 0.0 for x in full_trap])
    early_term_rate_fixed = np.mean([1.0 if ("terminate" in " ".join(x.actions).lower()) else 0.0 for x in fixed_trap])
    acc_full_trap = np.mean([x.correct for x in full_trap])
    acc_fixed_trap = np.mean([x.correct for x in fixed_trap])
    print(
        f"Trap-set: FullBACS early-term rate={early_term_rate_full:.3f}, acc={acc_full_trap:.3f}; "
        f"FixedDepth early-term rate={early_term_rate_fixed:.3f}, acc={acc_fixed_trap:.3f}"
    )

    # Save results JSON
    out_json = {k: [serialize_episode(ep) for ep in v] for k, v in outs.items()}
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f)

    # Plot pair: accuracy per condition
    figA = plt.figure(figsize=(6, 4))
    names = list(conditions.keys())
    vals = [np.mean([o.correct for o in outs[n]]) for n in names]
    sns.barplot(x=names, y=vals, color="#4C78A8")
    plt.ylabel("Accuracy")
    plt.title("Ablation accuracy (compute-matched)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "accuracy_ablation_pair1.pdf"), bbox_inches="tight")
    plt.close(figA)

    # Early termination on trap set
    figB = plt.figure(figsize=(6, 4))
    names2 = ["FullBACS", "FixedDepth"]
    vals2 = [early_term_rate_full, early_term_rate_fixed]
    sns.barplot(x=names2, y=vals2, color="#F58518")
    plt.ylabel("Early termination rate (trap)")
    plt.title("Trap-set early termination")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "early_termination_trap_pair2.pdf"), bbox_inches="tight")
    plt.close(figB)

    return outs


# -----------------
# Helper wrappers
# -----------------
class _FixedCascadeWrapper:
    def __init__(self):
        self._delegate = None

    def decide(self, problem, trace, prm_scores, features):
        # Lazy import from train to avoid circular import at module import time
        from .train import FixedCascade
        if self._delegate is None:
            self._delegate = FixedCascade()
        return self._delegate.decide(problem, trace, prm_scores, features)


class _StaticBeamWrapper:
    def __init__(self):
        self._delegate = None

    def decide(self, problem, trace, prm_scores, features):
        from .train import StaticBeam
        if self._delegate is None:
            self._delegate = StaticBeam()
        return self._delegate.decide(problem, trace, prm_scores, features)


class _StaticCoTWrapper:
    def __init__(self):
        self._delegate = None

    def decide(self, problem, trace, prm_scores, features):
        from .train import StaticCoT
        if self._delegate is None:
            self._delegate = StaticCoT()
        return self._delegate.decide(problem, trace, prm_scores, features)
