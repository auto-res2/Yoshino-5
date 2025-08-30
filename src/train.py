import os
import json
import time
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sympy import sympify

# -----------------------------
# Determinism and small helpers
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# --------------
# Core constants
# --------------
class Action(str, Enum):
    ADVANCE = "advance_one_step"
    BRANCH = "branch_beam"
    CALL_CALC = "call_calculator"
    CALL_PY = "call_python"
    CALL_RET = "call_retrieval"
    CALL_VER = "call_verifier"
    SKIP_FINAL = "skip_to_final"
    TERMINATE = "terminate"


ALL_ACTIONS = [
    Action.ADVANCE,
    Action.BRANCH,
    Action.CALL_CALC,
    Action.CALL_PY,
    Action.CALL_RET,
    Action.CALL_VER,
    Action.SKIP_FINAL,
    Action.TERMINATE,
]


@dataclass
class Costs:
    flops: Dict[Action, float]
    ms: Dict[Action, float]


DEFAULT_COSTS = Costs(
    flops={
        Action.ADVANCE: 8e8,
        Action.BRANCH: 1.5e9,
        Action.CALL_CALC: 2e8,
        Action.CALL_PY: 6e8,
        Action.CALL_RET: 5e8,
        Action.CALL_VER: 3e8,
        Action.SKIP_FINAL: 1e8,
        Action.TERMINATE: 5e7,
    },
    ms={
        Action.ADVANCE: 20.0,
        Action.BRANCH: 38.0,
        Action.CALL_CALC: 5.0,
        Action.CALL_PY: 12.0,
        Action.CALL_RET: 10.0,
        Action.CALL_VER: 7.0,
        Action.SKIP_FINAL: 2.0,
        Action.TERMINATE: 1.0,
    },
)

# Energy model (illustrative)
FLOPS_PER_JOULE = 5e10


# ------------------------
# Data structures & utils
# ------------------------
@dataclass
class Problem:
    pid: str
    question: str
    expression: str
    answer: float
    difficulty: str  # easy | medium | hard
    trap: bool


def safe_eval_expr(expr: str) -> float:
    try:
        return float(sympify(expr))
    except Exception:
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return float("nan")


# ---------------------------
# Tools and PRM abstractions
# ---------------------------
class Tools:
    def calculator(self, expr: str) -> str:
        val = safe_eval_expr(expr)
        return f"CALC_RESULT={val}"

    def python_exec(self, code: str) -> str:
        loc = {}
        try:
            exec(code, {"__builtins__": {}}, loc)
            val = loc.get("result", None)
            return f"PY_RESULT={val}"
        except Exception as e:
            return f"PY_ERROR={e}"

    def retrieval(self, query: str) -> str:
        if "answer is" in query:
            return "DOC: The exact answer is given above."
        hints = [
            "DOC: remember PEMDAS.",
            "DOC: addition and subtraction are inverse.",
            "DOC: compute multiplication before addition.",
            "DOC: check for rounding errors.",
        ]
        return random.choice(hints)

    def verifier(self, answer_text: str, gold: float) -> str:
        try:
            pred = float(answer_text)
            ok = abs(pred - gold) <= 1e-6
            return f"VERDICT={'CORRECT' if ok else 'INCORRECT'}"
        except Exception:
            return "VERDICT=UNKNOWN"


class SimplePRM:
    def score_step(self, step_text: str, gold: float) -> float:
        g = self._extract_guess(step_text)
        if g is None:
            return -1.0
        err = abs(g - gold)
        return -math.log1p(err)

    @staticmethod
    def _extract_guess(text: str):
        import re
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except Exception:
            return None


# -------------------------
# Backbone simulator (LLM)
# -------------------------
class SyntheticBackbone:
    def __init__(self):
        self.reset()

    def reset(self):
        self.used_tools = set()

    @staticmethod
    def _current_guess(trace_tokens: List[str]) -> float:
        txt = " ".join(trace_tokens)
        g = SimplePRM._extract_guess(txt)
        return float("nan") if g is None else g

    def generate_next_step(self, problem: Problem, trace_tokens: List[str]) -> Tuple[str, float]:
        prev_guess = self._current_guess(trace_tokens)
        if math.isnan(prev_guess):
            prev_guess = 0.0
        alpha = {"easy": 0.8, "medium": 0.5, "hard": 0.3}[problem.difficulty]
        noise = np.random.normal(0, 0.1 if problem.difficulty == "easy" else (0.5 if problem.difficulty == "medium" else 1.0))
        new_guess = prev_guess + alpha * (problem.answer - prev_guess) + noise
        step = f"Step: refine guess from {prev_guess:.3f} -> {new_guess:.3f}"
        step_logp = -abs(new_guess - problem.answer)
        return step, float(step_logp)

    def branch_beam(self, problem: Problem, trace_tokens: List[str], k: int = 2) -> List[str]:
        return [self.generate_next_step(problem, trace_tokens)[0] for _ in range(k)]

    @staticmethod
    def extract_expr(problem: Problem, trace_tokens: List[str]) -> str:
        return problem.expression

    @staticmethod
    def extract_code(problem: Problem, trace_tokens: List[str]) -> str:
        return f"result = {problem.expression}"

    @staticmethod
    def extract_query(problem: Problem, trace_tokens: List[str]) -> str:
        return problem.question

    @staticmethod
    def format_tool_result(out_text: str) -> str:
        return f"TOOL>> {out_text}"

    @staticmethod
    def format_docs(docs: str) -> str:
        return f"RETRIEVAL>> {docs}"

    @staticmethod
    def format_verdict(verdict: str) -> str:
        return f"VERIFIER>> {verdict}"

    @staticmethod
    def provisional_answer(trace_tokens: List[str]) -> str:
        g = SimplePRM._extract_guess(" ".join(trace_tokens))
        return "" if g is None else f"{g}"

    def finalize(self, problem: Problem, trace_tokens: List[str]) -> str:
        text = " ".join(trace_tokens)
        if "CALC_RESULT=" in text:
            try:
                return text.split("CALC_RESULT=")[-1].split()[0]
            except Exception:
                pass
        g = SimplePRM._extract_guess(text)
        if g is None:
            return f"{safe_eval_expr(problem.expression)}"
        return f"{g}"

    @staticmethod
    def checker(problem: Problem, ans_text: str) -> bool:
        try:
            pred = float(ans_text)
            return abs(pred - problem.answer) <= 1e-6
        except Exception:
            return False


# -----------------------------
# Controllers: teacher/student
# -----------------------------
class HeuristicTeacher:
    def __init__(self):
        self.name = "teacher"

    def decide(self, problem: Problem, trace: List[str], prm_scores: List[float], features: Dict[str, Any]) -> Tuple[Action, float, np.ndarray]:
        step_idx = features.get("step_idx", 0)
        budget_rem = features.get("budget_remaining", 0.0)
        mean_prm = features.get("mean_prm", 0.0)
        last_slope = 0.0 if len(prm_scores) < 2 else (prm_scores[-1] - prm_scores[-2])
        logits = np.full(len(ALL_ACTIONS), -5.0, dtype=np.float32)

        if budget_rem < DEFAULT_COSTS.flops[Action.ADVANCE] * 1.1:
            logits[ALL_ACTIONS.index(Action.TERMINATE)] = 5.0
            return Action.TERMINATE, 0.95, logits

        if problem.trap and step_idx >= 1:
            logits[ALL_ACTIONS.index(Action.TERMINATE)] = 4.0
            logits[ALL_ACTIONS.index(Action.SKIP_FINAL)] = 3.0
            logits[ALL_ACTIONS.index(Action.ADVANCE)] = -2.0
            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
            aidx = int(np.argmax(probs))
            return ALL_ACTIONS[aidx], float(np.max(probs)), logits

        if step_idx == 0:
            if problem.difficulty in ("medium", "hard"):
                logits[ALL_ACTIONS.index(Action.CALL_RET)] = 2.0
                logits[ALL_ACTIONS.index(Action.CALL_CALC)] = 2.5
                logits[ALL_ACTIONS.index(Action.ADVANCE)] = 3.0
            else:
                logits[ALL_ACTIONS.index(Action.ADVANCE)] = 3.5
                logits[ALL_ACTIONS.index(Action.CALL_CALC)] = 1.0
        else:
            if last_slope < -0.05 and step_idx >= 2:
                logits[ALL_ACTIONS.index(Action.TERMINATE)] = 3.5
            if mean_prm > -0.05:
                logits[ALL_ACTIONS.index(Action.CALL_VER)] = 3.0
                logits[ALL_ACTIONS.index(Action.SKIP_FINAL)] = 2.6
            logits[ALL_ACTIONS.index(Action.ADVANCE)] = 3.2
            if problem.difficulty == "hard" and step_idx % 3 == 0:
                logits[ALL_ACTIONS.index(Action.BRANCH)] = 2.5
            logits[ALL_ACTIONS.index(Action.CALL_CALC)] = 2.0

        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        aidx = int(np.argmax(probs))
        return ALL_ACTIONS[aidx], float(np.max(probs)), logits


class StudentNet(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.pi = nn.Linear(128, num_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(feats)
        return self.pi(h), self.v(h)


class StudentController:
    def __init__(self, model: StudentNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.inference_mode()
    def decide(self, problem: Problem, trace: List[str], prm_scores: List[float], features: Dict[str, Any], lam: float = 1.0) -> Tuple[Action, float, np.ndarray]:
        feat = torch.tensor([
            [
                features.get("step_idx", 0),
                features.get("cum_prm_logp", 0.0),
                features.get("elapsed_flops", 0.0) / 1e9,
                features.get("budget_remaining", 0.0) / 1e9,
                features.get("mean_prm", 0.0),
                {"easy": 0, "medium": 1, "hard": 2}[problem.difficulty],
                1.0 if problem.trap else 0.0,
                float(lam),
            ]
        ], dtype=torch.float32, device=self.device)
        logits, _ = self.model(feat)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        aidx = int(np.argmax(probs))
        return ALL_ACTIONS[aidx], float(np.max(probs)), logits.cpu().numpy()[0]


# --------------------------------
# Runner utilities and evaluation
# --------------------------------
@dataclass
class EpisodeResult:
    pid: str
    correct: int
    flops: float
    latency_ms_sim: float
    energy_J: float
    actions: List[str]
    prm_scores: List[float]


def run_episode(
    problem: Problem,
    backbone: SyntheticBackbone,
    prm: SimplePRM,
    controller_decide: Callable[[Problem, List[str], List[float], Dict[str, Any]], Tuple[Action, float, Any]],
    costs: Costs,
    flop_cap: float,
    tools: Tools,
) -> EpisodeResult:
    trace: List[str] = []
    prm_scores: List[float] = []
    elapsed_flops = 0.0
    elapsed_ms = 0.0
    actions_taken: List[str] = []

    backbone.reset()

    while True:
        features = {
            "step_idx": len(prm_scores),
            "cum_prm_logp": float(np.sum(prm_scores) if prm_scores else 0.0),
            "elapsed_flops": elapsed_flops,
            "budget_remaining": max(0.0, flop_cap - elapsed_flops),
            "mean_prm": float(np.mean(prm_scores) if prm_scores else -1.0),
        }
        action, conf, _ = controller_decide(problem, trace, prm_scores, features)

        if elapsed_flops + costs.flops[action] > flop_cap and action != Action.TERMINATE:
            action = Action.TERMINATE

        actions_taken.append(action.value)

        if action == Action.ADVANCE:
            step, logp = backbone.generate_next_step(problem, trace)
            trace.append(step)
            prm_scores.append(prm.score_step(step, problem.answer))
        elif action == Action.BRANCH:
            beams = backbone.branch_beam(problem, trace, k=2)
            scores = [prm.score_step(b, problem.answer) for b in beams]
            best = beams[int(np.argmax(scores))]
            trace.append(best)
            prm_scores.append(max(scores))
        elif action == Action.CALL_CALC:
            out = tools.calculator(backbone.extract_expr(problem, trace))
            trace.append(backbone.format_tool_result(out))
        elif action == Action.CALL_PY:
            code = backbone.extract_code(problem, trace)
            out = tools.python_exec(code)
            trace.append(backbone.format_tool_result(out))
        elif action == Action.CALL_RET:
            q = backbone.extract_query(problem, trace)
            docs = tools.retrieval(q)
            trace.append(backbone.format_docs(docs))
        elif action == Action.CALL_VER:
            ans = backbone.provisional_answer(trace)
            v = tools.verifier(ans, problem.answer)
            trace.append(backbone.format_verdict(v))
        elif action in (Action.SKIP_FINAL, Action.TERMINATE):
            break

        elapsed_flops += costs.flops[action]
        elapsed_ms += costs.ms[action]

        if elapsed_flops >= flop_cap:
            break

    pred = backbone.finalize(problem, trace)
    correct = int(backbone.checker(problem, pred))
    energy = elapsed_flops / FLOPS_PER_JOULE
    return EpisodeResult(
        pid=problem.pid,
        correct=correct,
        flops=elapsed_flops,
        latency_ms_sim=elapsed_ms,
        energy_J=energy,
        actions=actions_taken,
        prm_scores=prm_scores,
    )


# ------------------------
# Baseline controllers
# ------------------------
class FixedCascade:
    def __init__(self):
        self.name = "fixed_cascade"

    def decide(self, problem, trace, prm_scores, features):
        step_idx = features.get("step_idx", 0)
        if step_idx == 0:
            return Action.CALL_RET, 1.0, None
        if step_idx in (1, 2):
            return Action.ADVANCE, 1.0, None
        if step_idx == 3:
            return Action.CALL_CALC, 1.0, None
        if step_idx == 4:
            return Action.CALL_VER, 1.0, None
        return Action.TERMINATE, 1.0, None


class StaticBeam:
    def __init__(self):
        self.name = "beam5_prm"

    def decide(self, problem, trace, prm_scores, features):
        step_idx = features.get("step_idx", 0)
        if step_idx < 4:
            return Action.ADVANCE, 1.0, None
        return Action.TERMINATE, 1.0, None


class StaticCoT:
    def __init__(self):
        self.name = "static_cot13b"

    def decide(self, problem, trace, prm_scores, features):
        step_idx = features.get("step_idx", 0)
        if step_idx < 6:
            return Action.ADVANCE, 1.0, None
        return Action.TERMINATE, 1.0, None


# ------------------
# Distillation utils
# ------------------
@dataclass
class StateTuple:
    feat: np.ndarray
    action_id: int
    value_target: float
    teacher_logits: np.ndarray


def collect_teacher_rollouts(
    teacher: HeuristicTeacher,
    problems: List[Problem],
    backbone: SyntheticBackbone,
    prm: SimplePRM,
    costs: Costs,
    flop_cap: float,
    tools: Tools,
) -> Tuple[List[StateTuple], List[EpisodeResult]]:
    dataset: List[StateTuple] = []
    episodes: List[EpisodeResult] = []

    for pb in problems:
        trace: List[str] = []
        prm_scores: List[float] = []
        elapsed_flops = 0.0
        backbone.reset()
        while True:
            feats = {
                "step_idx": len(prm_scores),
                "cum_prm_logp": float(np.sum(prm_scores) if prm_scores else 0.0),
                "elapsed_flops": elapsed_flops,
                "budget_remaining": max(0.0, flop_cap - elapsed_flops),
                "mean_prm": float(np.mean(prm_scores) if prm_scores else -1.0),
            }
            action, conf, logits = teacher.decide(pb, trace, prm_scores, feats)

            feat_vec = np.array(
                [
                    feats["step_idx"],
                    feats["cum_prm_logp"],
                    feats["elapsed_flops"] / 1e9,
                    feats["budget_remaining"] / 1e9,
                    feats["mean_prm"],
                    {"easy": 0, "medium": 1, "hard": 2}[pb.difficulty],
                    1.0 if pb.trap else 0.0,
                    1.0,  # lambda placeholder
                ],
                dtype=np.float32,
            )
            a_id = ALL_ACTIONS.index(action)
            t_logits = np.array(logits if logits is not None else np.zeros(len(ALL_ACTIONS)), dtype=np.float32)
            dataset.append(StateTuple(feat=feat_vec, action_id=a_id, value_target=0.0, teacher_logits=t_logits))

            if elapsed_flops + costs.flops[action] > flop_cap and action != Action.TERMINATE:
                action = Action.TERMINATE

            if action == Action.ADVANCE:
                step, _ = backbone.generate_next_step(pb, trace)
                trace.append(step)
                prm_scores.append(prm.score_step(step, pb.answer))
            elif action == Action.BRANCH:
                beams = backbone.branch_beam(pb, trace, k=2)
                scores = [prm.score_step(b, pb.answer) for b in beams]
                best = beams[int(np.argmax(scores))]
                trace.append(best)
                prm_scores.append(max(scores))
            elif action == Action.CALL_CALC:
                out = tools.calculator(backbone.extract_expr(pb, trace))
                trace.append(backbone.format_tool_result(out))
            elif action == Action.CALL_PY:
                code = backbone.extract_code(pb, trace)
                out = tools.python_exec(code)
                trace.append(backbone.format_tool_result(out))
            elif action == Action.CALL_RET:
                q = backbone.extract_query(pb, trace)
                docs = tools.retrieval(q)
                trace.append(backbone.format_docs(docs))
            elif action == Action.CALL_VER:
                ans = backbone.provisional_answer(trace)
                v = tools.verifier(ans, pb.answer)
                trace.append(backbone.format_verdict(v))
            elif action in (Action.SKIP_FINAL, Action.TERMINATE):
                break

            elapsed_flops += costs.flops[action]
            if elapsed_flops >= flop_cap:
                break

        # Also store an episode run for metrics
        ep = run_episode(
            pb,
            backbone,
            prm,
            lambda p, t, ps, f: teacher.decide(p, t, ps, f),
            costs,
            flop_cap,
            tools,
        )
        episodes.append(ep)

    return dataset, episodes


def train_student_model(
    train_tuples: List[StateTuple],
    val_tuples: List[StateTuple],
    epochs: int = 3,
    lr: float = 1e-3,
) -> Tuple[StudentNet, List[float]]:
    device = "cpu"  # keep small and portable
    feat_dim = len(train_tuples[0].feat)
    model = StudentNet(feat_dim=feat_dim, num_actions=len(ALL_ACTIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def to_batch(tuples: List[StateTuple], bs: int = 64):
        for i in range(0, len(tuples), bs):
            batch = tuples[i : i + bs]
            feats = torch.tensor([x.feat for x in batch], dtype=torch.float32)
            acts = torch.tensor([x.action_id for x in batch], dtype=torch.long)
            tlog = torch.tensor([x.teacher_logits for x in batch], dtype=torch.float32)
            vals = torch.tensor([x.value_target for x in batch], dtype=torch.float32)
            yield feats, acts, tlog, vals

    train_losses: List[float] = []
    for ep in range(epochs):
        model.train()
        losses: List[float] = []
        for feats, acts, tlog, vals in to_batch(train_tuples):
            feats, acts, tlog, vals = feats.to(device), acts.to(device), tlog.to(device), vals.to(device)
            logits, v = model(feats)
            ce = F.cross_entropy(logits, acts)
            with torch.no_grad():
                tprob = F.softmax(tlog, dim=-1)
            kl = F.kl_div(F.log_softmax(logits, dim=-1), tprob, reduction="batchmean")
            mse = F.mse_loss(v.squeeze(-1), vals)
            loss = ce + 0.1 * kl + 0.5 * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        avg_loss = float(np.mean(losses))
        train_losses.append(avg_loss)
        # quick validation: action agreement
        model.eval()
        agrees = []
        with torch.no_grad():
            for feats, acts, _, _ in to_batch(val_tuples, bs=128):
                logits, _ = model(feats.to(device))
                pred = logits.argmax(dim=-1).cpu()
                agrees.append((pred == acts).float().mean().item())
        print(f"[Distill] Epoch {ep+1}/{epochs} loss={avg_loss:.4f} val_action_agree={np.mean(agrees):.3f}")

    return model, train_losses


# ------------------------
# Dataset I/O helpers
# ------------------------

def load_dataset(path: str) -> List[Problem]:
    problems: List[Problem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            problems.append(
                Problem(
                    pid=obj["pid"],
                    question=obj["question"],
                    expression=obj["expression"],
                    answer=float(obj["answer"]),
                    difficulty=obj["difficulty"],
                    trap=bool(obj["trap"]),
                )
            )
    return problems


# ------------------------------------
# High-level training (Exp 3 + artifacts)
# ------------------------------------

def experiment3_distillation(
    train_probs: List[Problem],
    val_probs: List[Problem],
    images_dir: str,
    epochs: int = 3,
    lr: float = 1e-3,
    flop_cap: float = 1.2e10,
) -> Dict[str, Any]:
    ensure_dir(images_dir)
    sns.set_theme(style="whitegrid")

    teacher = HeuristicTeacher()
    backbone = SyntheticBackbone()
    prm = SimplePRM()
    tools = Tools()

    train_tuples, _ = collect_teacher_rollouts(teacher, train_probs, backbone, prm, DEFAULT_COSTS, flop_cap, tools)
    val_tuples, _ = collect_teacher_rollouts(teacher, val_probs, backbone, prm, DEFAULT_COSTS, flop_cap, tools)
    print(f"Collected states: train={len(train_tuples)} val={len(val_tuples)}")

    student_model, losses = train_student_model(train_tuples, val_tuples, epochs=epochs, lr=lr)

    # Loss curve
    fig = plt.figure(figsize=(6, 4))
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Meta-reward student training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "training_loss_meta_reward.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Action agreement/confusion
    def tuples_to_preds(tuples: List[StateTuple]):
        y_true, y_pred = [], []
        with torch.no_grad():
            for t in tuples:
                feats = torch.tensor([t.feat], dtype=torch.float32)
                logits, _ = student_model(feats)
                pred = int(torch.argmax(logits, dim=-1).item())
                y_true.append(int(t.action_id))
                y_pred.append(pred)
        return np.array(y_true), np.array(y_pred)

    y_true, y_pred = tuples_to_preds(val_tuples)
    agree = float(np.mean(y_true == y_pred))
    print(f"Student action agreement vs teacher (val): {agree:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_ACTIONS))))
    figcm = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[a.name for a in ALL_ACTIONS],
        yticklabels=[a.name for a in ALL_ACTIONS],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Teacher")
    plt.title("Action confusion matrix (student vs teacher)")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "confusion_matrix_actions.pdf"), bbox_inches="tight")
    plt.close(figcm)

    return {
        "student_model": student_model,
        "train_losses": losses,
        "agreement": agree,
    }


def save_model(model: StudentNet, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(path: str) -> StudentNet:
    model = StudentNet(feat_dim=8, num_actions=len(ALL_ACTIONS))
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model
