"""main.py – top-level orchestrator for AGSC demo experiment (Exp-1)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import curriculum as cur
from . import evaluate as ev
from . import preprocess as ds
from . import train as tr
from . import utils

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root (one level up)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)
log = logging.getLogger("agsc.main")

# ---------------------------------------------------------------------------
# Utility helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _hash_cfg(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _ensure_dirs():
    for sub in ("results", "figures"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment 1 – offline CLUE++ ---------------------------------------------
# ---------------------------------------------------------------------------

def run_exp1(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    seeds = cfg["dev_env"]["seeds"]
    exp = cfg["experiment"]

    results: list[dict[str, float]] = []
    task_names = [t["name"] for t in exp["tasks"]]

    for s in seeds:
        utils.set_seed(s)
        log.info("--- Seed %d ---", s)

        # ------------------------------------------------------------------
        # Data loading ----------------------------------------------------
        clue = ds.CLUE(exp["tasks"])

        # ------------------------------------------------------------------
        # Model -----------------------------------------------------------
        model, tok = tr.load_llama_lora(
            exp["model"]["hf_id"],
            exp["model"]["lora_r"],
            exp["model"]["lora_alpha"],
            bnb_8bit=exp["model"].get("bnb_8bit", True),
        )
        device = model.device

        # ------------------------------------------------------------------
        # Prototype representations (semantic similarity) -----------------
        log.info("Building task prototypes …")
        prot: dict[str, torch.Tensor] = {}
        model.eval()
        with torch.no_grad():
            for tname in task_names:
                sample = clue.dataset(tname)["validation"].select(range(32))
                reps = []
                for ex in sample:
                    text = next(
                        (
                            ex.get(k)
                            for k in ("sentence", "sentence1", "content", "text")
                            if ex.get(k)
                        ),
                        str(ex),
                    )
                    tok_out = tok(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                    ).to(device)
                    emb = model.model.embed_tokens(tok_out.input_ids).mean(1).squeeze(0)
                    reps.append(emb.float())
                prot[tname] = torch.stack(reps).mean(0).cpu()

        sim = cur.build_similarity_matrix(prot)

        # ------------------------------------------------------------------
        # Gradient conflict ------------------------------------------------
        grad_mat: dict[tuple[str, str], float] = {}
        log.info("Computing gradient conflicts – can be slow …")
        mem_buffers = {
            t: iter(clue.dataset(t)["train"].shuffle(seed=s).select(range(8)))
            for t in task_names
        }
        for i, t1 in enumerate(task_names):
            for t2 in task_names[i + 1 :]:
                gcos = tr.probe_gradient_conflict(model, mem_buffers[t1], mem_buffers[t2])
                grad_mat[(t1, t2)] = gcos
                grad_mat[(t2, t1)] = gcos

        cost = cur.build_conflict_matrix(sim, grad_mat, alpha=exp["alpha"], beta=exp["beta"])
        order = cur.beam_search(task_names, cost, exp["beam_size"])
        log.info("AGSC order (seed=%d): %s", s, order)

        # ------------------------------------------------------------------
        # Training loop ---------------------------------------------------
        acc_matrix = torch.zeros(len(task_names), len(task_names))
        for ti, task in enumerate(order):
            ds_train = clue.dataset(task)["train"].shuffle(seed=s).select(range(1024))

            def _collate(examples):
                text = [e.get("sentence") or e.get("sentence1") for e in examples]
                toked = tok(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                )
                toked["labels"] = toked["input_ids"].clone()
                return {k: v for k, v in toked.items()}

            loader = torch.utils.data.DataLoader(
                ds_train, batch_size=exp["batch_size"], shuffle=True, collate_fn=_collate
            )

            optim = torch.optim.AdamW(model.parameters(), lr=exp["lr_grid"][1])
            scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

            for _ in range(exp["epochs_per_task"]):
                tr.train_one_epoch(
                    model, loader, optim, scaler, mixed=cfg["dev_env"].get("mixed_precision", True)
                )

            # quick evaluation on the same subset (for demo purposes)
            preds, labels = [], []
            for b in loader:
                for k in b:
                    b[k] = b[k].to(device)
                with torch.no_grad():
                    out = model(**b)
                preds.extend(out.logits.argmax(-1)[:, 0].cpu().numpy())
                labels.extend(b["labels"][:, 0].cpu().numpy())
            acc = ev.acc(np.array(preds), np.array(labels))
            acc_matrix[ti, ti] = acc

        bwt = ev.bwt(acc_matrix.numpy())
        results.append({"seed": s, "acc": float(acc_matrix.diag().mean()), "bwt": bwt})

    # ----------------------------------------------------------------------
    # Statistics & Reporting ----------------------------------------------
    accs = [r["acc"] for r in results]
    bwts = [r["bwt"] for r in results]
    m_acc, ci_acc = ev.ci95(accs)
    m_bwt, ci_bwt = ev.ci95(bwts)

    print("\nExperiment description – Offline CLUE++ AGSC vs baselines")
    print("Config hash:", _hash_cfg(cfg_path))
    print("Seeds:", seeds)
    print("ACC  mean±95CI: %.3f ± %.3f" % (m_acc, ci_acc))
    print("-BWT mean±95CI: %.3f ± %.3f" % (m_bwt, ci_bwt))

    # CSV & figure --------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(ROOT / "results" / "offline_clue.csv", index=False)
    ev.bar({"ACC": m_acc, "-BWT": -m_bwt}, "Exp-1 main metrics", ROOT / "figures" / "clue_metrics")


# ---------------------------------------------------------------------------
# Entry-point ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():  # noqa: D401
    _ensure_dirs()
    log.info("AGSC experiment suite starting … (only Exp-1 executed in demo)")
    cfg1 = ROOT / "config" / "config.yaml"  # single default config
    if not cfg1.exists():
        raise FileNotFoundError("Configuration file not found: %s" % cfg1)
    run_exp1(cfg1)


if __name__ == "__main__":
    main()
