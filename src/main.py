"""src/main.py
----------------------------------------------------------------------
Orchestration script.  Execute with:
    python -m src.main
"""
from __future__ import annotations

import yaml
import os
import numpy as np

from torch.utils.data import DataLoader

from . import train as tr
from . import preprocess as pp
from . import evaluate as ev

# ------------------------------------------------------------------
# Load configuration ------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ------------------------------------------------------------------
# Experiment 1 – IATG vs. baselines (NLP branch only)
# ------------------------------------------------------------------

def experiment_one():
    cfg = CFG["exp1"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    results_summary = {}

    # --------------------------------------------------------------
    # Build dataloaders for each NI task
    # --------------------------------------------------------------
    tokenizer_prototype = tr.load_llm(model_cfg)[1]  # tokenizer only (cheap)
    dataloaders = {}
    for task in cfg["datasets"]["nlp_tasks"]:
        tr_set = pp.NaturalInstructionTask(task, "train", tokenizer_prototype, train_cfg["max_seq_len"])
        val_set = pp.NaturalInstructionTask(task, "validation", tokenizer_prototype, train_cfg["max_seq_len"])
        te_set = pp.NaturalInstructionTask(task, "test", tokenizer_prototype, train_cfg["max_seq_len"])
        dl_tr = DataLoader(
            tr_set,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=lambda b: pp.collate_fn_lm(b, tokenizer_prototype.pad_token_id),
        )
        dl_val = DataLoader(
            val_set,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=lambda b: pp.collate_fn_lm(b, tokenizer_prototype.pad_token_id),
        )
        dl_te = DataLoader(
            te_set,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=lambda b: pp.collate_fn_lm(b, tokenizer_prototype.pad_token_id),
        )
        dataloaders[task] = (dl_tr, dl_val, dl_te)

    # --------------------------------------------------------------
    # Build IATG scheduler to obtain proposed ordering
    # --------------------------------------------------------------
    scheduler_cfg = cfg["scheduler"]
    scheduler = tr.IATGScheduler(beam=scheduler_cfg["beam_width"])
    ordering_proposed: list[str] = []

    # Use a lightweight model snapshot for probing (reduces VRAM)
    model_probe, tokenizer_probe = tr.load_llm(model_cfg)
    model_probe.to(tr.DEVICE)
    model_probe.zero_grad(set_to_none=True)

    for task in cfg["datasets"]["nlp_tasks"]:
        calibr_dl = DataLoader(
            pp.NaturalInstructionTask(task, "train", tokenizer_probe),
            batch_size=32,
            shuffle=True,
            collate_fn=lambda b: pp.collate_fn_lm(b, tokenizer_probe.pad_token_id),
        )
        batch = next(iter(calibr_dl))
        batch = {k: v.to(tr.DEVICE) for k, v in batch.items()}
        out = model_probe(**batch, output_hidden_states=True)
        loss = out.loss
        loss.backward()
        grad_vec = torch.cat(
            [p.grad.flatten().detach().float() for p in model_probe.parameters() if p.grad is not None]
        )
        feat_vec = (
            out.hidden_states[-1].detach().float().mean(dim=1).flatten()
            if hasattr(out, "hidden_states")
            else torch.randn(768)
        )
        ordering_proposed = scheduler.add_task(task, grad_vec, feat_vec)
        print("[SCHED] current order:", ordering_proposed)
    del model_probe  # free VRAM

    baseline_orders = {
        "iatg": ordering_proposed,
        "chronological": tr.OrderingFactory.chronological(cfg["datasets"]["nlp_tasks"]),
        "random": tr.OrderingFactory.random(cfg["datasets"]["nlp_tasks"]),
    }

    # --------------------------------------------------------------
    # Run training for each ordering × seed
    # --------------------------------------------------------------
    for ordering_name, order in baseline_orders.items():
        for seed in cfg["seeds"]:
            model, tokenizer = tr.load_llm(model_cfg)  # fresh model per run
            learner = tr.ContinualLearner(train_cfg, model, tokenizer, ordering_name, order, seed)
            acc, bwt = learner.run_stream(dataloaders)
            results_summary.setdefault(ordering_name, []).append(acc)
            print(
                f"[RESULT] ordering={ordering_name} seed={seed} ACC={acc:.3f} BWT={bwt:.3f}"
            )

    # --------------------------------------------------------------
    # Print statistics & plot
    # --------------------------------------------------------------
    for name, values in results_summary.items():
        mean, ci = ev.summary_ci(values)
        print(f"{name}: ACC={mean:.3f} ±{ci:.3f} (95% CI)")

    ev.plot_bar({k: np.mean(v) for k, v in results_summary.items()}, "Final ACC – Exp1", "accuracy_exp1.pdf")

# ------------------------------------------------------------------
# Entry point -------------------------------------------------------

def main():
    print("================ Experiment 1: IATG vs Baselines ================\n")
    experiment_one()

if __name__ == "__main__":
    main()
