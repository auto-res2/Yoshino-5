import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import TinyModel, exact_match, faithfulness
from preprocess import ToyTask, collate


def experiment1(images_dir: str):
    tasks = {
        "toy_gsm8k": ToyTask(128, seed=1),
        "toy_mmlu_logic": ToyTask(128, seed=2),
        "toy_hellaswag": ToyTask(128, seed=3),
    }
    systems = {
        "S0_Vanilla": TinyModel(with_graph_head=False, seed=0),
        "S1_CoT": TinyModel(with_graph_head=False, seed=1),
        "S2_KPOD": TinyModel(with_graph_head=False, seed=2),
        "S3_GoTD": TinyModel(with_graph_head=True, seed=3),
        "S4_ExeGoT": TinyModel(with_graph_head=True, seed=4),
    }
    os.makedirs(images_dir, exist_ok=True)
    metrics = []
    for tname, dset in tasks.items():
        loader = DataLoader(dset, batch_size=32, shuffle=False, collate_fn=collate)
        gt_all = []
        for _, gt in loader:
            gt_all.append(gt)
        import torch
        gt_all = torch.cat(gt_all)
        for sname, model in systems.items():
            preds, graphs = [], []
            for qbatch, _ in loader:
                a, g = model.generate(qbatch, return_graph=True)
                preds += a
                graphs += g
            acc = exact_match(preds, gt_all)
            fth = faithfulness(graphs, preds)
            metrics.append((tname, sname, acc, fth))
            print(f"{tname:<15} {sname:<12} acc={acc:5.2f} faith={fth:5.2f}")
    labels = [f"{t}\n{s}" for (t, s, _, _) in metrics]
    accs = [m[2] for m in metrics]
    fths = [m[3] for m in metrics]
    x = np.arange(len(accs))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, accs, 0.4, label="Accuracy")
    plt.bar(x + 0.2, fths, 0.4, label="Faithfulness")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=7)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    pdf = f"{images_dir}/benchmark_scores.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    return {"benchmark_pdf": pdf}
