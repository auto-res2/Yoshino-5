import numpy as np
import torch
import torch.nn.functional as F
import copy
from contextlib import nullcontext
from scipy.stats import ttest_rel
import pandas as pd


def get_gradients(model, data_loader, device, num_classes, precision):
    model.train()
    images, labels, _ = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    unique_labels = torch.unique(labels)
    label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
    labels = torch.tensor([label_map[int(l)] for l in labels], device=device, dtype=torch.long)

    model.zero_grad()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    amp_ctx = (
        torch.cuda.amp.autocast(dtype=dtype) if device.type == "cuda" else nullcontext()
    )
    with amp_ctx:
        outputs = model(images)
        loss = F.cross_entropy(outputs[:, :num_classes], labels)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def compute_metric_matrices(model, candidate_dataloaders, device, classes_per_task, precision, ablations):
    m = len(candidate_dataloaders)
    s_matrix = np.zeros((m, m))
    g_matrix = np.zeros((m, m))
    if m <= 1:
        return s_matrix, g_matrix

    grad_vectors = []
    for loader in candidate_dataloaders:
        grads = get_gradients(copy.deepcopy(model), loader, device, classes_per_task, precision)
        grad_vectors.append(grads)

    for r in range(m):
        for c in range(r + 1, m):
            grad1, grad2 = grad_vectors[r], grad_vectors[c]
            if not ablations.get("g_only", False):
                sim = F.cosine_similarity(grad1, grad2, dim=0).item()
                s_matrix[r, c] = s_matrix[c, r] = sim
            if not ablations.get("s_only", False):
                inter = -F.cosine_similarity(grad1, grad2, dim=0).item()
                g_matrix[r, c] = g_matrix[c, r] = inter

    return s_matrix, g_matrix


def analyze_and_print_results(all_results):
    print("\n--- FINAL RESULTS (Mean ± SD over seeds) ---")
    summary = {}
    for policy, metrics in all_results.items():
        summary[policy] = {
            "aacc_mean": np.mean(metrics["AACC"]),
            "aacc_std": np.std(metrics["AACC"]),
            "af_mean": np.mean(metrics["AF"]),
            "af_std": np.std(metrics["AF"]),
        }

    df_data = []
    for policy, data in summary.items():
        aacc_str = f"{data['aacc_mean']:.2f} ± {data['aacc_std']:.2f}"
        af_str = f"{data['af_mean']:.2f} ± {data['af_std']:.2f}"
        df_data.append([policy, aacc_str, af_str])

    df = pd.DataFrame(df_data, columns=["Policy", "AACC (%)", "AF (%)"])
    print(df.to_string(index=False))

    print("\nSummary Dictionary:")
    print(summary)

    if "rl_top" in all_results and "random" in all_results:
        best_baseline = "random"
        if len(all_results["rl_top"]["AACC"]) > 1 and len(all_results[best_baseline]["AACC"]) > 1:
            try:
                stat_aacc, p_aacc = ttest_rel(
                    all_results["rl_top"]["AACC"], all_results[best_baseline]["AACC"]
                )
                stat_af, p_af = ttest_rel(
                    all_results["rl_top"]["AF"], all_results[best_baseline]["AF"]
                )
                print("\nPaired t-test (RL-TOP vs. Random):")
                print(f"AACC: p-value = {p_aacc:.4f}")
                print(f"AF: p-value = {p_af:.4f} (lower is better)")
            except Exception as e:  # pragma: no cover
                print(f"\nCould not run t-test: {e}")

    return summary
