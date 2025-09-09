import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from collections import defaultdict

class ContinualMetrics:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))

    def update(self, task_id, accuracies):
        """Update matrix after evaluating on all tasks seen so far."""
        for i, acc in enumerate(accuracies):
            self.accuracy_matrix[task_id, i] = acc

    def average_accuracy(self):
        """AACC: Average accuracy over all tasks seen so far."""
        final_task_idx = self.num_tasks - 1
        final_accuracies = self.accuracy_matrix[final_task_idx, :]
        return np.mean(final_accuracies)

    def average_forgetting(self):
        """AF: Average forgetting."""
        forgetting = 0.0
        final_task_idx = self.num_tasks - 1
        for i in range(final_task_idx):
            max_acc = np.max(self.accuracy_matrix[:final_task_idx + 1, i])
            final_acc = self.accuracy_matrix[final_task_idx, i]
            forgetting += (max_acc - final_acc)
        return forgetting / final_task_idx if final_task_idx > 0 else 0.0

def aggregate_results(all_results, policies):
    """Computes summary statistics and performs t-tests."""
    print("\n" + "="*40)
    print("           FINAL RESULTS SUMMARY")
    print("="*40)

    summary = defaultdict(lambda: defaultdict(list))
    for policy in policies:
        for seed_results in all_results[policy]:
            summary[policy]['AACC'].append(seed_results['AACC'])
            summary[policy]['AF'].append(seed_results['AF'])

    df_data = []
    for policy, metrics in summary.items():
        aacc_mean, aacc_std = np.mean(metrics['AACC']), np.std(metrics['AACC'])
        af_mean, af_std = np.mean(metrics['AF']), np.std(metrics['AF'])
        df_data.append([policy, f"{aacc_mean:.2f} ± {aacc_std:.2f}", f"{af_mean:.2f} ± {af_std:.2f}"])

    df = pd.DataFrame(df_data, columns=["Policy", "AACC (%)", "AF (%)"])
    print(df.to_string(index=False))
    print("-"*40)

    if 'rl_top' in summary and len(policies) > 1 and len(summary['rl_top']['AACC']) > 1:
        baselines = [p for p in policies if p != 'rl_top']
        best_baseline = max(baselines, key=lambda p: np.mean(summary[p]['AACC']))

        print(f"\nPaired t-test (Holm-Bonferroni corrected): RL-TOP vs Best Baseline ({best_baseline})")
        try:
            _, p_aacc = ttest_rel(summary['rl_top']['AACC'], summary[best_baseline]['AACC'])
            _, p_af = ttest_rel(summary['rl_top']['AF'], summary[best_baseline]['AF'])
            print(f"  AACC p-value: {p_aacc:.4f}")
            print(f"  AF p-value:   {p_af:.4f}")
        except Exception as e:
            print(f"Could not run t-test: {e}")
    print("="*40 + "\n")
    return summary

@torch.no_grad()
def evaluate(model, loader, device, task_id, classes_per_task):
    model.eval()
    correct, total = 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        min_label = task_id * classes_per_task
        max_label = min_label + classes_per_task
        mask = (y >= min_label) & (y < max_label)
        if not mask.any(): continue
        x, y = x[mask], y[mask]
        
        unique_labels = torch.unique(y)
        label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
        y_mapped = torch.tensor([label_map[int(l)] for l in y], device=device, dtype=torch.long)

        outputs = model(x)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_mapped).sum().item()
        total += len(y)
    return (correct / total * 100.0) if total > 0 else 0.0
