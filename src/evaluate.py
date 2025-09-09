import torch
import numpy as np
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def calculate_accuracy(model, data_loader, device):
    """Calculates accuracy for a single task."""
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0


def evaluate_model(model, task_loaders, seen_tasks_indices, device):
    """
    Evaluates the model on all previously seen tasks.
    Returns a list of accuracies for each seen task.
    """
    accuracies = []
    for task_idx in seen_tasks_indices:
        acc = calculate_accuracy(model, task_loaders[task_idx], device)
        accuracies.append(acc)
        print(f"Accuracy on task {task_idx}: {acc:.2f}%")
    return accuracies


def calculate_metrics(accuracy_matrix):
    """
    Calculates Average Accuracy (AACC) and Average Forgetting (AF).
    Args:
        accuracy_matrix (np.array): A T x T matrix where A[i, j] is the accuracy
                                     on task j after training on task i.
    """
    num_tasks = accuracy_matrix.shape[0]
    if num_tasks == 0:
        return 0.0, 0.0

    # Average Accuracy (AACC) at the end of training
    final_accuracies = accuracy_matrix[-1, :]
    aacc = np.mean(final_accuracies)

    # Average Forgetting (AF)
    forgetting = 0.0
    for j in range(num_tasks - 1):
        # Max accuracy on task j
        max_acc_j = np.max(accuracy_matrix[:j+2, j])
        # Accuracy on task j after the final task
        final_acc_j = accuracy_matrix[-1, j]
        forgetting += (max_acc_j - final_acc_j)

    af = forgetting / (num_tasks - 1) if num_tasks > 1 else 0.0
    return aacc, af


def plot_results(results, _save_path_ignored):
    """Plots the evolution of AACC and AF over tasks and saves the figure to
    the mandated directory: `.research/iteration10/images`. The caller-provided
    ``save_path`` argument is ignored to comply with the execution rules.
    """
    save_path = ".research/iteration10/images"
    os.makedirs(save_path, exist_ok=True)

    task_indices = range(1, len(results['aacc']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot AACC
    plt.subplot(1, 2, 1)
    plt.plot(task_indices, results['aacc'], marker='o', linestyle='-')
    plt.title('Average Accuracy (AACC) vs. Tasks Trained')
    plt.xlabel('Number of Tasks Trained')
    plt.ylabel('AACC (%)')
    plt.grid(True)
    plt.xticks(task_indices)

    # Plot AF
    plt.subplot(1, 2, 2)
    plt.plot(task_indices, results['af'], marker='s', linestyle='-', color='r')
    plt.title('Average Forgetting (AF) vs. Tasks Trained')
    plt.xlabel('Number of Tasks Trained')
    plt.ylabel('AF (%)')
    plt.grid(True)
    plt.xticks(task_indices)

    plt.tight_layout()
    plot_filename = os.path.join(save_path, "aacc_af_plot.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot to {plot_filename}")
