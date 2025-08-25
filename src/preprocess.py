import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_synthetic_cifar100_tasks(num_tasks=2, num_classes=10, samples_per_task=100):
    """Generate synthetic data to mimic Split-CIFAR100 tasks.
       Each task has a fixed number of samples and a set of labels [0, num_classes-1].
    """
    tasks = []
    for t in range(num_tasks):
        data = torch.randn(samples_per_task, 3, 32, 32)
        labels = torch.randint(0, num_classes, (samples_per_task,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        tasks.append(loader)
    return tasks

def get_synthetic_permuted_mnist_tasks(num_tasks=2, samples_per_task=100):
    """Generate synthetic data to mimic Permuted-MNIST tasks.
       Use 1x28x28 images. For each task, apply a fixed permutation.
    """
    tasks = []
    base_data = torch.randn(samples_per_task, 1, 28, 28)
    base_labels = torch.randint(0, 10, (samples_per_task,))
    for t in range(num_tasks):
        perm = torch.randperm(28*28)
        permuted_data = base_data.view(samples_per_task, -1)[:, perm].view(samples_per_task, 1, 28, 28)
        dataset = TensorDataset(permuted_data, base_labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        tasks.append(loader)
    return tasks
