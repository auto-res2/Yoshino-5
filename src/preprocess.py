import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

class TaskDataset(Dataset):
    """A wrapper for a dataset to assign a specific task_id."""
    def __init__(self, dataset, task_id, transform=None):
        self.dataset = dataset
        self.task_id = task_id
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        # Remap label to be within the task
        return img, label, self.task_id

def get_datasets(name, num_tasks, data_path):
    """
    Prepares continual learning datasets.
    Currently supports 'Split-CIFAR-100'.
    """
    if name.lower() != "split-cifar-100":
        raise ValueError(f"Dataset '{name}' not supported.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    try:
        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True)
        test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True)
    except Exception as e:
        print(f"Failed to download CIFAR-100. Please check your network connection or data path permissions.")
        raise e

    train_tasks = []
    test_tasks = []
    
    classes_per_task = 100 // num_tasks
    class_order = np.arange(100)
    # np.random.shuffle(class_order) # Optionally shuffle class order

    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        task_classes = class_order[start_class:end_class]

        # --- Training data ---
        train_indices = [i for i, target in enumerate(train_dataset.targets) if target in task_classes]
        train_subset = Subset(train_dataset, train_indices)
        train_task_dataset = TaskDataset(train_subset, task_id, transform=transform)
        train_tasks.append(train_task_dataset)
        
        # --- Test data ---
        test_indices = [i for i, target in enumerate(test_dataset.targets) if target in task_classes]
        test_subset = Subset(test_dataset, test_indices)
        test_task_dataset = TaskDataset(test_subset, task_id, transform=transform)
        test_tasks.append(test_task_dataset)

    print(f"Successfully created {num_tasks} tasks for Split-CIFAR-100.")
    return train_tasks, test_tasks
