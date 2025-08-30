import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from torchvision import transforms


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_default() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RotatedMNIST(torch.utils.data.Dataset):
    def __init__(self, base, angle: float):
        self.base = base
        self.angle = angle
        self.rotate = transforms.RandomRotation((angle, angle))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = self.rotate(x)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x, y


class PermutedMNIST(torch.utils.data.Dataset):
    def __init__(self, base, perm: torch.Tensor):
        self.base = base
        self.perm = perm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = x.view(-1)[self.perm].view(1, 28, 28)
        x = x.repeat(3, 1, 1)
        return x, y


def get_rotated_mnist_tasks(num_tasks: int = 20, train: bool = True, data_root: str = "./data") -> List[torch.utils.data.Dataset]:
    tfm = transforms.ToTensor()
    base = tv.datasets.MNIST(root=data_root, train=train, download=True, transform=tfm)
    angles = np.linspace(0, 180, num_tasks)
    return [RotatedMNIST(base, float(a)) for a in angles]


def get_permuted_mnist_tasks(num_tasks: int = 20, train: bool = True, data_root: str = "./data") -> List[torch.utils.data.Dataset]:
    tfm = transforms.ToTensor()
    base = tv.datasets.MNIST(root=data_root, train=train, download=True, transform=tfm)
    perms = [torch.randperm(28 * 28) for _ in range(num_tasks)]
    return [PermutedMNIST(base, p) for p in perms]


def get_cifar100_split_tasks(num_tasks: int = 20, classes_per_task: int = 5, train: bool = True, data_root: str = "./data") -> List[torch.utils.data.Dataset]:
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    ds = tv.datasets.CIFAR100(root=data_root, train=train, download=True, transform=(tfm_train if train else tfm_test))
    class_order = list(range(100))
    tasks: List[torch.utils.data.Dataset] = []
    for t in range(num_tasks):
        cls = class_order[t * classes_per_task:(t + 1) * classes_per_task]
        idx = [i for i, (_, y) in enumerate(ds) if y in cls]
        tasks.append(Subset(ds, idx))
    return tasks


def build_task_loaders(tasks: List[torch.utils.data.Dataset], batch_train: int = 64, batch_eval: int = 128, num_workers: int = 2):
    ldr_tr = [DataLoader(t, batch_size=batch_train, shuffle=True, num_workers=num_workers) for t in tasks]
    ldr_te = [DataLoader(t, batch_size=batch_eval, shuffle=False, num_workers=num_workers) for t in tasks]
    return ldr_tr, ldr_te


def get_images_dir() -> str:
    path = os.path.join('.research', 'iteration3', 'images')
    os.makedirs(path, exist_ok=True)
    return path
