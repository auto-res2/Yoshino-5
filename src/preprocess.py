"""Dataset and transformation helpers.
This module also applies a *compatibility patch* for recent versions of
`transformers` that expect `torch.utils._pytree.register_pytree_node` to be
present (the symbol was renamed upstream to *_register_pytree_node* starting
with PyTorch 2.1).  Importing `torchvision` → `torch.onnx` transitively imports
`transformers`, so we need the patch **before** we import anything from
`torchvision`.
"""
from __future__ import annotations

import torch

# -----------------------------------------------------------------------------
# Compatibility patch for PyTorch ≥2.1  vs.  transformers ≥4.37
# -----------------------------------------------------------------------------
try:
    from torch.utils import _pytree as _torch_pytree  # type: ignore
    if (not hasattr(_torch_pytree, 'register_pytree_node') and
            hasattr(_torch_pytree, '_register_pytree_node')):
        _torch_pytree.register_pytree_node = _torch_pytree._register_pytree_node  # type: ignore
except Exception:
    # If for some reason the internal module layout changes, we simply skip the
    # patch – the import error will make debugging straightforward.
    pass

# Now it is safe to import torchvision / avalanche which will indirectly import
# transformers.
from torchvision import transforms  # noqa: E402
from avalanche.benchmarks.classic import SplitCIFAR100, PermutedMNIST  # noqa: E402
from avalanche.benchmarks.utils import benchmark_with_validation_stream  # noqa: E402

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _get_transforms(dataset_name: str, image_size: int, is_train: bool):
    if 'cifar' in dataset_name:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        if is_train:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    elif 'mnist' in dataset_name:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    raise ValueError(f"Transforms for dataset {dataset_name} not defined.")


def get_benchmark(config):
    """Factory function to get the specified benchmark. Returns a benchmark object."""
    dataset_name = config.dataset.lower()
    train_transform = _get_transforms(dataset_name, config.image_size, is_train=True)
    eval_transform = _get_transforms(dataset_name, config.image_size, is_train=False)

    if 'split_cifar100' in dataset_name:
        benchmark = SplitCIFAR100(
            n_experiences=config.num_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=config.dataset_root,
        )
    elif 'permuted_mnist' in dataset_name:
        benchmark = PermutedMNIST(
            n_experiences=config.num_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=config.dataset_root,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Optionally add a validation stream – Avalanche returns a *new benchmark*
    # with an additional validation stream, so we just overwrite the reference.
    if getattr(config, 'validation_size', 0) and config.validation_size > 0:
        benchmark = benchmark_with_validation_stream(benchmark, validation_size=config.validation_size)

    return benchmark