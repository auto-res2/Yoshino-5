from importlib import import_module
import torch
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100, PermutedMNIST, SplitCIFAR10

BENCHMARK_WVS_PATH = "avalanche.benchmarks.utils"

try:
    benchmark_with_validation_stream = getattr(
        import_module(BENCHMARK_WVS_PATH), "benchmark_with_validation_stream"
    )
except (ImportError, AttributeError):
    print(
        "WARNING: Could not import 'benchmark_with_validation_stream' from "
        f"'{BENCHMARK_WVS_PATH}'. Using a no-op fallback. Validation streams will be disabled."
    )
    def benchmark_with_validation_stream(benchmark, validation_size=0.0):
        return benchmark, None

def get_benchmark(config, validation_size=0.1):
    """Factory function to get the specified benchmark."""
    if config.dataset == 'split_cifar100':
        return get_split_cifar100_benchmark(config, validation_size)
    elif config.dataset == 'cifar6':
        return get_cifar6_benchmark(config, validation_size)
    elif config.dataset == 'permuted_mnist':
        return get_permuted_mnist_benchmark(config, validation_size)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

def get_split_cifar100_benchmark(config, validation_size):
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    benchmark = SplitCIFAR100(
        n_experiences=config.num_tasks,
        fixed_class_order=list(range(100)),
        seed=config.seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_root=config.dataset_root,
    )
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None

def get_cifar6_benchmark(config, validation_size):
    class_order = [0, 2, 1, 3, 5, 9]
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    benchmark = SplitCIFAR10(
        n_experiences=config.num_tasks,
        fixed_class_order=class_order,
        seed=config.seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_root=config.dataset_root,
    )
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None

def get_permuted_mnist_benchmark(config, validation_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    benchmark = PermutedMNIST(
        n_experiences=config.num_tasks, 
        seed=config.seed, 
        train_transform=transform,
        eval_transform=transform,
        dataset_root=config.dataset_root
    )
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None
