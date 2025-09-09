"""Dataset and transformation helpers with a PyTorch⇄transformers compatibility shim."""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Compatibility patch for PyTorch <2.2  vs.  transformers ≥4.37
# -----------------------------------------------------------------------------
# transformers≥4.37 expects `torch.utils._pytree.register_pytree_node` **with** a
# `serialized_type_name` kw-arg.  In PyTorch≤2.1 this function is still called
# *_register_pytree_node* and **lacks** that parameter, breaking import.
#
# We monkey-patch at import-time *before* torchvision / transformers are pulled
# in so that the symbol exists with a flexible signature that ignores the extra
# argument when the underlying implementation does not support it.
# -----------------------------------------------------------------------------
import inspect
from typing import Any, Callable

try:
    from torch.utils import _pytree as _torch_pytree  # type: ignore

    original_impl: Callable | None = getattr(
        _torch_pytree,
        '_register_pytree_node',
        getattr(_torch_pytree, 'register_pytree_node', None),
    )

    if original_impl is not None:

        def _safe_register_pytree_node(cls: type, flatten_func: Callable, unflatten_func: Callable, **kwargs: Any):  # noqa: E501
            """Wrapper that forwards to the original impl after sanitising kwargs."""
            if (
                'serialized_type_name' in kwargs
                and 'type_name' not in inspect.signature(original_impl).parameters
            ):
                kwargs.pop('serialized_type_name')  # not supported in older Torch
            return original_impl(cls, flatten_func, unflatten_func, **kwargs)  # type: ignore[arg-type]

        # Expose both names so any caller variant works.
        _torch_pytree.register_pytree_node = _safe_register_pytree_node  # type: ignore[attr-defined]
except Exception:  # pragma: no cover – defensive import
    pass

# -----------------------------------------------------------------------------
# Safe to import heavy deps now (they may transitively import transformers).
# -----------------------------------------------------------------------------
import torch  # noqa: E402 – required by other modules
from torchvision import transforms  # noqa: E402
from avalanche.benchmarks.classic import SplitCIFAR100, PermutedMNIST  # noqa: E402

# -----------------------------------------------------------------------------
# `benchmark_with_validation_stream` is *not* available in some Avalanche
# versions (e.g. 0.6.0).  We therefore try to import it and, if unsuccessful,
# fall back to a no-op shim so downstream code keeps working.
# -----------------------------------------------------------------------------
try:
    from avalanche.benchmarks.utils import benchmark_with_validation_stream  # type: ignore
except (ImportError, AttributeError):

    def benchmark_with_validation_stream(benchmark, validation_size=0.0, **kwargs):  # type: ignore
        """Fallback stub: returns *benchmark* unchanged when the real helper is missing.

        The main training loop never consumes the validation stream directly, so
        for the purposes of unit-/integration-tests this no-op suffices. A clear
        warning is printed so that users are aware that no dedicated validation
        split was created.
        """
        if validation_size:
            print(
                "[WARN] `benchmark_with_validation_stream` unavailable in the "
                "installed Avalanche version – proceeding without a dedicated "
                "validation stream."
            )
        return benchmark

# -----------------------------------------------------------------------------
# Normalisation constants
# -----------------------------------------------------------------------------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


# -----------------------------------------------------------------------------
# Transform helpers
# -----------------------------------------------------------------------------

def _get_transforms(dataset_name: str, image_size: int, is_train: bool):
    if 'cifar' in dataset_name:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.15)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    elif 'mnist' in dataset_name:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        )
    raise ValueError(f"Transforms for dataset {dataset_name} not defined.")


# -----------------------------------------------------------------------------
# Benchmark factory
# -----------------------------------------------------------------------------

def get_benchmark(config):
    """Return an Avalanche benchmark according to *config*."""
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

    # Optionally add a validation stream.
    if getattr(config, 'validation_size', 0) and config.validation_size > 0:
        benchmark = benchmark_with_validation_stream(
            benchmark, validation_size=config.validation_size
        )

    return benchmark
