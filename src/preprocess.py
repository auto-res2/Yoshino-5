from importlib import import_module

# -----------------------------------------------------------------------------
# Monkey-patch Torch < 2.2 for compatibility with HuggingFace/Transformers >= 4.56
# -----------------------------------------------------------------------------
# Starting with Transformers 4.56 the library expects the public symbols
# `register_pytree_node` and `register_pytree_node_class` to be available from
# `torch.utils._pytree`.  These were only promoted from their internal
# counterparts in PyTorch 2.2.  When we run with the 2.1 series we therefore add
# lightweight shims that delegate to the internal implementations (or become
# harmless no-ops).  The patch must execute *before* `torchvision` is imported
# because `torchvision -> torch.onnx -> transformers` triggers the import chain
# that requires these symbols.
# -----------------------------------------------------------------------------
import torch.utils._pytree as _pytree  # noqa: E402  (import after top docstring)


def _safe_call_register_pytree_node(type_, flatten_func, unflatten_func, **kwargs):
    """Call the private Torch < 2.2 implementation while discarding new kwargs.

    Newer versions of Transformers pass keyword-only arguments such as
    `serialized_type_name` and `version`.  The internal implementation present
    in Torch 2.1 (``_register_pytree_node``) does *not* accept them, which would
    otherwise raise a ``TypeError``.  We therefore strip any unexpected kwargs
    before forwarding the call.  When running on Torch ≥ 2.2 the public
    ``register_pytree_node`` already exists and no shim is needed.
    """
    unsupported_keys = {k: kwargs.pop(k) for k in list(kwargs.keys())}
    if unsupported_keys:
        # Silently ignore unsupported keys – they are only metadata for ONNX ↔
        # Pytree round-tripping and are not required for basic functionality.
        pass
    return _pytree._register_pytree_node(type_, flatten_func, unflatten_func)  # type: ignore[attr-defined]


# Expose alias for `register_pytree_node` if missing (present as
# `_register_pytree_node` in < 2.2).
if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
    _pytree.register_pytree_node = _safe_call_register_pytree_node  # type: ignore[attr-defined]


# Provide a best-effort stub for `register_pytree_node_class` so that the import
# does not fail.  The decorator form used by Transformers simply returns the
# class unchanged when Torch lacks first-class support.  That behaviour is
# replicated here.
if not hasattr(_pytree, "register_pytree_node_class"):

    def _register_pytree_node_class(cls=None, *, flatten=None, unflatten=None):  # noqa: N802
        """No-op stand-in matching the signature used by Transformers."""
        # When used as ``@register_pytree_node_class()`` it returns a decorator;
        # when used directly the class is passed as the first positional arg.
        if cls is None:
            # Called as a decorator with parentheses.
            def decorator(c):
                return c

            return decorator
        # Called directly without parentheses.
        return cls

    _pytree.register_pytree_node_class = _register_pytree_node_class  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Standard library imports that rely on the patched symbols follow below.
# -----------------------------------------------------------------------------
import torch  # noqa: E402
from torchvision import transforms  # noqa: E402
from avalanche.benchmarks.classic import (
    SplitCIFAR100,
    PermutedMNIST,
    SplitCIFAR10,
)  # noqa: E402

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

    def benchmark_with_validation_stream(benchmark, validation_size=0.0):  # type: ignore
        return benchmark, None


def get_benchmark(config, validation_size=0.1):
    """Factory function to get the specified benchmark."""
    if config.dataset == "split_cifar100":
        return get_split_cifar100_benchmark(config, validation_size)
    elif config.dataset == "cifar6":
        return get_cifar6_benchmark(config, validation_size)
    elif config.dataset == "permuted_mnist":
        return get_permuted_mnist_benchmark(config, validation_size)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


def _make_transforms(mean, std, train):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_split_cifar100_benchmark(config, validation_size):
    train_transform = _make_transforms((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), True)
    eval_transform = _make_transforms((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), False)
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
    train_transform = _make_transforms((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), True)
    eval_transform = _make_transforms((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), False)
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
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    benchmark = PermutedMNIST(
        n_experiences=config.num_tasks,
        seed=config.seed,
        train_transform=transform,
        eval_transform=transform,
        dataset_root=config.dataset_root,
    )
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None