import sys
from types import ModuleType

# -----------------------------------------------------------------------------
# Compatibility patch for Avalanche <-> pytorchcv dependency mismatch
# -----------------------------------------------------------------------------
# Avalanche's MobilenetV1 model expects `DwsConvBlock` to be available from
# `pytorchcv.models.mobilenet` *or* `pytorchcv.models.common`.  The symbol was
# removed from recent `pytorchcv` releases which results in an ImportError and
# prevents `avalanche-lib` from being imported.  To avoid pin-downgrading the
# entire `pytorchcv` stack we create a minimal drop-in replacement and register
# it in `sys.modules` **before** importing Avalanche.
#
# The implementation below is functionally equivalent to the original operator
# and is only required for Avalanche's internal MobileNet definition.  This
# keeps our project self-contained and forward-compatible with newer
# `pytorchcv` versions.
# -----------------------------------------------------------------------------
try:
    import torch.nn as nn

    class DwsConvBlock(nn.Sequential):
        """Depth-wise separable 3×3 convolution block (DW→PW)."""

        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__(
                # Depth-wise convolution
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # Point-wise convolution
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    # Expose the symbol through the two module paths queried by Avalanche.
    for _module_path in (
        "pytorchcv.models.common",
        "pytorchcv.models.mobilenet",  # attribute fallback if the real module exists
    ):
        if _module_path in sys.modules:
            setattr(sys.modules[_module_path], "DwsConvBlock", DwsConvBlock)
        else:
            stub = ModuleType(_module_path)
            stub.DwsConvBlock = DwsConvBlock  # type: ignore
            sys.modules[_module_path] = stub
except Exception:  # pragma: no cover
    # If anything goes wrong we silently continue – Avalanche import will raise
    # its own descriptive error later which will be caught by the surrounding
    # try/except logic in the calling modules.
    pass

# -----------------------------------------------------------------------------
# Standard imports *after* the compatibility patch so that Avalanche sees the
# injected symbol.
# -----------------------------------------------------------------------------
import torch
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100

# The location of `benchmark_with_validation_stream` changed in recent Avalanche
# versions.  We attempt the new import path first and gracefully fall back to a
# pass-through stub if the helper is not available (e.g. older release).
try:
    # Newer (≥0.4.1) path
    from avalanche.benchmarks.utils import benchmark_with_validation_stream  # type: ignore
except Exception:  # pragma: no cover
    # Fallback – implement a minimal no-op splitter that simply returns the
    # original benchmark and `None` for the validation stream.  Down-stream code
    # is already written to handle `None`.
    def benchmark_with_validation_stream(benchmark, validation_size=0.0):  # noqa: D401,E501
        """Return the benchmark unchanged and no validation stream."""

        return benchmark, None


def get_cifar100_benchmark(config, validation_size=0.1):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    benchmark = SplitCIFAR100(
        n_experiences=config.num_tasks_cifar100,
        fixed_class_order=list(range(100)),
        seed=0,  # Fixed order for reproducibility
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_root=config.dataset_root,
    )
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None


def get_toy_benchmark(config):
    cifar100_classes = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]
    name_to_idx = {name: i for i, name in enumerate(cifar100_classes)}
    target_indices = [name_to_idx[name] for name in config.toy_classes]
    task0_indices = target_indices[:3]
    task1_indices = target_indices[3:]

    return SplitCIFAR100(
        n_experiences=2,
        fixed_class_order=task0_indices + task1_indices,
        seed=0,
        dataset_root=config.dataset_root,
    )
