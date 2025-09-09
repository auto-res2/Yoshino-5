import random
import sys
from torch.utils.data import IterableDataset
from torchvision import transforms

# Attempt to import Avalanche, provide helpful errors if missing
try:
    import avalanche as avl
except ImportError:
    print("Avalanche-lib not found. Please install with 'pip install avalanche-lib'")
    sys.exit(1)


def get_transforms():
    """ [IMPLEMENTED] Component: Exact Preprocessing Pipelines """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, test_transform


def get_benchmark(dataset_config, train_transform, test_transform):
    """ [IMPLEMENTED] Component: Complete Dataset Suite (Avalanche Benchmarks) """
    name = list(dataset_config.keys())[0]
    params = dataset_config[name]
    print(f"Loading benchmark: {name}")
    if name == "split_cifar100":
        return avl.benchmarks.SplitCIFAR100(
            n_experiences=params['n_experiences'],
            train_transform=train_transform, eval_transform=test_transform,
            shuffle=True
        )
    elif name == "permutted_mnist":  # note: key kept for backward-compatibility
        return avl.benchmarks.PermutedMNIST(
            n_experiences=params['n_experiences'],
            train_transform=train_transform, eval_transform=test_transform
        )
    elif name == "split_tiny_imagenet":
        # NOTE: TinyImageNet must be downloaded manually to `~/.avalanche/data/tiny-imagenet-200/`
        return avl.benchmarks.SplitTinyImageNet(
            n_experiences=params['n_experiences'],
            train_transform=train_transform, eval_transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


class CIFAR100BlurStream(IterableDataset):
    """ [IMPLEMENTED] Component: Fuzzy Task Boundary Stream (CIFAR-100-Blur) """
    def __init__(self, config, train_transform):
        self.config = config
        self.transform = train_transform
        cifar_train = avl.benchmarks.datasets.CIFAR100(root="./data", train=True, download=True)
        self.data = [cifar_train[i] for i in range(len(cifar_train))]
        self.step = 0
        self.class_window_start = 0

    def __iter__(self):
        random.shuffle(self.data)
        for img, label, _ in self.data:
            # Check if label is in the current active window
            if self.class_window_start <= label < self.class_window_start + self.config['window_width']:
                self.step += 1
                if self.step % self.config['shift_every_steps'] == 0:
                    self.class_window_start = (self.class_window_start + 2) % (100 - self.config['window_width'] + 1)

                if self.transform:
                    img = self.transform(img)
                yield img, label
