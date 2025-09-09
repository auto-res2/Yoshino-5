import torch
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.generators import benchmark_with_validation_stream

def get_cifar100_benchmark(config, validation_size=0.1):
    train_transform = transforms.Compose([
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
    benchmark = SplitCIFAR100(n_experiences=config.num_tasks_cifar100, 
                              fixed_class_order=list(range(100)),
                              seed=0, # Use fixed order for comparability
                              train_transform=train_transform, 
                              eval_transform=eval_transform, 
                              dataset_root=config.dataset_root)
    if validation_size > 0:
        return benchmark_with_validation_stream(benchmark, validation_size=validation_size)
    return benchmark, None

def get_toy_benchmark(config):
    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    name_to_idx = {name: i for i, name in enumerate(cifar100_classes)}
    target_indices = [name_to_idx[name] for name in config.toy_classes]
    task0_indices = target_indices[:3]
    task1_indices = target_indices[3:]

    return SplitCIFAR100(n_experiences=2,
                         fixed_class_order=task0_indices + task1_indices,
                         seed=0,
                         dataset_root=config.dataset_root)
