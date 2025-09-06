"""src/preprocess.py
Data-loading & preprocessing utilities decoupled from the training logic so that
other learners (e.g. NLP) can reuse them.
"""
from __future__ import annotations

import os, tarfile, urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import Subset, DataLoader, random_split
import torch
from PIL import Image


class MiniCtrl40Stream:
    """Deterministic 40-task ImageNet stream (disjoint class partitions)."""
    def __init__(self, cfg, batch_size: int):
        self.cfg = cfg  # cfg is AttrDict coming from YAML
        root_var = cfg.imagenet_root_env
        if root_var not in os.environ:
            raise RuntimeError(
                f"[FATAL] Environment variable {root_var} not defined. Set it to the path "
                "of the extracted ImageNet ILSVRC2012 train split."
            )
        self.root = Path(os.environ[root_var])
        if not (self.root / "n01440764").exists():  # simple sanity check
            raise RuntimeError(f"{self.root} does not look like a valid ImageNet directory.")

        self.bs = batch_size
        self.transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tasks: List[Dict[str, Any]] = self._construct_tasks()

    # ------------------------------------------------------------------
    def _construct_tasks(self):
        manifest = self.cfg.ctrl40_split
        synset_dirs = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        idx2syn = {i: s for i, s in enumerate(synset_dirs)}
        tasks = []
        for class_ids in manifest["tasks"]:
            synsets = [idx2syn[i] for i in class_ids]
            full_ds = ImageFolder(self.root, transform=self.transform)
            task_indices = [i for i, (p, lbl) in enumerate(full_ds.samples) if full_ds.classes[lbl] in synsets]
            subset = Subset(full_ds, task_indices)
            sizes = [int(0.625 * len(subset)), int(0.125 * len(subset))]
            sizes.append(len(subset) - sum(sizes))
            train_ds, val_ds, test_ds = random_split(subset, sizes)
            tasks.append({"train": train_ds, "val": val_ds, "test": test_ds, "n_classes": 10})
        return tasks

    # ------------------------------------------------------------------
    def loader(self, split):
        return DataLoader(split, batch_size=self.bs, shuffle=True, num_workers=4, pin_memory=True)


class CIFARSmoke:
    """Two-task CIFAR-10 ↔ CIFAR-10-C stream used as a lightweight sanity check."""
    def __init__(self, cfg, batch_size: int):
        self.cfg, self.bs = cfg, batch_size
        data_root = Path(cfg.paths.data_dir) / "cifar10"
        data_root.mkdir(parents=True, exist_ok=True)

        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        resize = T.Resize(224)

        # clean CIFAR-10 --------------------------------------------------------
        self.clean = CIFAR10(data_root, train=True, download=True, transform=T.Compose([resize, T.ToTensor(), norm]))

        # corrupted CIFAR-10-C ---------------------------------------------------
        c_tar = Path(cfg.paths.data_dir) / "CIFAR-10-C.tar"
        if not c_tar.exists():
            print("[Download] CIFAR-10-C …")
            urllib.request.urlretrieve(cfg.cifar_c_url, c_tar)
        with tarfile.open(c_tar) as tar:
            tar.extractall(path=cfg.paths.data_dir)
        c_data = np.load(Path(cfg.paths.data_dir) / "CIFAR-10-C" / "gaussian_noise.npy")
        c_labels = np.load(Path(cfg.paths.data_dir) / "CIFAR-10-C" / "labels.npy")
        start, end = 30000 * 2, 30000 * 3  # severity 3 slice
        samples = [(Image.fromarray(img), int(lbl)) for img, lbl in zip(c_data[start:end], c_labels[start:end])]

        class _Arr(torch.utils.data.Dataset):
            def __init__(self, arr):
                self.arr = arr
            def __len__(self):
                return len(self.arr)
            def __getitem__(self, idx):
                im, lbl = self.arr[idx]
                im = T.Compose([resize, T.ToTensor(), norm])(im)
                return im, lbl

        corrupt_ds = _Arr(samples)
        self.tasks = [
            {"train": self.clean, "val": None, "test": None, "n_classes": 10},
            {"train": corrupt_ds, "val": None, "test": None, "n_classes": 10},
        ]

    # ------------------------------------------------------------------
    def loader(self, split):
        return DataLoader(split, batch_size=self.bs, shuffle=True, num_workers=2, pin_memory=True)
