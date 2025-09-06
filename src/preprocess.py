"""src/preprocess.py
Data loading & preprocessing for the Imagenette vision stream and a subset of
FLAN NLP tasks.  Only the ImagenetteTasks class is used in the condensed
example; FlanMiniTasks is retained for completeness.
"""
from __future__ import annotations

import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Dict

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, default_data_collator

# -----------------------------------------------------------------------------
#                              Imagenette Tasks
# -----------------------------------------------------------------------------

class ImagenetteTasks:
    """Public proxy for Mini-CTrL-40: the 10 Imagenette classes are each split
    into 4 shards → 40 sequential tasks. The whole dataset is downloaded and
    extracted on-demand into *root*.
    """

    def __init__(self, root: Path, download_url: str, batch_size: int):
        self.root = root
        self.download_url = download_url
        self.batch_size = batch_size
        self._ensure_data()
        self.transforms = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tasks = self._build_tasks()

    # ------------------------------------------------------------------
    def _ensure_data(self):
        if (self.root / "imagenette2-160" / "train").exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        archive = self.root / "imagenette2.tgz"
        if not archive.exists():
            print("[Download] Imagenette …")
            urllib.request.urlretrieve(self.download_url, archive)
        print("[Extract] Imagenette …")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=self.root)

    # ------------------------------------------------------------------
    def _build_tasks(self):
        dataset = ImageFolder(self.root / "imagenette2-160" / "train", transform=self.transforms)
        indices_per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            indices_per_class[label].append(idx)
        tasks = []
        for cls in range(10):
            idxs = indices_per_class[cls]
            random.shuffle(idxs)
            shard_size = len(idxs) // 4
            for s in range(4):
                shard = idxs[s * shard_size : (s + 1) * shard_size]
                subset = torch.utils.data.Subset(dataset, shard)
                train_len = int(0.7 * len(subset))
                val_len = int(0.15 * len(subset))
                test_len = len(subset) - train_len - val_len
                train_ds, val_ds, test_ds = random_split(subset, [train_len, val_len, test_len])
                tasks.append({"train": train_ds, "val": val_ds, "test": test_ds, "n_classes": 10})
        print(f"[ImagenetteTasks] Built {len(tasks)} tasks (proxy for Mini-CTrL-40)")
        return tasks

    # ------------------------------------------------------------------
    def dataloader_for(self, split_ds):
        return DataLoader(
            split_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

# -----------------------------------------------------------------------------
#                                 FLAN Mini
# -----------------------------------------------------------------------------

class FlanMiniTasks:
    """Tiny utility to load a set of FLAN tasks via HF Datasets."""

    def __init__(self, task_names: List[str], batch_size: int, max_len: int = 256):
        self.task_names = task_names
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tasks = self._load_tasks()

    # ------------------------------------------------------------------
    def _tokenise(self, examples):
        return self.tokenizer(
            examples["inputs"],
            text_target=examples["targets"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )

    # ------------------------------------------------------------------
    def _load_tasks(self):
        tasks = []
        for name in self.task_names:
            ds_dict: DatasetDict = load_dataset(name)
            if {"train", "validation", "test"}.issubset(ds_dict):
                train_ds = ds_dict["train"].map(self._tokenise, batched=True)
                val_ds = ds_dict["validation"].map(self._tokenise, batched=True)
                test_ds = ds_dict["test"].map(self._tokenise, batched=True)
            else:
                full = ds_dict["train"].map(self._tokenise, batched=True)
                train_len = int(0.7 * len(full))
                val_len = int(0.15 * len(full))
                test_len = len(full) - train_len - val_len
                train_ds, val_ds, test_ds = random_split(full, [train_len, val_len, test_len])
            tasks.append({"train": train_ds, "val": val_ds, "test": test_ds, "n_classes": None})
        print(f"[FlanMiniTasks] Loaded {len(tasks)} NLP tasks (FLAN-Mini proxy)")
        return tasks

    # ------------------------------------------------------------------
    def dataloader_for(self, split_ds):
        return DataLoader(
            split_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=2,
            pin_memory=True,
        )
