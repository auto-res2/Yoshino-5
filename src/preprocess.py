"""src/preprocess.py
----------------------------------------------------------------------
Dataset loading & preprocessing utilities.
Only NLP (Natural-Instructions) branch is included; Vision branch kept
for completeness.
"""
from __future__ import annotations

import os
from typing import List, Dict

import datasets as hfds
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ------------------------------------------------------------------
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ------------------------------------------------------------------
# NLP  – Natural-Instructions single-task wrapper
# ------------------------------------------------------------------
class NaturalInstructionTask(Dataset):
    """HF Natural-Instructions wrapper providing tokenised samples."""

    def __init__(self, task_name: str, split: str, tokenizer, max_len: int = 256):
        self.task_name = task_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ds = hfds.load_dataset(
            "Muennighoff/natural-instructions",
            task_name,
            split=split,
            cache_dir=CACHE_DIR,
        )

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        target = item.get("output", "")
        enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        tgt = self.tokenizer(
            target, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

# ------------------------------------------------------------------
# Collate helpers
# ------------------------------------------------------------------

def collate_fn_lm(batch, pad_id: int):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        collated[k] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_id
        )
    return collated

# ------------------------------------------------------------------
# Vision  – ImageNet-R subset (optional)
# ------------------------------------------------------------------
from torchvision import datasets as tvd  # noqa: E402

class ImageNetRTask(Dataset):
    def __init__(self, class_list: List[str], split: str, transform):
        hf_ds = hfds.load_dataset("axiong/imagenet-r", split=split, cache_dir=CACHE_DIR)
        hf_ds = hf_ds.filter(lambda x: x["label"] in class_list)
        self.ds = hf_ds
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(sorted(class_list))}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        label = self.class_to_idx[ex["label"]]
        return self.transform(img), label

# Default transforms ------------------------------------------------
cv_train_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandAugment(num_ops=2, magnitude=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cv_test_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
