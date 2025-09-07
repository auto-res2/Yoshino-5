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

        # ------------------------------------------------------------------
        # The Muennighoff/natural-instructions dataset exposes all tasks under
        # a single "default" config (no per-task BuilderConfig).  Attempting
        # to pass the task name as the config therefore raises the ValueError
        # observed in the logs.  We handle this gracefully by first trying the
        # user-requested config and, if that fails, falling back to the default
        # config and filtering for the desired task.
        # ------------------------------------------------------------------
        try:
            # Newer versions of the dataset may add per-task configs – keep the
            # fast path so our code continues to work if that happens.
            self.ds = hfds.load_dataset(
                "Muennighoff/natural-instructions",
                task_name,  # attempted config name
                split=split,
                cache_dir=CACHE_DIR,
            )
        except ValueError:
            # Fallback path – load the full dataset once and slice.
            full_ds = hfds.load_dataset(
                "Muennighoff/natural-instructions",
                split=split,
                cache_dir=CACHE_DIR,
            )

            # The dataset stores the task identifier under the key "task_name"
            # (example: "ni2002").  If this key is absent we also check the
            # classic Natural-Instructions "task_id" field for robustness.
            def _match(example):
                return (
                    ("task_name" in example and example["task_name"] == task_name)
                    or ("task_id" in example and example["task_id"] == task_name)
                )

            filtered = full_ds.filter(_match)
            if len(filtered) == 0:
                raise ValueError(
                    f"Task '{task_name}' not found inside Muennighoff/natural-instructions dataset."
                )
            self.ds = filtered

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        target = item.get("output", "")
        # The "output" field is often a list of acceptable answers –
        # use the first one if that is the case.
        if isinstance(target, list):
            target = target[0] if len(target) > 0 else ""
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
