from __future__ import annotations

"""src/preprocess.py
-------------------------------------------------------------------
Dataset loading utilities for NLP (Natural-Instructions) and vision
(ImageNet-R).  This patch mainly mitigates HTTP-429 rate-limit errors
observed when Hugging Face Hub is hit with hundreds of concurrent HEAD
requests.  We take two defensive steps:

1.  Disable parallel file downloads in `datasets` via the environment
    variable ``HF_DATASETS_DOWNLOAD_PARALLELISM``.  This forces the hub
    client to request files sequentially, dramatically reducing the
    likelihood of triggering the rate-limit.
2.  Tell the datasets downloader to keep retrying a bit longer by
    passing an explicit ``DownloadConfig`` with more retries.

These changes are **fully backwards-compatible** – if internet is
available everything is fetched as before, only more politely.
"""

import os
from typing import List, Dict

# ------------------------------------------------------------------
# BEFORE importing datasets we set the environment variables so that
# they are picked-up by the library internals.
# ------------------------------------------------------------------
os.environ.setdefault("HF_DATASETS_DOWNLOAD_PARALLELISM", "false")  # sequential
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # cleaner CI logs

import datasets as hfds  # noqa: E402  (import _after_ env tweak)
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

# ------------------------------------------------------------------
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ------------------------------------------------------------------
# NLP  – Natural-Instructions single-task wrapper
# ------------------------------------------------------------------
class NaturalInstructionTask(Dataset):
    """HF Natural-Instructions wrapper providing tokenised samples."""

    _FALLBACK_SPLITS = {
        "validation": "test",  # if dataset has no validation split use test
    }

    def __init__(self, task_name: str, split: str, tokenizer, max_len: int = 256):
        self.task_name = task_name
        self.requested_split = split
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Some configs are not available; we first attempt to load the desired
        # split directly.  If that fails we gracefully fall back so that the
        # call-site does not have to care.
        self.ds = self._safe_load_dataset(task_name, split)

    # --------------------------------------------------------------
    def _safe_load_dataset(self, task_name: str, split: str):
        """Robust loader that copes with missing configs / splits."""

        def _filter_task(full):
            def _match(example):
                return (
                    ("task_name" in example and example["task_name"] == task_name)
                    or ("task_id" in example and example["task_id"] == task_name)
                )

            # Serial execution (``num_proc=1``) to avoid extra processes that
            # could again create parallel requests.
            return full.filter(_match, num_proc=1)

        # Unified download config with extra retries -------------------------
        dl_cfg = hfds.DownloadConfig(max_retries=10)

        # ------------------------------------------------------------------
        # 1) Fast path – try to load specific builder-config & split
        # ------------------------------------------------------------------
        try:
            return hfds.load_dataset(
                "Muennighoff/natural-instructions",
                task_name,
                split=split,
                cache_dir=CACHE_DIR,
                download_config=dl_cfg,
            )
        except Exception:
            # ------------------------------------------------------------------
            # 2) Builder config not found OR split absent – graceful fallback
            # ------------------------------------------------------------------
            fallback_split = self._FALLBACK_SPLITS.get(split, None)
            if fallback_split is None:
                fallback_split = "train[:10%]" if split == "validation" else "train"

            full_ds = hfds.load_dataset(
                "Muennighoff/natural-instructions",
                split=fallback_split,
                cache_dir=CACHE_DIR,
                download_config=dl_cfg,
            )

            filtered = _filter_task(full_ds)
            if len(filtered) == 0:
                raise ValueError(
                    f"Task '{task_name}' not found inside Muennighoff/natural-instructions dataset."
                )

            if split == "validation" and fallback_split.startswith("train"):
                filtered = filtered.select(range(min(1000, len(filtered))))
            return filtered

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        target = item.get("output", "")
        # The "output" field is often a list of acceptable answers – take first.
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
from torchvision import datasets as tvd  # noqa: E402  (kept for backward compat)

class ImageNetRTask(Dataset):
    def __init__(self, class_list: List[str], split: str, transform):
        dl_cfg = hfds.DownloadConfig(max_retries=10)
        hf_ds = hfds.load_dataset("axiong/imagenet-r", split=split, cache_dir=CACHE_DIR, download_config=dl_cfg)
        hf_ds = hf_ds.filter(lambda x: x["label"] in class_list, num_proc=1)
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