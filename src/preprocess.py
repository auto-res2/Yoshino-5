"""
preprocess.py – data utilities (seed, synthetic shards, datasets, collation)
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# 0.  Reproducibility helpers
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED_SEQUENCE = [11, 29, 42]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 1.  Synthetic tiny Natural-Instructions shards (offline smoke test)
# -----------------------------------------------------------------------------

SPLITS = ["train", "validation", "test"]
N_PER_SPLIT = {"train": 140, "validation": 30, "test": 30}


def ensure_synthetic_data(tasks: List[str]):
    """Create small deterministic CSV shards for given NI task IDs."""
    for tid in tasks:
        for split in SPLITS:
            shard = DATA_DIR / f"{tid}-{split}.csv"
            if shard.exists():
                continue
            rng = np.random.default_rng(hash((tid, split)) & 0xFFFF)
            with shard.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["instruction", "input", "output"])
                for idx in range(N_PER_SPLIT[split]):
                    instr = f"[{tid}] Please classify the sentiment of the sentence below."
                    inp = f"Dummy example {idx} about cats {rng.integers(0,100)}"
                    label = rng.choice(["positive", "negative", "neutral"])
                    w.writerow([instr, inp, label])

# -----------------------------------------------------------------------------
# 2.  Dataset & collator
# -----------------------------------------------------------------------------

class MiniNIDataset(Dataset):
    """CSV → tokenised tensors; used only for EXP-1."""

    def __init__(self, task: str, split: str, tokenizer, max_len: int):
        fname = DATA_DIR / f"{task}-{split}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Required shard {fname} not found – run ensure_synthetic_data first."
            )
        self.rows = list(csv.DictReader(fname.open()))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        enc = self.tokenizer(
            r["instruction"] + "\n" + r["input"],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            r["output"], truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": tgt.input_ids.squeeze(0),
        }


def collate_pad(batch, pad_id: int):
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        out[k] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_id
        )
    return out
