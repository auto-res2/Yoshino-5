"""src/preprocess.py
Data loading, tiny synthetic smoke-test generation, download helpers.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader

# external tokenizer only needed during CSV generation
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
NI_HF_NAME = "Muennighoff/natural-instructions"
IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  Tiny sentiment dataset used for unit/CPU smoke tests (EXP-1)
# -----------------------------------------------------------------------------

class TinySentimentSmoke(torch.utils.data.Dataset):
    """Pre-tokenised sentiment dataset stored in CSV (see ensure_smoke_csv)."""

    LABEL_VOCAB = {"positive": 0, "negative": 1, "neutral": 2}

    def __init__(self, csv_file: Path):
        import pandas as pd  # local import to avoid heavyweight dep at import time
        df = pd.read_csv(csv_file)
        self.inputs = torch.from_numpy(np.stack(df["input_ids"].apply(eval))).long()
        self.attn = torch.from_numpy(np.stack(df["attention_mask"].apply(eval))).long()
        self.labels = torch.from_numpy(np.stack(df["labels"].apply(eval))).long()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.attn[idx],
            "labels": self.labels[idx],
        }


# -----------------------------------------------------------------------------
#  CSV generation for the smoke test – deterministic, no network required
# -----------------------------------------------------------------------------

def ensure_smoke_csv(tokenizer_name: str, tasks: list[str]) -> Dict[str, Path]:
    """Generate deterministic CSVs with token IDs so EXP-1 can run offline."""
    import pandas as pd

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    csv_paths: Dict[str, Path] = {}
    for task in tasks:
        fn = DATA_DIR / f"{task}.csv"
        csv_paths[task] = fn
        if fn.exists():
            continue
        rows = []
        rng = np.random.default_rng(int.from_bytes(task.encode(), "little") & 0xFFFF)
        labels = list(TinySentimentSmoke.LABEL_VOCAB)
        for i in range(200):
            lab = rng.choice(labels)
            text = f"This sample {i} is a {lab} example."
            enc = tok(text, truncation=True, max_length=64)
            rows.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": [TinySentimentSmoke.LABEL_VOCAB[lab]],
            })
        pd.DataFrame(rows).to_csv(fn, index=False)
    return csv_paths


# -----------------------------------------------------------------------------
#  Natural-Instructions loader (cached on disk to avoid repeated downloads)
# -----------------------------------------------------------------------------

def load_ni_task(task_name: str) -> DatasetDict:
    cache_path = DATA_DIR / "ni_cache" / task_name
    if cache_path.exists():
        return load_from_disk(cache_path)
    ds = load_dataset(NI_HF_NAME, task_name, trust_remote_code=True)
    ds.save_to_disk(cache_path)
    return ds


# -----------------------------------------------------------------------------
#  (Very) lightweight ImageNet-R subset builder – keeps dependencies minimal
# -----------------------------------------------------------------------------

def _download_imagenet_r(dest: Path) -> Path:
    tar_path = dest / "imagenet-r.tar"
    if not tar_path.exists():
        print("[DL] Downloading ImageNet-R (~500 MB)…")
        import requests, tqdm as tq
        with requests.get(IMAGENET_R_URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(tar_path, "wb") as f, tq.tqdm(total=total, unit="B", unit_scale=True) as p:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        p.update(len(chunk))
    img_root = dest / "imagenet-r"
    if not img_root.exists():
        import tarfile
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest)
    return img_root


def build_imagenet_r_subset(idx: int) -> DatasetDict:
    """Return a 25-class subset of ImageNet-R as HuggingFace DatasetDict."""
    img_root = _download_imagenet_r(DATA_DIR)
    classes = sorted({p.parent.name for p in img_root.rglob("*.png")})[idx * 25 : (idx + 1) * 25]
    rows_train, rows_test = [], []
    for cls in classes:
        for img_fp in (img_root / cls).glob("*.png"):
            (rows_train if random.random() < 0.8 else rows_test).append({"image": str(img_fp), "label": cls})
    return DatasetDict({
        "train": Dataset.from_list(rows_train),
        "test": Dataset.from_list(rows_test),
    })
