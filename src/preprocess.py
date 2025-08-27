import os
import json
import random
import pathlib
import gzip
from typing import List, Dict, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

class SAPTraceDataset(Dataset):
    def __init__(self, jsonl_file: str, max_records: Optional[int] = None):
        self._path = pathlib.Path(jsonl_file)
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        
        self._fp = gzip.open(jsonl_file, "rt") if jsonl_file.endswith(".gz") else open(jsonl_file)
        self._index = []
        offset = 0
        for ln, line in enumerate(self._fp):
            self._index.append(offset)
            offset += len(line.encode())
            if max_records and ln + 1 >= max_records:
                break
        self._fp.close()
        self._fp = None

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if self._fp is None:
            self._fp = gzip.open(self._path, "rt") if str(self._path).endswith(".gz") else open(self._path)
        self._fp.seek(self._index[idx])
        rec = json.loads(self._fp.readline())
        return {
            "input_ids": torch.tensor(rec["input_ids"], dtype=torch.long),
            "labels": torch.tensor(rec["labels"], dtype=torch.long),
            "z_t": torch.tensor(rec["z_t"], dtype=torch.float32),
            "m_t": torch.tensor(rec["m_t"], dtype=torch.float32),
            "p_t": torch.tensor(rec["p_t"], dtype=torch.float32)
        }

class TraceCollator:
    def __call__(self, batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_id = 0
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        
        for i, ex in enumerate(batch):
            L = ex["input_ids"].size(0)
            input_ids[i, :L] = ex["input_ids"]
            labels[i, :L] = ex["labels"]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "z_t": [x["z_t"] for x in batch],
            "m_t": [x["m_t"] for x in batch],
            "p_t": [x["p_t"] for x in batch],
        }

def generate_synthetic_data(num_samples: int = 256, min_len: int = 8, max_len: int = 32, 
                          trace_dim: int = 64, mask_dim: int = 128, vocab_size: int = 1000,
                          output_file: str = "synthetic_data.jsonl"):
    data = []
    for _ in range(num_samples):
        L = random.randint(min_len, max_len)
        data.append({
            "input_ids": [random.randint(1, vocab_size-1) for _ in range(L)],
            "labels": [random.randint(1, vocab_size-1) for _ in range(L)],
            "z_t": (np.random.randn(L, trace_dim) * 0.1).tolist(),
            "m_t": (np.random.randint(0, 2, size=(L, mask_dim))).tolist(),
            "p_t": (np.random.randn(L, vocab_size) * 0.01).tolist()
        })
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as fp:
        for rec in data:
            fp.write(json.dumps(rec) + "\n")
    
    return output_file

def validate_trace_data(jsonl_file: str, max_check: int = 10) -> Dict[str, Any]:
    dataset = SAPTraceDataset(jsonl_file, max_records=max_check)
    stats = {
        "total_samples": len(dataset),
        "avg_seq_len": 0,
        "trace_dims": None,
        "mask_dims": None,
        "vocab_size": 0
    }
    
    if len(dataset) > 0:
        sample = dataset[0]
        stats["avg_seq_len"] = sample["input_ids"].size(0)
        stats["trace_dims"] = sample["z_t"].shape
        stats["mask_dims"] = sample["m_t"].shape
        stats["vocab_size"] = sample["p_t"].size(-1)
    
    return stats
