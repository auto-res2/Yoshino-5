import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")


def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def now_ms():
    return time.perf_counter() * 1e3


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def measure_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    else:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except Exception:
            return -1


# ------------------------------
# Synthetic dataset generators
# ------------------------------
@dataclass
class SyntheticDataConfig:
    vocab_size: int = 200
    seq_len: int = 64
    batch_size: int = 8
    n_batches: int = 64


def generate_language_data(cfg: SyntheticDataConfig, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    data = []
    for _ in range(cfg.n_batches):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        for b in range(cfg.batch_size):
            for t in range(cfg.seq_len - 1):
                if int(x[b, t].item()) % 7 == 0:
                    x[b, t+1] = (x[b, t] + 1) % cfg.vocab_size
        data.append({"input_ids": x})
    return data


def generate_classification_data(n_samples: int = 512, seq_len: int = 32, vocab_size: int = 200, device: Optional[torch.device] = None):
    device = device if device is not None else get_device()
    X = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    y = (((X[:, -5:].sum(dim=1) % 2) == 0) & (X[:, 0] < vocab_size // 2)).long()
    return X, y
