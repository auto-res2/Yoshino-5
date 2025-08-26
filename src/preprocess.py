import random
import string
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import numpy as np

VOCAB = list(string.ascii_lowercase) + list(string.digits) + [" "]
TOKEN2ID = {c: i + 1 for i, c in enumerate(VOCAB)}
ID2TOKEN = {i: c for c, i in TOKEN2ID.items()}

def encode_text(text: str, max_len: int = 32) -> torch.Tensor:
    ids = [TOKEN2ID.get(c, 0) for c in text.lower()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def synthetic_math_problem(n: int = 2) -> Tuple[str, str]:
    a, b = random.randint(0, 9), random.randint(0, 9)
    if n == 2:
        q = f"What is {a}+{b}?"
        ans = str(a + b)
    else:
        c = random.randint(0, 9)
        q = f"Compute {a}+{b}+{c}."
        ans = str(a + b + c)
    return q, ans

LANG_PREFIX = {
    "de": "[DE] ", "es": "[ES] ", "fr": "[FR] ", 
    "ar": "[AR] ", "hi": "[HI] ", "zh": "[ZH] "
}

class ToyMathDataset(Dataset):
    
    def __init__(self, n_samples: int = 200, multilingual: bool = False, 
                 shuffle_sent: bool = False, add_noise: bool = False):
        self.data: List[Tuple[str, str]] = []
        for _ in range(n_samples):
            q, a = synthetic_math_problem(random.choice([2, 3]))
            
            if multilingual:
                lang = random.choice(list(LANG_PREFIX))
                q = LANG_PREFIX[lang] + q
            
            if shuffle_sent:
                words = q.split()
                random.shuffle(words)
                q = " ".join(words)
            
            if add_noise:
                if random.random() < 0.1:
                    q += " [NOISE]"
            
            self.data.append((q, a))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        enc_q = encode_text(q)
        enc_a = encode_text(a, max_len=4)
        return enc_q, enc_a

def create_datasets():
    datasets = {
        "clean": ToyMathDataset(1000),
        "shuffled": ToyMathDataset(500, shuffle_sent=True),
        "multilingual": ToyMathDataset(500, multilingual=True),
        "noisy": ToyMathDataset(500, add_noise=True),
        "combined": ToyMathDataset(500, multilingual=True, shuffle_sent=True, add_noise=True)
    }
    return datasets
