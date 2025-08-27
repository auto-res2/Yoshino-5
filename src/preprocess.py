import random
import torch
from torch.utils.data import Dataset, DataLoader


class ToyTask(Dataset):
    def __init__(self, n: int, seed: int = 0):
        random.seed(seed)
        self.samples = []
        for _ in range(n):
            a = random.randint(0, 9)
            q = f"Add zero to {a}. What is the result?"
            self.samples.append({"question": q, "answer": a})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    qs = [b["question"] for b in batch]
    ans = torch.tensor([b["answer"] for b in batch], dtype=torch.long)
    return qs, ans


def get_dataloaders(seed: int = 42, n_train: int = 128, n_val: int = 128, batch_size: int = 32):
    train = ToyTask(n_train, seed=seed)
    val = ToyTask(n_val, seed=seed + 1)
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate),
    }
