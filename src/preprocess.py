import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def synthetic_video(static: bool, T: int = 32, H: int = 224, W: int = 224) -> torch.Tensor:
    base = torch.rand(3, H, W)
    noise = torch.zeros(T, 3, H, W) if static else torch.randn(T, 3, H, W) * 0.3
    vid = base.unsqueeze(0).repeat(T, 1, 1, 1) + noise
    return vid.clamp(0, 1)

class ToyVideoQADataset(Dataset):
    def __init__(self, n: int = 200, static_prob: float = 0.5):
        self.samples = []
        for _ in range(n):
            is_static = random.random() < static_prob
            video = synthetic_video(is_static)
            question = torch.randint(0, 1000, (random.randint(3, 8),))
            answer_id = random.randint(0, 4)
            self.samples.append((video, question, answer_id, is_static))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    videos, qs, ans, stat = zip(*batch)
    videos = torch.stack(videos)
    qs_pad = nn.utils.rnn.pad_sequence(qs, batch_first=True)
    ans = torch.tensor(ans)
    stat = torch.tensor(stat)
    return videos, qs_pad, ans, stat

def create_dataloader(dataset_size: int = 200, batch_size: int = 16, static_prob: float = 0.5):
    dataset = ToyVideoQADataset(dataset_size, static_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
