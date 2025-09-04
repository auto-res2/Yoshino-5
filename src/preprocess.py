import os
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import torchvision as tv
    from torchvision import transforms as T
except Exception:
    tv = None
    T = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RandomLabelDataset(Dataset):
    def __init__(self, base_ds: Dataset, num_classes: int = 10, seed: int = 0):
        self.base = base_ds
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=len(base_ds)).astype(int)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.labels[i])


class PermutePixels:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.perm = None
    def __call__(self, img):
        if not torch.is_tensor(img):
            img = T.ToTensor()(img)
        c, h, w = img.shape
        flat = img.view(c, -1)
        if self.perm is None:
            self.perm = self.rng.permutation(h * w)
        flat = flat[:, self.perm]
        out = flat.view(c, h, w)
        return out


def build_e1_tasks_vision(num_tasks: int = 6, probe_size: int = 256, batch_size: int = 64,
                           use_fake_data: bool = True, fast_test: bool = True, seed: int = 0) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    set_seed(seed)
    assert tv is not None and T is not None, "torchvision is required for vision tasks"
    task_ids = list(range(1, num_tasks + 1))

    to_rgb = T.Lambda(lambda z: z if z.shape[0] == 3 else z.repeat(3, 1, 1))

    # Use FakeData to simulate tasks with different transforms
    def make_fake(angle=0, permute=False, noisy=False, jitter=False):
        tfms = [T.ToTensor()]
        if jitter:
            tfms.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))
        if angle != 0:
            tfms.append(T.RandomRotation((angle, angle)))
        tfms.append(T.Resize((32, 32)))
        if permute:
            tfms.append(PermutePixels(seed=seed))
        tfms.append(to_rgb)
        tfms.append(T.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
        transform = T.Compose([t for t in tfms if t is not None])
        base_train = tv.datasets.FakeData(size=2000 if not fast_test else 300, image_size=(3, 32, 32),
                                          num_classes=10, transform=transform)
        base_test = tv.datasets.FakeData(size=500 if not fast_test else 100, image_size=(3, 32, 32),
                                         num_classes=10, transform=transform)
        if noisy:
            base_train = RandomLabelDataset(base_train, num_classes=10, seed=seed)
            base_test = RandomLabelDataset(base_test, num_classes=10, seed=seed + 1)
        return base_train, base_test

    task_specs = [
        {"angle": 0, "permute": False, "noisy": False, "jitter": False},
        {"angle": 15, "permute": False, "noisy": False, "jitter": False},
        {"angle": 30, "permute": False, "noisy": False, "jitter": False},
        {"angle": 45, "permute": False, "noisy": False, "jitter": False},
        {"angle": 0, "permute": True,  "noisy": False, "jitter": False},
        {"angle": 0,  "permute": False, "noisy": False, "jitter": True},
    ]
    task_specs = task_specs[:num_tasks]

    task_dls: Dict[int, Dict[str, Any]] = {}
    for i, spec in enumerate(task_specs, start=1):
        tr, te = make_fake(**spec)
        n = len(tr)
        idx = torch.randperm(n).tolist()
        val_size = min(200, n // 5)
        tr_size = n - val_size
        tr_idx, val_idx = idx[:tr_size], idx[tr_size:]
        probe_idx = tr_idx[:min(probe_size, len(tr_idx))]
        train_dl = DataLoader(Subset(tr, tr_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        probe_dl = DataLoader(Subset(tr, probe_idx), batch_size=batch_size, shuffle=False, num_workers=0)
        val_dl = DataLoader(Subset(tr, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
        test_dl = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)
        task_dls[i] = {
            'train': train_dl, 'probe': probe_dl, 'val': val_dl, 'test': test_dl, 'n_classes': 10
        }
    return task_ids, task_dls


# -------------------------
# Simple synthetic NLP tasks (fallback for E3 quick tests)
# -------------------------

class SyntheticTextDataset(Dataset):
    def __init__(self, n_samples=1000, vocab_size=5000, seq_len=32, n_classes=2, topic_shift: int = 0, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.X = []
        self.y = []
        base_probs = np.ones(vocab_size)
        if topic_shift > 0:
            idxs = np.arange(vocab_size)
            bump = (np.sin((idxs + topic_shift) / 50.0) + 1.5)
            base_probs = base_probs * bump
        base_probs = base_probs / base_probs.sum()
        for _ in range(n_samples):
            label = rng.randint(0, n_classes)
            p = base_probs.copy()
            p[(label * 13 + np.arange(0, vocab_size, 17)) % vocab_size] += 0.1
            p /= p.sum()
            seq = rng.choice(vocab_size, size=seq_len, replace=True, p=p)
            self.X.append(torch.tensor(seq, dtype=torch.long))
            self.y.append(int(label))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_e3_tasks_text(num_tasks: int = 4, probe_size: int = 256, batch_size: int = 64,
                         vocab_size: int = 5000, seq_len: int = 64, fast_test: bool = True, seed: int = 0):
    set_seed(seed)
    tasks: Dict[int, Dict[str, Any]] = {}
    task_ids = list(range(1, num_tasks + 1))
    topic_shifts = [0, 25, 50, 75, 100, 150][:num_tasks]
    for i, shift in enumerate(topic_shifts, start=1):
        n_train = 2000 if not fast_test else 300
        n_val = 400 if not fast_test else 100
        n_test = 800 if not fast_test else 100
        tr = SyntheticTextDataset(n_samples=n_train + n_val, vocab_size=vocab_size, seq_len=seq_len, topic_shift=shift, seed=seed + i)
        te = SyntheticTextDataset(n_samples=n_test, vocab_size=vocab_size, seq_len=seq_len, topic_shift=shift + 5, seed=seed + 100 + i)
        idx = torch.randperm(len(tr)).tolist()
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        probe_idx = tr_idx[:min(probe_size, len(tr_idx))]
        def collate(batch):
            xs, ys = zip(*batch)
            return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)
        tasks[i] = {
            'train': DataLoader(Subset(tr, tr_idx), batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0),
            'probe': DataLoader(Subset(tr, probe_idx), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
            'val': DataLoader(Subset(tr, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
            'test': DataLoader(te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0),
            'n_classes': 2,
            'vocab_size': vocab_size,
        }
    return task_ids, tasks


def prepare_output_dirs() -> str:
    out_dir = os.path.join('.research', 'iteration1', 'images')
    ensure_dir(out_dir)
    return out_dir
