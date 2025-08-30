import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# -----------------------------
# Repro and device
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------
# Synthetic video generator with motion complexity and domain shift
# -----------------------------

def gen_synthetic_shapes(T: int = 64, H: int = 128, W: int = 128, n_obj: int = 3,
                          n_segments: int = 1, seed: int = 0, domain: str = 'squares',
                          add_noise: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
      frames: np.ndarray [T, H, W, 3] uint8
      meta: dict with keys 'segments', 'first_change_t', 'length', 'domain'
    """
    rng = np.random.RandomState(seed)
    frames = []
    objs = []
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    seg_len = max(2, T // n_segments)
    change_points = [min(T-1, (i+1)*seg_len) for i in range(n_segments-1)]
    first_change_t = change_points[0] if len(change_points) > 0 else T//2
    for i in range(n_obj):
        x, y = rng.randint(10, W-10), rng.randint(10, H-10)
        vel_profile = []
        for s in range(n_segments):
            vx_s = int(rng.choice([-4,-3,-2,-1,1,2,3,4]))
            vy_s = int(rng.choice([-4,-3,-2,-1,1,2,3,4]))
            length_s = seg_len if s < n_segments-1 else T - seg_len*(n_segments-1)
            vel_profile += [(vx_s, vy_s)] * max(1, length_s)
        objs.append({
            "pos": [x,y],
            "vel": vel_profile[:T],
            "color": colors[i % len(colors)],
            "size": int(rng.randint(6, 12))
        })
    for t in range(T):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for o in objs:
            vx, vy = o["vel"][t]
            o["pos"][0] = int(np.clip(o["pos"][0] + vx, 0, W-1))
            o["pos"][1] = int(np.clip(o["pos"][1] + vy, 0, H-1))
            x, y = o["pos"]
            s = o["size"]
            x0, y0 = max(0, x-s), max(0, y-s)
            x1, y1 = min(W, x+s), min(H, y+s)
            if domain == 'squares':
                img[y0:y1, x0:x1, :] = o["color"]
            else:
                yy, xx = np.ogrid[:H, :W]
                mask = (yy - y)**2 + (xx - x)**2 <= s**2
                img[mask] = o["color"]
        if add_noise:
            noise = rng.randint(0, 30, size=(H, W, 3)).astype(np.uint8)
            img = np.clip(img + noise, 0, 255)
        frames.append(img)
    frames = np.stack(frames, axis=0)
    meta = {
        "segments": n_segments,
        "first_change_t": int(first_change_t),
        "length": T,
        "domain": domain
    }
    return frames, meta


# -----------------------------
# Mock CLIP-like feature extractor (tiny CNN -> pooled -> linear -> normalized)
# -----------------------------

class MockCLIP(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8,8))
        self.proj = nn.Linear(32*8*8, d)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        # frames: [T, 3, H, W] in [0,1]
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x  # [T, D]


# -----------------------------
# Dataset and DataLoader for synthetic experiments
# -----------------------------

class SynthVideoDataset(Dataset):
    def __init__(self, n: int = 256, complexities=(1,2,4,8), lengths=(16,32,64),
                 domains=('squares','circles'), noise_prob: float = 0.3, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.items = []
        for i in range(n):
            seg = int(rng.choice(complexities))
            T = int(rng.choice(lengths))
            domain = str(rng.choice(domains))
            add_noise = bool(rng.rand() < noise_prob)
            self.items.append((seg, T, domain, add_noise, i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seg, T, domain, add_noise, seed = self.items[idx]
        frames, meta = gen_synthetic_shapes(T=T, n_segments=seg, seed=seed, domain=domain, add_noise=add_noise)
        # QA class: segments mapped to classes {1->0, 2->1, 4->2, 8->3}
        seg_to_cls = {1:0, 2:1, 4:2, 8:3}
        qa_cls = seg_to_cls.get(seg, 0)
        first_change_t = meta['first_change_t'] if seg > 1 else T//2
        return {
            'frames': frames,           # [T,H,W,3] uint8
            'qa_cls': qa_cls,           # int 0..3
            'first_change_t': first_change_t,  # int
            'segments': seg,            # int
            'length': T,                # int
            'domain': meta['domain']
        }
