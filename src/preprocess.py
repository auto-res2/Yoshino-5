# -*- coding: utf-8 -*-
"""
Preprocessing and synthetic dataset generation for FedU-Align experiments.
Creates multimodal synthetic items and federated client splits with modality heterogeneity.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SyntheticMMItem:
    def __init__(self, vid_tokens, aud_tokens, txt_tokens, label: int):
        self.vision_tokens = vid_tokens
        self.audio_tokens = aud_tokens
        self.text_tokens = txt_tokens
        self.label = label


class SyntheticMultimodalDataset(Dataset):
    def __init__(
        self,
        items: List[SyntheticMMItem],
        owned_modalities: Tuple[bool, bool, bool] = (True, True, True),
        missing_rate: float = 0.0,
        seed: int = 0,
    ):
        self.items = items
        self.owned_modalities = owned_modalities
        self.missing_rate = missing_rate
        self.rng = np.random.RandomState(seed)
        # Infer shapes
        if len(items) == 0:
            raise ValueError("Dataset has no items")
        self.Tv, self.Dv = items[0].vision_tokens.shape
        self.Ta, self.Da = items[0].audio_tokens.shape
        self.Tt, self.Dt = items[0].text_tokens.shape

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        # Apply missingness per-sample independently for each owned modality
        mask_v = self.owned_modalities[0] and (self.rng.rand() > self.missing_rate)
        mask_a = self.owned_modalities[1] and (self.rng.rand() > self.missing_rate)
        mask_t = self.owned_modalities[2] and (self.rng.rand() > self.missing_rate)
        # Ensure at least one modality present if any owned
        if not (mask_v or mask_a or mask_t):
            owned_idx = [i for i, o in enumerate(self.owned_modalities) if o]
            if len(owned_idx) > 0:
                choose = self.rng.choice(owned_idx)
                if choose == 0:
                    mask_v = True
                elif choose == 1:
                    mask_a = True
                else:
                    mask_t = True
        v = torch.tensor(it.vision_tokens, dtype=torch.float32)
        a = torch.tensor(it.audio_tokens, dtype=torch.float32)
        t = torch.tensor(it.text_tokens, dtype=torch.float32)
        if not mask_v:
            v = torch.zeros_like(v)
        if not mask_a:
            a = torch.zeros_like(a)
        if not mask_t:
            t = torch.zeros_like(t)
        sample = {
            'vision_tokens': v,
            'audio_tokens': a,
            'text_tokens': t,
            'label': torch.tensor(it.label, dtype=torch.long),
            'mask': torch.tensor([int(mask_v), int(mask_a), int(mask_t)], dtype=torch.long),
        }
        return sample


def generate_synthetic_items(
    n_samples: int,
    num_classes: int,
    token_dims: Tuple[int, int, int],
    tokens_per_mod: Tuple[int, int, int],
    class_sep: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 0,
) -> List[SyntheticMMItem]:
    rng = np.random.RandomState(seed)
    D_shared = min(token_dims)
    class_mu = rng.randn(num_classes, D_shared).astype(np.float32) * class_sep
    P_v = rng.randn(D_shared, token_dims[0]).astype(np.float32)
    P_a = rng.randn(D_shared, token_dims[1]).astype(np.float32)
    P_t = rng.randn(D_shared, token_dims[2]).astype(np.float32)

    items = []
    for _ in range(n_samples):
        y = rng.randint(0, num_classes)
        z = class_mu[y] + rng.randn(D_shared).astype(np.float32) * 0.5
        v_tokens = (z @ P_v + rng.randn(token_dims[0]).astype(np.float32) * noise_std)
        a_tokens = (z @ P_a + rng.randn(token_dims[1]).astype(np.float32) * noise_std)
        t_tokens = (z @ P_t + rng.randn(token_dims[2]).astype(np.float32) * noise_std)
        Tv, Ta, Tt = tokens_per_mod
        v_seq = np.tile(v_tokens, (Tv, 1)) + rng.randn(Tv, token_dims[0]).astype(np.float32) * (noise_std * 0.5)
        a_seq = np.tile(a_tokens, (Ta, 1)) + rng.randn(Ta, token_dims[1]).astype(np.float32) * (noise_std * 0.5)
        t_seq = np.tile(t_tokens, (Tt, 1)) + rng.randn(Tt, token_dims[2]).astype(np.float32) * (noise_std * 0.5)
        items.append(SyntheticMMItem(v_seq, a_seq, t_seq, int(y)))
    return items


@dataclass
class ClientConfig:
    client_id: int
    owns_modalities: Tuple[bool, bool, bool]
    train_dataset: SyntheticMultimodalDataset
    val_dataset: SyntheticMultimodalDataset


def create_federated_clients(
    num_clients: int,
    total_samples: int,
    num_classes: int,
    token_dims: Tuple[int, int, int] = (32, 32, 32),
    tokens_per_mod: Tuple[int, int, int] = (8, 8, 8),
    alpha: float = 0.2,
    mono_frac: float = 0.25,
    bi_frac: float = 0.5,
    tri_frac: float = 0.25,
    missing_rate_train: float = 0.1,
    missing_rate_val: float = 0.2,
    seed: int = 0,
) -> List[ClientConfig]:
    rng = np.random.RandomState(seed)
    assert abs(mono_frac + bi_frac + tri_frac - 1.0) < 1e-6
    ownerships = []
    for _ in range(num_clients):
        r = rng.rand()
        if r < mono_frac:
            which = rng.choice(3)
            owns = [False, False, False]
            owns[which] = True
            ownerships.append(tuple(owns))
        elif r < mono_frac + bi_frac:
            idx = rng.choice(3, size=2, replace=False)
            owns = [False, False, False]
            for j in idx:
                owns[j] = True
            ownerships.append(tuple(owns))
        else:
            ownerships.append((True, True, True))

    client_label_dist = rng.dirichlet([alpha] * num_classes, size=num_clients)
    client_sizes = rng.multinomial(total_samples, [1 / num_clients] * num_clients)

    clients: List[ClientConfig] = []
    start_seed = seed * 13 + 7
    for cid in range(num_clients):
        n_i = int(client_sizes[cid])
        if n_i == 0:
            n_i = 1
        ys = rng.choice(num_classes, size=n_i, p=client_label_dist[cid])
        items: List[SyntheticMMItem] = []
        for y in ys:
            items_y = generate_synthetic_items(
                1, num_classes=num_classes, token_dims=token_dims, tokens_per_mod=tokens_per_mod,
                class_sep=2.0, noise_std=0.5, seed=start_seed + cid * 1000 + int(y)
            )
            items.extend(items_y)
        split = max(1, int(0.8 * len(items)))
        train_items = items[:split]
        val_items = items[split:]
        m_own = ownerships[cid]
        train_ds = SyntheticMultimodalDataset(train_items, owned_modalities=m_own, missing_rate=missing_rate_train, seed=start_seed + cid)
        val_ds = SyntheticMultimodalDataset(val_items, owned_modalities=m_own, missing_rate=missing_rate_val, seed=start_seed + cid + 999)
        clients.append(ClientConfig(client_id=cid, owns_modalities=m_own, train_dataset=train_ds, val_dataset=val_ds))
    return clients
