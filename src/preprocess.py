# -*- coding: utf-8 -*-
"""
Data generation, preprocessing utilities, and runtime helpers.
"""
import os
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_and_mp_dtype():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp_dtype = torch.float32
    if device.type == 'cuda':
        major, minor = torch.cuda.get_device_capability()
        # Ampere (8.0+) supports bf16 efficiently; otherwise prefer fp16 on T4 (7.5)
        if major >= 8:
            mp_dtype = torch.bfloat16
        else:
            mp_dtype = torch.float16
    return device, mp_dtype


def autocast_if_cuda(mp_dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=mp_dtype)
    else:
        class Dummy:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return Dummy()


# ------------------------------
# Synthetic datasets
# ------------------------------

def gen_darcy_like_batch(B: int, Nx: int, Ny: int, device: torch.device, noise: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # Random permeability k(x,y); target velocity field is gradient of a smoothed streamfunction
    k = torch.randn(B, 1, Nx, Ny, device=device)
    if noise > 0:
        k = k + noise * torch.randn_like(k)
    # build streamfunction psi by smoothing k
    kernel = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], device=device, dtype=k.dtype).view(1, 1, 3, 3)
    psi = k.clone()
    for _ in range(5):
        psi = F.conv2d(psi, kernel, padding=1) / 8.0
    # derive u = (dpsi/dy, -dpsi/dx)
    kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=device, dtype=k.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=device, dtype=k.dtype).view(1, 1, 3, 3)
    ux = F.conv2d(psi, ky, padding=1)
    uy = -F.conv2d(psi, kx, padding=1)
    vel = torch.cat([ux, uy], dim=1)
    return k, vel


def gen_burgers_bc_variants(B: int, Nx: int, Ny: int, device: torch.device, bc_noise: float = 0.0, jitter: float = 0.0):
    # Build base field from a few sinusoids; noise perturbs boundary values; jitter perturbs grid coords
    xs = torch.linspace(0, 2 * math.pi, Nx, device=device)
    ys = torch.linspace(0, 2 * math.pi, Ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    u0 = torch.sin(2 * X) * torch.sin(1 * Y)
    v0 = torch.cos(1 * X) * torch.sin(2 * Y)
    u = u0.unsqueeze(0).repeat(B, 1, 1)
    v = v0.unsqueeze(0).repeat(B, 1, 1)
    base = torch.stack([u, v], dim=1)  # [B,2,Nx,Ny]
    if bc_noise > 0:
        base[:, :, 0, :] += bc_noise * torch.randn_like(base[:, :, 0, :])
        base[:, :, -1, :] += bc_noise * torch.randn_like(base[:, :, -1, :])
        base[:, :, :, 0] += bc_noise * torch.randn_like(base[:, :, :, 0])
        base[:, :, :, -1] += bc_noise * torch.randn_like(base[:, :, :, -1])
    # jitter grid positions (apply bilinear sampling)
    if jitter > 0:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, Nx, device=device), torch.linspace(-1, 1, Ny, device=device), indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        noise = jitter * torch.randn_like(grid)
        grid = (grid + noise).clamp(-1, 1)
        base = F.grid_sample(base, grid, align_corners=True)
    # inputs could be some forcing; use zeros here
    inp = torch.zeros(B, 1, Nx, Ny, device=device)
    return inp, base


def gen_molecular_like_batch(B: int, L: int, device: torch.device, rot_noise: float = 0.0, pos_noise: float = 0.0):
    # positions ~ N(0,1); atomic numbers ~ {1,6,7,8} scaled
    Z_choices = torch.tensor([1.0, 6.0, 7.0, 8.0], device=device)
    Z_idx = torch.randint(0, len(Z_choices), (B, L), device=device)
    Z = Z_choices[Z_idx] / 8.0
    pos = torch.randn(B, L, 3, device=device)
    if pos_noise > 0:
        pos = pos + pos_noise * torch.randn_like(pos)
    if rot_noise > 0:
        # random rotation around z for simplicity
        theta = torch.rand(B, device=device) * 2 * math.pi
        c, s = torch.cos(theta), torch.sin(theta)
        Rz = torch.stack([torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
                          torch.stack([s, c, torch.zeros_like(c)], dim=-1),
                          torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1)], dim=-2)
        pos = torch.einsum('bij,bkj->bki', Rz, pos)
    tokens = torch.cat([pos, Z.unsqueeze(-1)], dim=-1)  # (x,y,z,Znorm)
    # property target: sum_{i<j} exp(-gamma * ||ri-rj||) * (Zi+Zj)
    gamma = 0.7
    dists = torch.cdist(pos, pos)  # [B,L,L]
    mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    y = (torch.exp(-gamma * dists) * mask).sum(dim=(1, 2))
    y = y + 0.1 * Z.sum(dim=1)
    return tokens, y


# ------------------------------
# Physics basis builders
# ------------------------------

def build_divfree_basis_fourier(Nx: int, Ny: int, K: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(0, 2 * math.pi, steps=Nx, device=device)
    ys = torch.linspace(0, 2 * math.pi, steps=Ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    modes = []
    max_k = int(max(1, K ** 0.5)) + 2
    for kx in range(1, max_k):
        for ky in range(1, max_k):
            if len(modes) >= K:
                break
            psi = torch.sin(kx * X) * torch.sin(ky * Y)
            ux = kx * torch.cos(kx * X) * torch.sin(ky * Y)
            uy = -(ky * torch.sin(kx * X) * torch.cos(ky * Y))
            modes.append(torch.stack([ux, uy], dim=0))  # [2, Nx, Ny]
        if len(modes) >= K:
            break
    B = torch.stack(modes, dim=-1)  # [2, Nx, Ny, K]
    B = B.reshape(2 * Nx * Ny, K)
    Q, _ = torch.linalg.qr(B.float(), mode='reduced')
    # Store in reduced precision if CUDA else float32
    store_dtype = torch.float32 if device.type == 'cpu' else (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    return Q[:, :K].to(store_dtype)
