# -*- coding: utf-8 -*-
"""
Evaluation utilities and plotting helpers for Phi-GaLore experiments.
"""
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Use a non-interactive backend for headless environments


def ensure_pdf_backend_and_style():
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    sns.set_context("paper")
    sns.set_style("whitegrid")


def save_plot(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')


# Physics-related evaluation


def divergence_and_laplacian(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # u: [B, 2, Nx, Ny]
    kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    ux_x = F.conv2d(u[:, 0:1], kx, padding=1)
    uy_y = F.conv2d(u[:, 1:2], ky, padding=1)
    div = ux_x + uy_y
    uxx = F.conv2d(u[:, 0:1], lap, padding=1)
    uyy = F.conv2d(u[:, 1:2], lap, padding=1)
    lap_u = torch.cat([uxx, uyy], dim=1)
    return div, lap_u


def physics_loss(u_pred: torch.Tensor, u_ref: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
    # rL2 data term + divergence penalty + Laplacian smoothness as PDE proxy
    data = F.mse_loss(u_pred, u_ref)
    div, lap_u = divergence_and_laplacian(u_pred)
    div_pen = (div.pow(2).mean())
    smooth = (lap_u.pow(2).mean())
    return data + 0.1 * div_pen + 0.01 * smooth


def relative_l2(u_pred: torch.Tensor, u_ref: torch.Tensor) -> float:
    num = torch.norm(u_pred - u_ref).item()
    den = torch.norm(u_ref).item() + 1e-12
    return num / den
