# -*- coding: utf-8 -*-
"""
Main entry point for Phi-GaLore experiments.
Run from project root: python -m src.main
"""
import os
import yaml
from typing import Any, Dict

from .preprocess import get_device_and_mp_dtype
from .train import (
    experiment_1_micro_bench,
    experiment_2_physics_aware,
    experiment_3_hierarchical_sharing,
)


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def main():
    ensure_dirs()
    cfg = load_config('config/config.yaml')

    device, mp_dtype = get_device_and_mp_dtype()
    print(f"Using device: {device} | mixed-precision dtype: {mp_dtype}")

    images_dir = cfg.get('images_dir', '.research/iteration1/images')

    # Experiment 1
    if cfg.get('exp1', {}).get('run', True):
        s = cfg.get('exp1', {}).get('steps', 8)
        nx = cfg.get('exp1', {}).get('Nx', 16)
        ny = cfg.get('exp1', {}).get('Ny', 16)
        experiment_1_micro_bench(images_dir=images_dir, steps=s, Nx=nx, Ny=ny, device=device, mp_dtype=mp_dtype)

    # Experiment 2
    if cfg.get('exp2', {}).get('run', True):
        s = cfg.get('exp2', {}).get('steps', 8)
        nx = cfg.get('exp2', {}).get('Nx', 16)
        ny = cfg.get('exp2', {}).get('Ny', 16)
        experiment_2_physics_aware(images_dir=images_dir, Nx=nx, Ny=ny, steps=s, device=device, mp_dtype=mp_dtype)

    # Experiment 3
    if cfg.get('exp3', {}).get('run', True):
        s = cfg.get('exp3', {}).get('steps', 8)
        L = cfg.get('exp3', {}).get('seq_len', 16)
        experiment_3_hierarchical_sharing(images_dir=images_dir, steps=s, L=L, device=device, mp_dtype=mp_dtype)

    print("All configured experiments finished.")


if __name__ == '__main__':
    main()
