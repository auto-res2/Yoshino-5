#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for MuViC synthetic experiments.
Run from project root with:
    python -m src.main

This will load config/config.yaml and run the selected plan(s), saving all figures
as high-quality PDFs in .research/iteration2/images.
"""
from __future__ import annotations
import argparse
import os
from typing import Any, Dict

import yaml

try:
    from .evaluate import run_plan1_synthetic, run_plan2_robustness, run_plan3_blackbox
except ImportError:  # Fallback when executed as a script without package context
    from evaluate import run_plan1_synthetic, run_plan2_robustness, run_plan3_blackbox  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MuViC synthetic experiments")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--plan",
        type=str,
        default="all",
        choices=["all", "plan1", "plan2", "plan3"],
        help="Which experiment plan to run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    outdir = cfg.get("output_dir", ".research/iteration2/images")
    os.makedirs(outdir, exist_ok=True)

    print("================ MuViC Synthetic Experiments ================")
    print(f"Config: {os.path.abspath(args.config)}")
    print(f"Output dir (PDFs): {os.path.abspath(outdir)}")

    if args.plan in ("all", "plan1"):
        print("\n>>> Running Plan 1 (full pipeline synthetic)...")
        run_plan1_synthetic(seed=int(cfg.get("plan1_seed", 42)), outdir=outdir)

    if args.plan in ("all", "plan2"):
        print("\n>>> Running Plan 2 (robustness to timing)...")
        run_plan2_robustness(seed=int(cfg.get("plan2_seed", 43)), outdir=outdir)

    if args.plan in ("all", "plan3"):
        print("\n>>> Running Plan 3 (black-box SEL+GRO)...")
        run_plan3_blackbox(seed=int(cfg.get("plan3_seed", 44)), outdir=outdir)

    print("\nAll done. Inspect generated PDFs in:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
