[UPDATED]
import os
import sys
import argparse
import random
import yaml
from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd

# Switched to absolute imports – relative imports break when running the file
# as a script (i.e. `python src/main.py`).  Since the `src/` directory is in
# `sys.path`, these top-level imports resolve correctly.
from train import ContinualLearner  # noqa: E402
from evaluate import analyze_and_print_results  # noqa: E402


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)

    # Runtime adjustments
    config.device = config.device if torch.cuda.is_available() else "cpu"
    config.effective_batch_size = config.train_batch_size * config.grad_accum_steps
    config.state_dim = config.search_window_m ** 2 * 2 + config.search_window_m

    return config


# (the remainder of the file is unchanged)
