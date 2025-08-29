import os
from typing import Optional

from .train import ensure_dir


def preprocess(images_dir: str = ".research/iteration1/images", data_dir: str = "data", models_dir: str = "models"):
    """
    Prepare directories and any lightweight assets needed for the experiments.
    Since we use synthetic data, there is no heavy preprocessing. We just ensure dirs exist.
    """
    ensure_dir(images_dir)
    ensure_dir(data_dir)
    ensure_dir(models_dir)
    # Placeholders or additional setup can be added here.

    print("Preprocess complete.")
    print(" - images_dir:", os.path.abspath(images_dir))
    print(" - data_dir:", os.path.abspath(data_dir))
    print(" - models_dir:", os.path.abspath(models_dir))
