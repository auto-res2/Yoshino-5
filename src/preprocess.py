import json
import random
from pathlib import Path
from typing import Dict, List


def synth_dataset(n: int = 200, styles=("cot", "pot")) -> List[Dict]:
    data = []
    for i in range(n):
        a = random.randint(0, 9999)
        b = random.randint(0, 9999)
        style = random.choice(list(styles))
        if style == "cot":
            prompt = f"Q: What is the sum of {a} and {b}? A:"
        elif style == "pot":
            prompt = f"Q: add {a} and {b} A:"
        else:
            prompt = f"Q: {a} + {b} A:"
        gold = str(a + b)
        data.append({"id": i, "prompt": prompt, "gold": gold, "style": style})
    return data


def save_dataset(data: List[Dict], data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "synth_dataset.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    return out_path


def run(output_dir: Path, n_train: int = 200) -> Path:
    print("[Preprocess] Generating synthetic toy dataset...")
    data = synth_dataset(n_train)
    out_path = save_dataset(data, output_dir)
    print(f"[Preprocess] Saved dataset to {out_path}")
    return out_path
