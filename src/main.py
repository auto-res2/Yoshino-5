from __future__ import annotations

"""
main.py – top-level experiment orchestration
Run with:  python -m src.main  (preferred) or  python src/main.py

We ensure *src* can be imported even in script mode by manipulating
``sys.path`` *before* performing any intra-package imports.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# -----------------------------------------------------------------------------
# Path handling – must precede `from src.` imports ----------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Directory for all experiment artefacts
IMAGES_DIR = ROOT / ".research" / "iteration3" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Now that the repo root is on sys.path we can safely do absolute imports.
# -----------------------------------------------------------------------------
from src.train import load_llama_lora  # noqa: E402  (import after sys.path hack)
from src.preprocess import CLUESplit  # noqa: E402
from src import evaluate as ev  # noqa: F401, E402 (imported for side-effects)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("agsc.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_cfg() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file {CONFIG_PATH} missing – please create it first.")

    import yaml  # Local import keeps requirements.txt minimal

    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

# -----------------------------------------------------------------------------
# Example experiment – Offline curriculum search on CLUE++
# -----------------------------------------------------------------------------

def run_experiment_clue(cfg: Dict[str, Any]):
    logger.info("==== Experiment: Offline AGSC on CLUE++ ====")

    # 1) Data -----------------------------------------------------------------
    clue_ds = CLUESplit(cfg["datasets"])
    task_names = [d["name"] for d in cfg["datasets"]]

    # 2) Model ----------------------------------------------------------------
    model, tokenizer = load_llama_lora(
        model_id=cfg["model"]["id"],
        r=int(cfg["model"]["lora_r"]),
        alpha=int(cfg["model"]["lora_alpha"]),
        bnb_8bit=bool(cfg["model"].get("bnb_8bit", True)),
    )

    # 3) Obtain semantic prototypes -----------------------------------------
    device = next(model.parameters()).device
    prototypes: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for tname in task_names:
            ds = clue_ds.get_dataset(tname)["validation"].select(range(32))
            reps = []
            for ex in ds:
                # Attempt multiple common sentence fields; fallback to str(ex)
                text = ex.get("sentence") or ex.get("sentence1") or str(ex)
                tok = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                ).to(device)
                emb = model.model.embed_tokens(tok.input_ids).mean(1).squeeze(0)
                reps.append(emb.float())
            prototypes[tname] = torch.stack(reps).mean(0).cpu()

    # 4) Build similarity / cost matrices ------------------------------------
    from src import curriculum as cur  # Local import avoids circular deps

    sim_mat = cur.build_similarity_matrix(prototypes)
    grad_conf_mat = {k: 0.0 for k in sim_mat}  # Placeholder – offline mode
    cost_mat = cur.build_conflict_matrix(
        sim_mat,
        grad_conf_mat,
        alpha=float(cfg["curriculum"]["alpha"]),
        beta=float(cfg["curriculum"]["beta"]),
    )
    order = cur.beam_search(task_names, cost_mat, int(cfg["curriculum"]["beam_size"]))

    logger.info("AGSC beam-search order: %s", order)

    # 5) Persist schedule -----------------------------------------------------
    schedule_path = ROOT / "schedule_clue.json"
    schedule_path.write_text(json.dumps(order))
    logger.info("Schedule written to %s", schedule_path)

    logger.info("Experiment completed – training loop skipped in refactor version.")

# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main():
    cfg = load_cfg()
    seeds = cfg["dev_env"].get("seeds", [0])

    for s in seeds:
        set_seed(s)
        run_experiment_clue(cfg["exp1"])


if __name__ == "__main__":
    main()
