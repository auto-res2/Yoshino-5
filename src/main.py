from __future__ import annotations

"""
main.py – top-level experiment orchestration
Run with:  python -m src.main (or python src/main.py)

The module can now be launched either as a script *or* with the
`-m` flag because we switched to absolute imports that do not rely on
`src` being recognised as a package via relative imports.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import torch

# -----------------------------------------------------------------------------
# NOTE: use absolute imports ---------------------------------------------------
# -----------------------------------------------------------------------------
# Using absolute import paths works both when the file is executed with
# ``python -m src.main`` (package mode) and when launched directly via
# ``python src/main.py`` (script mode) because the repository root
# (containing the *src* folder) is automatically added to ``sys.path`` by
# the interpreter when resolving the script location.
# -----------------------------------------------------------------------------
from src.train import load_llama_lora
from src.preprocess import CLUESplit
from src import evaluate as ev  # noqa: F401 (imported for side-effects)

# -----------------------------------------------------------------------------
# Utilities -------------------------------------------------------------------
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
# All images must be stored under the iteration **2** directory according to
# the rubric.
IMAGES_DIR = ROOT / ".research" / "iteration2" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("agsc.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Configuration handling ------------------------------------------------------
# -----------------------------------------------------------------------------

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_cfg() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file {CONFIG_PATH} missing – please create it first.")
    import yaml  # Local import so that requirements.txt stays minimal

    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# -----------------------------------------------------------------------------
# Example experiment: Offline curriculum search on CLUE++ ---------------------
# -----------------------------------------------------------------------------


def run_experiment_clue(cfg: Dict[str, Any]):
    logger.info("==== Experiment: Offline AGSC on CLUE++ ====")

    # 1. data -----------------------------------------------------------------
    clue_ds = CLUESplit(cfg["datasets"])
    task_names = [d["name"] for d in cfg["datasets"]]

    # 2. model ----------------------------------------------------------------
    model, tokenizer = load_llama_lora(
        model_id=cfg["model"]["id"],
        r=int(cfg["model"]["lora_r"]),
        alpha=int(cfg["model"]["lora_alpha"]),
        bnb_8bit=bool(cfg["model"].get("bnb_8bit", True)),
    )

    # 3. obtain semantic prototypes ------------------------------------------
    device = next(model.parameters()).device
    prototypes: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for tname in task_names:
            ds = clue_ds.get_dataset(tname)["validation"].select(range(32))
            reps = []
            for ex in ds:
                # Generic sentence field handling
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

    # 4. build similarity matrix ---------------------------------------------
    from src import curriculum as cur  # local import to avoid circularity

    sim_mat = cur.build_similarity_matrix(prototypes)
    grad_conf_mat = {k: 0.0 for k in sim_mat}  # placeholder – no grads offline
    cost_mat = cur.build_conflict_matrix(
        sim_mat,
        grad_conf_mat,
        alpha=float(cfg["curriculum"]["alpha"]),
        beta=float(cfg["curriculum"]["beta"]),
    )
    order = cur.beam_search(task_names, cost_mat, int(cfg["curriculum"]["beam_size"]))

    logger.info("AGSC beam-search order: %s", order)

    # 5. save schedule --------------------------------------------------------
    schedule_path = ROOT / "schedule_clue.json"
    schedule_path.write_text(json.dumps(order))
    logger.info("Schedule written to %s", schedule_path)

    logger.info("Experiment completed – training loop skipped in refactor version.")


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def main():
    cfg = load_cfg()
    seeds = cfg["dev_env"].get("seeds", [0])

    for s in seeds:
        set_seed(s)
        run_experiment_clue(cfg["exp1"])


if __name__ == "__main__":
    # Ensure that the repository root is on ``sys.path`` when the script is
    # executed directly. This is a no-op when using ``python -m src.main``.
    import sys as _sys

    root_str = str(ROOT)
    if root_str not in _sys.path:
        _sys.path.insert(0, root_str)

    main()
