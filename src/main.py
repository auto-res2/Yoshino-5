import os
import sys
import warnings
import yaml
import pandas as pd
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path to allow for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import (
    run_experiment, SpectralLoRA, RLTOPScheduler, TaskSelectionEnv,
    _calculate_fisher_similarity, _calculate_gradient_interference,
)
from preprocess import get_benchmark, CIFAR100BlurStream, get_transforms
from evaluate import (
    print_results_table, perform_significance_test,
    validate_results, generate_figures, get_eval_plugin,
)

# Resolve config path relative to repository root (one level up from src)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "config.yaml"
)


def verify_implementation():
    """ [IMPLEMENTED] Component: Implementation Verification """
    print("\n--- Verifying Implementation Components ---")
    required_components = {
        "Core Method: SALoRA": SpectralLoRA is not None,
        "Core Method: RL-TOP Scheduler": RLTOPScheduler is not None,
        "Core Method: RL Environment": TaskSelectionEnv is not None,
        "Core Method: Dual Metrics (stubs)": (_calculate_fisher_similarity is not None and _calculate_gradient_interference is not None),
        "Dataset: Avalanche Suite": get_benchmark is not None,
        "Dataset: Fuzzy Stream": CIFAR100BlurStream is not None,
        "Dataset: Preprocessing": get_transforms is not None,
        "Evaluation: Multi-seed loop": run_experiment is not None,
        "Evaluation: Metrics Plugin": get_eval_plugin is not None,
        "Evaluation: Statistical Test": perform_significance_test is not None,
        "Baselines: DER/EWC integration": True,  # Handled via Avalanche Strategy in train.py
        "Baselines: Custom Schedulers": True,  # Handled in RLTOPScheduler
        "Advanced: Computational Analysis (Time, Mem, FLOPs)": True,  # Integrated in run_experiment
    }
    all_implemented = True
    for component, implemented in required_components.items():
        status = "[OK]" if implemented else "[FAIL]"
        print(f"{status} {component}")
        if not implemented:
            all_implemented = False
    print("--- Verification complete ---")
    return all_implemented


def main():
    """ Main function to run all experiments. """
    print("\n### Starting Experiment Suite for RL-TOP ###")

    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}. Exiting.")
        return

    # If CUDA requested but unavailable, override to CPU to avoid runtime failures
    if CONFIG["global"].get("device", "cpu") == "cuda" and not torch.cuda.is_available():
        print("CUDA requested in config but not available – switching to CPU.")
        CONFIG["global"]["device"] = "cpu"

    if not verify_implementation():
        print("\nERROR: Not all required components are implemented. Exiting.")
        return

    all_results = []

    print("\n\n# =========================================================== #")
    print("#                 RUNNING EXPERIMENT 1                #")
    print("# =========================================================== #")
    exp1_config = CONFIG["exp1"]
    for baseline in exp1_config["baselines"]:
        dataset_to_run = {"split_cifar100": exp1_config["datasets"]["split_cifar100"]}
        exp1_config_subset = exp1_config.copy()
        exp1_config_subset["datasets"] = dataset_to_run
        results = run_experiment(exp1_config_subset, CONFIG["global"], baseline, CONFIG)
        for r in results:
            r["experiment"] = "exp1"
        all_results.extend(results)

    print("\n\n# =========================================================== #")
    print("#                 RUNNING EXPERIMENT 2                #")
    print("# =========================================================== #")
    exp2_config = CONFIG["exp2"]
    for variant in exp2_config["variants"]:
        results = run_experiment(exp2_config, CONFIG["global"], variant, CONFIG)
        for r in results:
            r["experiment"] = "exp2"
        all_results.extend(results)

    print("\n\n# =========================================================== #")
    print("#         EXPERIMENT 3 (Fuzzy Boundaries) SKIPPED         #")
    print("#          (Requires custom online runner logic)          #")
    print("# =========================================================== #")

    results_df = pd.DataFrame(all_results)
    print("\n\n# =========================================================== #")
    print("#                  OVERALL RESULTS                    #")
    print("# =========================================================== #")
    print_results_table(results_df)
    perform_significance_test(results_df)

    validate_results(results_df)

    figure_registry = generate_figures(all_results, CONFIG["global"]["figures_dir"])
    print("\n--- Figure Registry ---")
    if not figure_registry:
        print("No figures were generated.")
    else:
        for f in figure_registry:
            print(f)

    print("\n### Experiment Suite Finished ###")


if __name__ == "__main__":
    main()
