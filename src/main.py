import os
import sys
import yaml
import random
import numpy as np
import torch
from types import SimpleNamespace
from collections import defaultdict
import gc
from preprocess import get_benchmark
from train import run_strategy
from evaluate import aggregate_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_results(results, config):
    """Validate that results meet experimental design expectations."""
    print("--- Validating Results against Expectations ---")
    if hasattr(config, 'ci_gate'):
        summary = aggregate_results(results, config.policies)
        aacc_rand = np.mean(summary['random']['AACC'])
        aacc_rl = np.mean(summary['rl_top']['AACC'])
        af_rl = np.mean(summary['rl_top']['AF'])
        af_rand = np.mean(summary['random']['AF'])

        checks = {
            'AACC_random >= min': aacc_rand >= config.ci_gate['aacc_random_min'],
            'AACC_rl >= AACC_random': aacc_rl >= aacc_rand,
            'AF_rl <= 0.5 * AF_random': af_rl <= 0.5 * af_rand
        }
        for check, status in checks.items():
            print(f"  - CI Gate '{check}': {'PASS' if status else 'FAIL'}")
        if not all(checks.values()):
            print("CI validation FAILED.")
            sys.exit(1)
        else:
            print("CI validation PASSED.")
    elif hasattr(config, 'results_gate'):
        rl_results = results['rl_top']
        aacc = np.mean([r['AACC'] for r in rl_results])
        af = np.mean([r['AF'] for r in rl_results])
        checks = {
            f'AACC >= {config.results_gate["aacc_min"]}': aacc >= config.results_gate["aacc_min"],
            f'AF <= {config.results_gate["af_max"]}': af <= config.results_gate["af_max"],
        }
        for check, status in checks.items():
            print(f"  - Result Gate '{check}': {'PASS' if status else 'FAIL (Expected)'}")
        if not all(checks.values()):
            print("WARNING: Main experiment did not meet performance targets.")
        else:
            print("SUCCESS: Main experiment met performance targets!")
    else:
        print("No result validation gates found for this experiment.")
    print()

def main():
    """Main execution script for the experiment."""
    config_path = 'config/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    print("\n" + "#"*60)
    print(f"# Running Experiment: {config.experiment_name}")
    print(f"# Description: {config.description}")
    print("#"*60 + "\n")

    all_results = defaultdict(list)

    for policy in config.policies:
        for seed in config.seeds:
            print(f"\n>>> Running Policy: {policy}, Seed: {seed} <<<")
            set_seed(seed)
            
            run_config = SimpleNamespace(**config.__dict__)
            run_config.policy = policy
            run_config.seed = seed

            benchmark, _ = get_benchmark(run_config)
            results = run_strategy(run_config, benchmark)
            all_results[policy].append(results)
            
            gc.collect()
            torch.cuda.empty_cache()

    aggregate_results(all_results, config.policies)
    
    if hasattr(config, 'results_gate') or hasattr(config, 'ci_gate'):
        validate_results(all_results, config)

if __name__ == '__main__':
    main()
