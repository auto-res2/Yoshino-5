#!/usr/bin/env python3
"""
HyperST-LoRA Main Experiment Script
Implements video understanding with adaptive LoRA rank and spectral gating.
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import HyperSTConfig, HyperSTModel, set_seed
from preprocess import create_dataloader
from evaluate import experiment1, experiment2, experiment3

def set_status(status: str):
    status_file = Path(".research/iteration1/status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    status_data = {
        "status_enum": status,
        "timestamp": time.time(),
        "experiment": "HyperST-LoRA"
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status set to: {status}")

def quick_test():
    print("===== QUICK FUNCTIONALITY TEST =====")
    set_seed(123)
    
    model = HyperSTModel(HyperSTConfig())
    loader = create_dataloader(32, 8)
    
    for vid, q, ans, _ in loader:
        pred, rank, T = model(vid, q)
        assert pred.shape == ans.shape
        print(f"Forward-pass ✓ | rank={rank} | T={T}")
        break
    
    print("Quick test completed successfully!")

def main():
    print("Starting HyperST-LoRA Experiments")
    print("=" * 50)
    
    set_status("running")
    
    try:
        quick_test()
        
        output_dir = ".research/iteration1/images"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 50)
        print("RUNNING FULL EXPERIMENTS")
        print("=" * 50)
        
        model = HyperSTModel(HyperSTConfig())
        loader = create_dataloader(240, 16)
        exp1_results = experiment1(model, loader)
        
        exp2_results = experiment2(output_dir)
        
        exp3_results = experiment3(output_dir)
        
        results_summary = {
            "experiment1": exp1_results,
            "experiment2": {k: {"acc": v[0], "flops": v[1], "params": v[2]} 
                           for k, v in exp2_results.items()},
            "experiment3": {f"{k/1e6:.0f}M": v for k, v in exp3_results.items()}
        }
        
        with open(f"{output_dir}/results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\n" + "=" * 50)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Results saved to: {output_dir}")
        print("PDF plots generated:")
        print(f"  - {output_dir}/accuracy_ablation.pdf")
        print(f"  - {output_dir}/accuracy_budget_curve.pdf")
        
        set_status("stopped")
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        set_status("error")
        raise

if __name__ == "__main__":
    main()
