#!/usr/bin/env python3
"""
DCAMFL++ (Dynamic Contrasting Adaptation for Multimodal Few-Shot Learning++)
Main experimental script for running all experiments.

This script implements and evaluates the DCAMFL++ framework with:
1. Task-Aware Uncertainty Estimation
2. Dynamic Cross-Modal Gating with Contrastive Alignment  
3. Hierarchical Meta-Learning Strategy
4. Cross-Modal Feature Disentanglement

Experiments:
1. Benchmark Comparison (Dynamic vs Static Fusion)
2. Ablation Study (Component Contributions)
3. Robustness to Conflicting Modalities
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate import run_all_experiments
from src.train import DCAMFLPP
from src.preprocess import create_few_shot_dataloader

def check_environment():
    """Check system environment and GPU availability."""
    print("Environment Check:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"NumPy version: {np.__version__}")
    print("-" * 50)

def test_model_instantiation():
    """Quick test to ensure model can be instantiated and run."""
    print("Testing model instantiation...")
    
    try:
        model = DCAMFLPP()
        print(f"✓ Model created successfully")
        
        batch_size = 4
        vision_input = torch.randn(batch_size, 2048)
        text_input = torch.randn(batch_size, 768)
        
        with torch.no_grad():
            outputs = model(vision_input, text_input)
            
        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"  Uncertainty vision shape: {outputs['uncertainty_vision'].shape}")
        print(f"  Uncertainty text shape: {outputs['uncertainty_text'].shape}")
        
        dataloader = create_few_shot_dataloader(batch_size=8, num_samples=32)
        vision, text, labels = next(iter(dataloader))
        print(f"✓ Dataloader working")
        print(f"  Vision batch shape: {vision.shape}")
        print(f"  Text batch shape: {text.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        print("✓ All components working correctly")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        raise e
    
    print("-" * 50)

def main():
    """Main experimental pipeline."""
    print("DCAMFL++ Experimental Framework")
    print("=" * 50)
    
    check_environment()
    
    test_model_instantiation()
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Starting comprehensive experimental evaluation...")
    print("This includes:")
    print("1. Benchmark Comparison (Dynamic vs Static)")
    print("2. Ablation Study (Component Analysis)")
    print("3. Robustness Testing (Conflicting Modalities)")
    print("-" * 50)
    
    try:
        results = run_all_experiments()
        
        print("\n" + "=" * 50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        print("Key Findings:")
        dynamic_acc = results['benchmark']['dynamic_acc']
        static_acc = results['benchmark']['static_acc']
        improvement = ((dynamic_acc - static_acc) / static_acc * 100)
        print(f"• Dynamic DCAMFL++ achieved {improvement:.1f}% improvement over static fusion")
        
        full_model_acc = results['ablation']['Full DCAMFL++']
        print(f"• Full model accuracy: {full_model_acc:.3f}")
        
        robustness_range = max(results['robustness']['accuracies']) - min(results['robustness']['accuracies'])
        print(f"• Robustness range across conflict levels: {robustness_range:.3f}")
        
        print(f"\nAll plots saved to: .research/iteration1/images/")
        print("Files generated:")
        print("• benchmark_comparison.pdf")
        print("• accuracy_comparison.pdf") 
        print("• ablation_study.pdf")
        print("• robustness_conflict.pdf")
        
        status_enum = "stopped"
        print(f"\nExperiment status: {status_enum}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
