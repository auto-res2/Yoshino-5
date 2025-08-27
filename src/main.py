import os
import sys
import random
import pathlib
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.experiment_config import (
    SEED, DEVICE, MODEL_CONFIG, TRAINING_CONFIG, 
    LOSS_CONFIG, SYNTHETIC_CONFIG
)
from src.preprocess import SAPTraceDataset, TraceCollator, generate_synthetic_data, validate_trace_data
from src.train import SapStudent, experiment1_train
from src.evaluate import experiment2_safety, experiment3_neuron, generate_probe_prompts

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def quick_test():
    print("=== SAP-CoTD Quick Test Mode ===")
    set_seed()
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    synthetic_file = os.path.join(data_dir, "synthetic_test.jsonl")
    generate_synthetic_data(
        num_samples=SYNTHETIC_CONFIG["num_samples"],
        min_len=SYNTHETIC_CONFIG["min_seq_len"],
        max_len=SYNTHETIC_CONFIG["max_seq_len"],
        trace_dim=SYNTHETIC_CONFIG["trace_dim"],
        mask_dim=SYNTHETIC_CONFIG["mask_dim"],
        vocab_size=SYNTHETIC_CONFIG["vocab_size"],
        output_file=synthetic_file
    )
    
    print(f"Generated synthetic data: {synthetic_file}")
    stats = validate_trace_data(synthetic_file)
    print(f"Data validation: {stats}")
    
    dataset = SAPTraceDataset(synthetic_file)
    loader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], 
                       shuffle=True, collate_fn=TraceCollator())
    
    model = SapStudent(
        vocab=MODEL_CONFIG["vocab_size"],
        hidden=MODEL_CONFIG["hidden_dim"],
        layers=MODEL_CONFIG["num_layers"],
        rank=MODEL_CONFIG["lora_rank"]
    ).to(DEVICE)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, loader, synthetic_file

def run_full_experiment():
    print("=== SAP-CoTD Full Experiment Suite ===")
    set_seed()
    
    images_dir = ".research/iteration1/images"
    os.makedirs(images_dir, exist_ok=True)
    
    model, loader, data_file = quick_test()
    
    print("\n--- Experiment 1: Training ---")
    losses, loss_plot = experiment1_train(
        train_loader=loader,
        model=model,
        device=DEVICE,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"],
        lambda_kl=LOSS_CONFIG["lambda_kl"],
        beta_sup=LOSS_CONFIG["beta_sup"],
        gamma_nce=LOSS_CONFIG["gamma_nce"],
        out_dir=images_dir
    )
    
    print(f"Training completed. Final loss: {losses[-1]:.4f}")
    print(f"Loss plot saved: {loss_plot}")
    
    print("\n--- Experiment 2: Safety Evaluation ---")
    probe_prompts = generate_probe_prompts(50)
    leak_rate, cka_score, safety_plot = experiment2_safety(
        student=model,
        probe_prompts=probe_prompts,
        device=DEVICE,
        out_dir=images_dir
    )
    
    print(f"Safety evaluation completed. Leak rate: {leak_rate:.4f}, CKA: {cka_score:.4f}")
    print(f"Safety plot saved: {safety_plot}")
    
    print("\n--- Experiment 3: Neuron Analysis ---")
    before_val, after_val, neuron_plot = experiment3_neuron(
        student=model,
        device=DEVICE,
        out_dir=images_dir
    )
    
    print(f"Neuron analysis completed. Before: {before_val:.4f}, After: {after_val:.4f}")
    print(f"Neuron plot saved: {neuron_plot}")
    
    print(f"\n=== Experiment Complete ===")
    print(f"All results saved to: {images_dir}")
    print(f"Generated files:")
    for file in pathlib.Path(images_dir).glob("*.pdf"):
        print(f"  - {file}")
    
    return {
        "training_loss": losses[-1] if losses else 0,
        "leak_rate": leak_rate,
        "cka_score": cka_score,
        "neuron_before": before_val,
        "neuron_after": after_val,
        "plots": [loss_plot, safety_plot, neuron_plot]
    }

def main():
    parser = argparse.ArgumentParser(description="SAP-CoTD Experiment Suite")
    parser.add_argument("--synthetic", action="store_true", 
                       help="Run in synthetic test mode only")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation test")
    
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    
    if args.quick:
        model, loader, data_file = quick_test()
        print("Quick test completed successfully!")
        return
    
    if args.synthetic:
        print("Running synthetic test mode...")
        model, loader, data_file = quick_test()
        
        print("Testing single training step...")
        for batch in loader:
            inp = batch["input_ids"].to(DEVICE)
            logits, hidden = model(inp)
            print(f"Forward pass successful. Logits shape: {logits.shape}, Hidden shape: {hidden.shape}")
            break
        
        print("Synthetic test completed successfully!")
        return
    
    results = run_full_experiment()
    print(f"\nFinal Results Summary:")
    for key, value in results.items():
        if key != "plots":
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
