#!/usr/bin/env python3

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import create_datasets, VOCAB
from train import create_model_variants, train_epoch
from evaluate import run_comprehensive_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path(".research/iteration1/images")
RESULTS_DIR = Path(".research/iteration1")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def setup_directories():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def print_system_info():
    print("=" * 60)
    print("PIVOT-XR Experimental Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"Random seed: {SEED}")
    print("=" * 60)

def run_experiment():
    
    print_system_info()
    setup_directories()
    
    print("\n📊 Creating datasets...")
    datasets = create_datasets()
    print(f"Created {len(datasets)} datasets:")
    for name, ds in datasets.items():
        print(f"  - {name}: {len(ds)} samples")
    
    loaders = {
        name: DataLoader(ds, batch_size=32, shuffle=True)
        for name, ds in datasets.items()
    }
    
    print("\n🤖 Creating model variants...")
    vocab_size = len(VOCAB)
    models = create_model_variants(vocab_size)
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - {name}: {num_params:,} total params, {trainable_params:,} trainable")
    
    num_epochs = 3
    learning_rate = 1e-3
    
    print(f"\n🚀 Starting training ({num_epochs} epochs, lr={learning_rate})...")
    
    training_results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        loss_history = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = train_epoch(model, loaders['clean'], optimizer, DEVICE)
            loss_history.append(epoch_loss)
            print(f"  Epoch {epoch+1}/{num_epochs}: loss = {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        training_results[model_name] = {
            'loss_history': loss_history,
            'training_time': training_time
        }
        
        print(f"  Training completed in {training_time:.1f}s")
    
    print("\n📈 Running comprehensive evaluation...")
    
    results = {
        'training_losses': {name: res['loss_history'] for name, res in training_results.items()},
        'training_times': {name: res['training_time'] for name, res in training_results.items()}
    }
    
    eval_results = run_comprehensive_evaluation(models, datasets, DEVICE, SAVE_DIR)
    results.update(eval_results)
    
    print("\n📋 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("\n🎯 Accuracy Results:")
    for model_name in models.keys():
        print(f"\n{model_name}:")
        for ds_name, acc in results['accuracies'][model_name].items():
            print(f"  {ds_name:12s}: {acc*100:5.1f}%")
    
    print("\n⚡ Performance Metrics:")
    for model_name, metrics in results['performance_metrics'].items():
        print(f"\n{model_name}:")
        print(f"  Latency: {metrics['latency_mean']:.1f} ± {metrics['latency_std']:.1f} ms")
        print(f"  Parameters: {metrics['num_parameters']:,}")
        if 'memory_mb' in metrics:
            print(f"  GPU Memory: {metrics['memory_mb']:.1f} MB")
    
    if 'aor_analysis' in results and 'alphas' in results['aor_analysis']:
        print("\n🧠 AOR Head Analysis:")
        alphas = np.concatenate(results['aor_analysis']['alphas'])
        print(f"  Mean α: {alphas.mean():.3f} ± {alphas.std():.3f}")
        print(f"  Order-sensitive heads (α > 0.5): {(alphas > 0.5).sum()}/{len(alphas)}")
        print(f"  Order-invariant heads (α < 0.5): {(alphas < 0.5).sum()}/{len(alphas)}")
    
    results_file = RESULTS_DIR / "experiment_results.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_json = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"📊 Plots saved to: {SAVE_DIR}")
    
    plot_files = list(SAVE_DIR.glob("*.pdf"))
    if plot_files:
        print(f"\n📈 Generated plots:")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file.name}")
    
    print("\n✅ Experiment completed successfully!")
    
    status_file = RESULTS_DIR / "status.json"
    with open(status_file, 'w') as f:
        json.dump({"status_enum": "stopped"}, f)
    
    print(f"🛑 Status set to 'stopped' in {status_file}")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        status_file = Path(".research/iteration1/status.json")
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump({"status_enum": "stopped"}, f)
        
        sys.exit(1)
