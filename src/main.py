#!/usr/bin/env python3
"""
Main experiment script for Adaptive MambaFormer research.
Implements dynamic computation scheduling vs static hybrid baseline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from train import train_model
from evaluate import evaluate_model
from preprocess import create_synthetic_dataset

torch.manual_seed(42)
np.random.seed(42)

class AdaptiveMambaFormer(nn.Module):
    def __init__(self, config, disable_spars=False, use_gate=True, heuristic_gate=False):
        """
        AdaptiveMambaFormer with options for ablation experiments.
        """
        super(AdaptiveMambaFormer, self).__init__()
        self.config = config
        self.disable_spars = disable_spars
        self.use_gate = use_gate
        self.heuristic_gate = heuristic_gate

        self.full_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=8, 
            batch_first=False
        )
        self.linear_ssm = nn.Linear(config['d_model'], config['d_model'])
        self.sparse_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=4, 
            batch_first=False
        )

        if self.use_gate:
            self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, x):
        token_stats = torch.mean(x, dim=-1, keepdim=True)
        
        if self.use_gate:
            if self.heuristic_gate:
                gate_weights = torch.full(token_stats.shape, 0.3, device=x.device)
            else:
                gate_weights = self.gate(token_stats)
        else:
            gate_weights = torch.full(token_stats.shape, 0.5, device=x.device)

        threshold = 0.5
        full_attention_mask = gate_weights > threshold

        out_full, _ = self.full_attention(x, x, x)
        out_sparse, _ = self.sparse_attention(x, x, x)
        out_linear = self.linear_ssm(x)

        if self.disable_spars:
            output = out_full
        else:
            gate_weights_expanded = gate_weights.expand_as(x)
            output = gate_weights_expanded * out_full + (1 - gate_weights_expanded) * out_sparse

        output = output + out_linear
        
        branch_usage = torch.mean((gate_weights > threshold).float()).item()
        return output, branch_usage


class StaticMambaFormer(nn.Module):
    def __init__(self, config):
        super(StaticMambaFormer, self).__init__()
        self.full_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=8, 
            batch_first=False
        )
        self.linear_ssm = nn.Linear(config['d_model'], config['d_model'])
        self.sparse_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=4, 
            batch_first=False
        )

    def forward(self, x):
        out_full, _ = self.full_attention(x, x, x)
        out_linear = self.linear_ssm(out_full)
        out_sparse, _ = self.sparse_attention(out_linear, out_linear, out_linear)
        output = out_sparse + out_linear
        return output


def measure_inference_time(model, dataloader, device):
    model.eval()
    times = []
    branch_usage_stats = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
            start_time = time.time()
            out = model(x)
            if isinstance(out, tuple):
                _, branch_usage = out
                branch_usage_stats.append(branch_usage)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    avg_branch_usage = sum(branch_usage_stats)/len(branch_usage_stats) if branch_usage_stats else None
    return avg_time, avg_branch_usage


def plot_and_save_curve(y_values, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(y_values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot as {filename}")


def experiment_1_inference_speed(device):
    """Experiment 1: Inference Speed vs. Accuracy Trade-off"""
    print("\n--- Experiment 1: Inference Speed vs. Accuracy Trade-off ---")
    
    config = {'d_model': 256}
    batch_size = 8
    seq_len = 32
    num_batches = 20
    
    synthetic_data = []
    for i in range(num_batches):
        data = torch.randn(batch_size, seq_len, config['d_model'])
        if i % 5 == 0:  # simulate high complexity
            data[:, 0] += 5.0
        synthetic_data.append(data)
    
    all_data = torch.cat(synthetic_data, dim=0)
    dataloader = DataLoader(TensorDataset(all_data), batch_size=batch_size)

    model_adaptive = AdaptiveMambaFormer(config).to(device)
    model_static = StaticMambaFormer(config).to(device)

    time_adaptive, branch_usage = measure_inference_time(model_adaptive, dataloader, device)
    time_static, _ = measure_inference_time(model_static, dataloader, device)
    
    print(f"Adaptive Model Inference Time: {time_adaptive:.6f}s (Branch usage: {branch_usage:.3f})")
    print(f"Static Model Inference Time: {time_static:.6f}s")
    
    labels = ['Adaptive', 'Static']
    times = [time_adaptive, time_static]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['blue', 'green'])
    plt.ylabel('Avg Inference Time (s)')
    plt.title('Inference Time Comparison: Adaptive vs Static MambaFormer')
    plt.grid(True, alpha=0.3)
    filename = '.research/iteration1/images/inference_latency.pdf'
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Inference latency plot saved as {filename}")
    
    return time_adaptive, time_static, branch_usage


def experiment_2_ablation_study(device):
    """Experiment 2: Ablation Study on Components"""
    print("\n--- Experiment 2: Ablation Study on Scheduler and Sparsification Components ---")
    
    config = {'d_model': 256}
    batch_size = 8
    seq_len = 32
    num_samples = 100
    
    synthetic_data = torch.randn(num_samples, seq_len, config['d_model'])
    for i in range(num_samples):
        if i % 10 == 0:
            synthetic_data[i, 0] += 5.0
    
    dataset = TensorDataset(synthetic_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    
    model_full = AdaptiveMambaFormer(config).to(device)
    model_no_spars = AdaptiveMambaFormer(config, disable_spars=True).to(device)
    model_no_gate = AdaptiveMambaFormer(config, use_gate=True, heuristic_gate=True).to(device)
    model_fixed = StaticMambaFormer(config).to(device)

    optim_full = optim.Adam(model_full.parameters(), lr=0.001)
    optim_no_spars = optim.Adam(model_no_spars.parameters(), lr=0.001)
    optim_no_gate = optim.Adam(model_no_gate.parameters(), lr=0.001)
    optim_fixed = optim.Adam(model_fixed.parameters(), lr=0.001)

    print("Training models...")
    loss_full = train_model(model_full, train_loader, optim_full, criterion, num_epochs=3, device=device)
    loss_no_spars = train_model(model_no_spars, train_loader, optim_no_spars, criterion, num_epochs=3, device=device)
    loss_no_gate = train_model(model_no_gate, train_loader, optim_no_gate, criterion, num_epochs=3, device=device)
    loss_fixed = train_model(model_fixed, train_loader, optim_fixed, criterion, num_epochs=3, device=device)

    plt.figure(figsize=(12, 8))
    plt.plot(loss_full, marker='o', label='Full Adaptive', linewidth=2)
    plt.plot(loss_no_spars, marker='s', label='No Sparsification', linewidth=2)
    plt.plot(loss_no_gate, marker='^', label='Heuristic Gate', linewidth=2)
    plt.plot(loss_fixed, marker='d', label='Static', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: Ablation Study')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = '.research/iteration1/images/training_loss.pdf'
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved as {filename}")
    
    return loss_full, loss_no_spars, loss_no_gate, loss_fixed


def experiment_3_real_time_simulation(device):
    """Experiment 3: On-Device Real-Time Simulation"""
    print("\n--- Experiment 3: On-Device Real-Time Simulation ---")
    
    config = {'d_model': 256}
    model_adaptive = AdaptiveMambaFormer(config).to(device)
    model_static = StaticMambaFormer(config).to(device)

    def simulate_real_time_inference(model, num_iterations):
        model.eval()
        latencies = []
        branch_decisions = []
        with torch.no_grad():
            for i in range(num_iterations):
                seq_len = 32
                batch_size = 1
                input_data = torch.randn(seq_len, batch_size, config['d_model']).to(device)
                if i % 7 == 0:
                    input_data[0] += 5.0
                
                start = time.time()
                output = model(input_data)
                if isinstance(output, tuple):
                    _, branch_usage = output
                    branch_decisions.append(branch_usage)
                end = time.time()
                
                latency = end - start
                latencies.append(latency)
                
                cpu_usage = psutil.cpu_percent(interval=None)
                mem_usage = psutil.virtual_memory().percent
                if i % 5 == 0:
                    print(f"Iteration {i}: latency = {latency:.4f}s, CPU = {cpu_usage}%, Memory = {mem_usage}%")
        
        return latencies, branch_decisions

    print("Simulating real-time inference...")
    latencies_adaptive, branch_logs_adaptive = simulate_real_time_inference(model_adaptive, 20)
    latencies_static, _ = simulate_real_time_inference(model_static, 20)

    plt.figure(figsize=(12, 8))
    plt.plot(latencies_adaptive, marker='o', label='Adaptive', linewidth=2, markersize=6)
    plt.plot(latencies_static, marker='s', label='Static', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Latency (s)')
    plt.title('Real-Time Inference Latency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = '.research/iteration1/images/inference_latency_real_time.pdf'
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Real-time inference latency plot saved as {filename}")
    
    return latencies_adaptive, latencies_static


def main():
    """Main experiment execution"""
    print("Starting Adaptive MambaFormer Experiments")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    
    try:
        time_adaptive, time_static, branch_usage = experiment_1_inference_speed(device)
        loss_results = experiment_2_ablation_study(device)
        latency_results = experiment_3_real_time_simulation(device)
        
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Experiment 1 - Inference Speed:")
        print(f"  Adaptive Model: {time_adaptive:.6f}s (Branch usage: {branch_usage:.3f})")
        print(f"  Static Model: {time_static:.6f}s")
        print(f"  Speed improvement: {((time_static - time_adaptive) / time_static * 100):.2f}%")
        
        print(f"\nExperiment 2 - Ablation Study:")
        print(f"  Training completed for all model variants")
        print(f"  Loss curves saved to training_loss.pdf")
        
        print(f"\nExperiment 3 - Real-time Simulation:")
        print(f"  Real-time latency comparison completed")
        print(f"  Results saved to inference_latency_real_time.pdf")
        
        print(f"\nAll experiments completed successfully!")
        print(f"Results saved in .research/iteration1/images/")
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
