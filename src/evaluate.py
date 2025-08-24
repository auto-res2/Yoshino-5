#!/usr/bin/env python3
"""
Evaluation module for Adaptive MambaFormer experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate_model(model, test_loader, criterion, device='cpu', verbose=True):
    """
    Evaluate a model on test data.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    inference_times = []
    branch_usage_stats = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating') if verbose else test_loader
        
        for batch in pbar:
            x = batch[0].to(device)
            x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
            batch_size = x.size(1)  # x is (seq_len, batch_size, d_model)
            
            start_time = time.time()
            output = model(x)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            if isinstance(output, tuple):
                output, branch_usage = output
                branch_usage_stats.append(branch_usage)
            
            loss = criterion(output, x)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            predictions.append(output.cpu())
            targets.append(x.cpu())
            
            if verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / total_samples
    avg_inference_time = np.mean(inference_times)
    avg_branch_usage = np.mean(branch_usage_stats) if branch_usage_stats else None
    
    all_predictions = torch.cat(predictions, dim=1)
    all_targets = torch.cat(targets, dim=1)
    mse_error = torch.mean((all_predictions - all_targets) ** 2).item()
    
    results = {
        'avg_loss': avg_loss,
        'mse_error': mse_error,
        'avg_inference_time': avg_inference_time,
        'total_samples': total_samples,
        'avg_branch_usage': avg_branch_usage
    }
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"MSE Error: {mse_error:.4f}")
        print(f"Average Inference Time: {avg_inference_time:.6f}s")
        print(f"Total Samples: {total_samples}")
        if avg_branch_usage is not None:
            print(f"Average Branch Usage (Full Attention): {avg_branch_usage:.3f}")
    
    return results


def compare_models(models, model_names, test_loader, criterion, device='cpu'):
    """
    Compare multiple models on the same test data.
    
    Args:
        models: List of PyTorch models
        model_names: List of model names
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    print("Comparing models...")
    print("=" * 50)
    
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        model_results = evaluate_model(model, test_loader, criterion, device, verbose=False)
        results[name] = model_results
        
        print(f"{name} Results:")
        print(f"  Loss: {model_results['avg_loss']:.4f}")
        print(f"  MSE Error: {model_results['mse_error']:.4f}")
        print(f"  Inference Time: {model_results['avg_inference_time']:.6f}s")
        if model_results['avg_branch_usage'] is not None:
            print(f"  Branch Usage: {model_results['avg_branch_usage']:.3f}")
    
    return results


def benchmark_inference_speed(model, input_shape, device='cpu', num_runs=100, warmup_runs=10):
    """
    Benchmark inference speed of a model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (seq_len, batch_size, d_model)
        device: Device to benchmark on
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    times = []
    branch_usage_stats = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc='Benchmarking'):
            start_time = time.time()
            output = model(dummy_input)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if isinstance(output, tuple):
                _, branch_usage = output
                branch_usage_stats.append(branch_usage)
    
    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'throughput': 1.0 / np.mean(times),  # inferences per second
        'avg_branch_usage': np.mean(branch_usage_stats) if branch_usage_stats else None
    }
    
    print(f"\nBenchmark Results:")
    print(f"Mean inference time: {results['mean_time']:.6f}s ± {results['std_time']:.6f}s")
    print(f"Min/Max time: {results['min_time']:.6f}s / {results['max_time']:.6f}s")
    print(f"Median time: {results['median_time']:.6f}s")
    print(f"Throughput: {results['throughput']:.2f} inferences/second")
    if results['avg_branch_usage'] is not None:
        print(f"Average branch usage: {results['avg_branch_usage']:.3f}")
    
    return results


def plot_evaluation_comparison(results_dict, save_path=None):
    """
    Plot comparison of evaluation results.
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
    """
    model_names = list(results_dict.keys())
    metrics = ['avg_loss', 'mse_error', 'avg_inference_time']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[name][metric] for name in model_names]
        axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        
        if len(model_names) > 2:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Evaluation comparison plot saved to {save_path}")
    
    plt.close()


def analyze_branch_usage_patterns(model, test_loader, device='cpu'):
    """
    Analyze branch usage patterns for adaptive models.
    
    Args:
        model: Adaptive model
        test_loader: DataLoader for test data
        device: Device to analyze on
    
    Returns:
        Dictionary with branch usage analysis
    """
    if not hasattr(model, 'use_gate') or not model.use_gate:
        print("Model does not support branch usage analysis")
        return None
    
    model.eval()
    branch_usage_per_batch = []
    complexity_indicators = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Analyzing branch usage'):
            x = batch[0].to(device)
            x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
            
            complexity = torch.var(x).item()
            complexity_indicators.append(complexity)
            
            output, branch_usage = model(x)
            branch_usage_per_batch.append(branch_usage)
    
    correlation = np.corrcoef(complexity_indicators, branch_usage_per_batch)[0, 1]
    
    results = {
        'branch_usage_per_batch': branch_usage_per_batch,
        'complexity_indicators': complexity_indicators,
        'correlation': correlation,
        'mean_branch_usage': np.mean(branch_usage_per_batch),
        'std_branch_usage': np.std(branch_usage_per_batch)
    }
    
    print(f"\nBranch Usage Analysis:")
    print(f"Mean branch usage: {results['mean_branch_usage']:.3f}")
    print(f"Std branch usage: {results['std_branch_usage']:.3f}")
    print(f"Correlation with complexity: {results['correlation']:.3f}")
    
    return results
