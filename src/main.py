import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import get_synthetic_cifar100_tasks, get_synthetic_permuted_mnist_tasks
from train import SimpleCNN, SimpleMLP, train_model, train_variant
from evaluate import (plot_results, plot_ablation_results, analyze_block_FIM, 
                     compute_dummy_block_FIM, calculate_backward_transfer, evaluate_memory_usage)

def experiment_1_benchmarking():
    """Experiment 1: Benchmarking Against Baselines on Synthetic CIFAR100 tasks"""
    print('\n' + '='*60)
    print('EXPERIMENT 1: BENCHMARKING TG-SMP-BDF vs BASELINE')
    print('='*60)
    
    tasks_cifar = get_synthetic_cifar100_tasks(num_tasks=2, num_classes=10, samples_per_task=64)
    
    print('\nTraining baseline (diagonal FIM) model...')
    model_diag = SimpleCNN(num_classes=10)
    if torch.cuda.is_available():
        model_diag = model_diag.cuda()
    acc_diag = train_model(model_diag, tasks_cifar, use_tg_smp_bdf=False, num_epochs=1, experiment_label="Baseline")
    
    print('\nTraining TG-SMP-BDF model...')
    model_tg = SimpleCNN(num_classes=10)
    if torch.cuda.is_available():
        model_tg = model_tg.cuda()
    acc_tg = train_model(model_tg, tasks_cifar, use_tg_smp_bdf=True, num_epochs=1, experiment_label="TG-SMP-BDF")
    
    save_dir = ".research/iteration1/images"
    plot_results(acc_diag, 'Baseline Diagonal FIM Accuracy', f'{save_dir}/accuracy_baseline.pdf')
    plot_results(acc_tg, 'TG-SMP-BDF Accuracy', f'{save_dir}/accuracy_tg_smp_bdf.pdf')
    
    bwt_improvement = calculate_backward_transfer(acc_diag, acc_tg)
    
    print(f'\nExperiment 1 Results:')
    print(f'Baseline accuracies: {acc_diag}')
    print(f'TG-SMP-BDF accuracies: {acc_tg}')
    print(f'Backward Transfer Improvement: {bwt_improvement:.2f}%')
    
    return acc_diag, acc_tg, bwt_improvement

def experiment_2_ablation_study():
    """Experiment 2: Ablation Study on Synthetic Permuted-MNIST tasks"""
    print('\n' + '='*60)
    print('EXPERIMENT 2: ABLATION STUDY')
    print('='*60)
    
    tasks_mnist = get_synthetic_permuted_mnist_tasks(num_tasks=2, samples_per_task=64)
    
    variants = {
        'Full': {'clustering': True,  'block_FIM': True,  'adaptive_buffer': True},
        'NoClustering': {'clustering': False, 'block_FIM': True,  'adaptive_buffer': True},
        'NoBlockFIM': {'clustering': True, 'block_FIM': False,  'adaptive_buffer': True},
        'NoAdaptiveBuffer': {'clustering': True, 'block_FIM': True,  'adaptive_buffer': False}
    }
    
    results_ablation = {}
    for variant_name, config in variants.items():
        print(f'\nTraining variant: {variant_name}')
        model_variant = SimpleMLP(input_dim=28*28, num_classes=10)
        if torch.cuda.is_available():
            model_variant = model_variant.cuda()
        acc_variant = train_variant(model_variant, tasks_mnist, config, num_epochs=1, experiment_label=variant_name)
        results_ablation[variant_name] = acc_variant
    
    save_dir = ".research/iteration1/images"
    plot_ablation_results(results_ablation, f'{save_dir}/accuracy_ablation.pdf')
    
    print(f'\nExperiment 2 Results:')
    for variant, accs in results_ablation.items():
        print(f'{variant}: {accs}')
    
    return results_ablation

def experiment_3_eigen_spectrum_analysis():
    """Experiment 3: Eigen-Spectrum and Sensitivity Analysis"""
    print('\n' + '='*60)
    print('EXPERIMENT 3: EIGEN-SPECTRUM ANALYSIS')
    print('='*60)
    
    model = SimpleCNN(num_classes=10)
    if torch.cuda.is_available():
        model = model.cuda()
    
    dummy_blocks = compute_dummy_block_FIM(model, num_blocks=3)
    
    eigenvalue_stats = []
    for idx, block in enumerate(dummy_blocks):
        eigen_vals, stats = analyze_block_FIM(block, block_id=idx, save_dir=".research/iteration1/images")
        eigenvalue_stats.append(stats)
    
    print(f'\nExperiment 3 Results:')
    for idx, stats in enumerate(eigenvalue_stats):
        print(f'Block {idx}: Mean={stats["mean"]:.4f}, Std={stats["std"]:.4f}, Range=[{stats["min"]:.4f}, {stats["max"]:.4f}]')
    
    return eigenvalue_stats

def main():
    """Main experiment execution function."""
    print('Starting TG-SMP-BDF Continual Learning Experiments...')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    
    evaluate_memory_usage()
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    try:
        acc_diag, acc_tg, bwt_improvement = experiment_1_benchmarking()
        results_ablation = experiment_2_ablation_study()
        eigenvalue_stats = experiment_3_eigen_spectrum_analysis()
        
        print('\n' + '='*60)
        print('EXPERIMENT SUMMARY')
        print('='*60)
        print(f'Experiment 1 - Baseline vs TG-SMP-BDF:')
        print(f'  Baseline final accuracy: {acc_diag[-1]:.2f}%')
        print(f'  TG-SMP-BDF final accuracy: {acc_tg[-1]:.2f}%')
        print(f'  Backward Transfer Improvement: {bwt_improvement:.2f}%')
        
        print(f'\nExperiment 2 - Ablation Study:')
        for variant, accs in results_ablation.items():
            print(f'  {variant}: Final accuracy {accs[-1]:.2f}%')
        
        print(f'\nExperiment 3 - Eigen-Spectrum Analysis:')
        print(f'  Analyzed {len(eigenvalue_stats)} blocks with eigenvalue distributions')
        
        print(f'\nAll experiments completed successfully!')
        print(f'Results saved to .research/iteration1/images/')
        
        status_enum = "stopped"
        print(f'\nStatus: {status_enum}')
        
    except Exception as e:
        print(f'\nError during experiment execution: {str(e)}')
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
