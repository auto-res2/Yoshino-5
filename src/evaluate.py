import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.train import DCAMFLPP, train_model
from src.preprocess import create_few_shot_dataloader, add_modality_conflict

def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for vision, text, labels in dataloader:
            vision, text, labels = vision.to(device), text.to(device), labels.to(device)
            outputs = model(vision, text)
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def experiment_benchmark_comparison(save_dir='.research/iteration1/images'):
    """
    Experiment 1: Compare DCAMFL++ Dynamic vs Static Fusion
    """
    print('=== Experiment 1: Benchmark Comparison ===')
    
    dataloader = create_few_shot_dataloader(batch_size=16, num_samples=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model_dynamic = DCAMFLPP()
    
    model_static = DCAMFLPP()
    for param in model_static.gating.parameters():
        param.requires_grad = False
    
    epochs = 3
    
    print('\nTraining Dynamic DCAMFL++ model...')
    losses_dynamic = train_model(model_dynamic, dataloader, epochs=epochs, device=device)
    
    print('\nTraining Static Fusion model...')
    losses_static = train_model(model_static, dataloader, epochs=epochs, device=device)
    
    acc_dynamic = evaluate_model(model_dynamic, dataloader, device)
    acc_static = evaluate_model(model_static, dataloader, device)
    
    print(f'\nResults:')
    print(f'Dynamic DCAMFL++ Accuracy: {acc_dynamic:.3f}')
    print(f'Static Fusion Accuracy: {acc_static:.3f}')
    print(f'Improvement: {((acc_dynamic - acc_static) / acc_static * 100):.1f}%')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses_dynamic, 'b-o', label='Dynamic DCAMFL++', linewidth=2)
    plt.plot(range(1, epochs+1), losses_static, 'r--s', label='Static Fusion', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison: Dynamic vs Static Fusion', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/benchmark_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    models = ['Dynamic DCAMFL++', 'Static Fusion']
    accuracies = [acc_dynamic, acc_static]
    colors = ['#2E86AB', '#A23B72']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylim(0, 1.0)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plots saved to {save_dir}/')
    return {'dynamic_acc': acc_dynamic, 'static_acc': acc_static}

def experiment_ablation_study(save_dir='.research/iteration1/images'):
    """
    Experiment 2: Ablation Study of DCAMFL++ Components
    """
    print('\n=== Experiment 2: Ablation Study ===')
    
    dataloader = create_few_shot_dataloader(batch_size=16, num_samples=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    variations = {
        'Full DCAMFL++': DCAMFLPP(),
        'No Uncertainty': DCAMFLPP(),
        'No Contrastive Loss': DCAMFLPP()
    }
    
    results = {}
    epochs = 3
    
    for name, model in variations.items():
        print(f'\nTraining {name}...')
        
        if name == 'No Uncertainty':
            for param in model.uncertainty_vision.parameters():
                param.requires_grad = False
            for param in model.uncertainty_text.parameters():
                param.requires_grad = False
        
        losses = train_model(model, dataloader, epochs=epochs, device=device)
        
        accuracy = evaluate_model(model, dataloader, device)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.3f}')
    
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[name] for name in names]
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    
    bars = plt.bar(names, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ablation Study: Component Contributions', fontsize=14)
    plt.xticks(rotation=15)
    plt.ylim(0, max(accuracies) * 1.1)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Ablation study plot saved to {save_dir}/')
    return results

def experiment_robustness_to_conflict(save_dir='.research/iteration1/images'):
    """
    Experiment 3: Robustness to Conflicting Modalities
    """
    print('\n=== Experiment 3: Robustness to Conflicting Modalities ===')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCAMFLPP().to(device)
    
    conflict_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []
    
    for conflict_level in conflict_levels:
        print(f'\nTesting conflict level: {conflict_level}')
        
        test_model = DCAMFLPP().to(device)
        
        dataloader = create_few_shot_dataloader(batch_size=16, num_samples=100)
        
        test_model.train()
        optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(2):  # Quick training
            for vision, text, labels in dataloader:
                vision, text, labels = vision.to(device), text.to(device), labels.to(device)
                
                vision_noisy, text_noisy = add_modality_conflict(
                    vision, text, conflict_level=conflict_level
                )
                
                optimizer.zero_grad()
                outputs = test_model(vision_noisy, text_noisy)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                optimizer.step()
        
        accuracy = evaluate_model(test_model, dataloader, device)
        accuracies.append(accuracy)
        print(f'Accuracy with conflict level {conflict_level}: {accuracy:.3f}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(conflict_levels, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Conflict Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Robustness to Conflicting Modalities', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(accuracies) * 1.1)
    
    for x, y in zip(conflict_levels, accuracies):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/robustness_conflict.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Robustness plot saved to {save_dir}/')
    return {'conflict_levels': conflict_levels, 'accuracies': accuracies}

def run_all_experiments():
    """Run all three experiments and return comprehensive results."""
    print('Starting DCAMFL++ Experimental Evaluation')
    print('=' * 50)
    
    import os
    os.makedirs('.research/iteration1/images', exist_ok=True)
    
    benchmark_results = experiment_benchmark_comparison()
    ablation_results = experiment_ablation_study()
    robustness_results = experiment_robustness_to_conflict()
    
    print('\n' + '=' * 50)
    print('EXPERIMENTAL SUMMARY')
    print('=' * 50)
    print(f'Benchmark - Dynamic: {benchmark_results["dynamic_acc"]:.3f}, Static: {benchmark_results["static_acc"]:.3f}')
    print('Ablation Results:')
    for name, acc in ablation_results.items():
        print(f'  {name}: {acc:.3f}')
    print(f'Robustness - Best: {max(robustness_results["accuracies"]):.3f}, Worst: {min(robustness_results["accuracies"]):.3f}')
    
    return {
        'benchmark': benchmark_results,
        'ablation': ablation_results,
        'robustness': robustness_results
    }
