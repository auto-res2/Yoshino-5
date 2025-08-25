import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def plot_results(accuracies, title, filename):
    """Plot and save accuracy results as PDF."""
    plt.figure(figsize=(8, 6))
    plt.plot(accuracies, marker='o', linewidth=2, markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Task Number', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_ablation_results(results_ablation, filename):
    """Plot ablation study results as grouped bar chart."""
    variants = list(results_ablation.keys())
    task0_acc = [results_ablation[v][0] for v in variants]
    task1_acc = [results_ablation[v][1] for v in variants]
    
    x = np.arange(len(variants))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - width/2, task0_acc, width, label='Task 0', alpha=0.8)
    plt.bar(x + width/2, task1_acc, width, label='Task 1', alpha=0.8)
    plt.xticks(x, variants, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Ablation Study: Task Accuracies by Component', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved ablation study plot to {filename}')

def compute_dummy_block_FIM(model, num_blocks=3):
    """Generate dummy block-diagonal Fisher matrices.
       For each block, create a random symmetric positive semi-definite matrix.
    """
    block_fim_matrices = []
    for b in range(num_blocks):
        A = np.random.rand(10, 10)
        block_matrix = np.dot(A, A.T)
        block_fim_matrices.append(block_matrix)
    return block_fim_matrices

def analyze_block_FIM(block_fim_matrix, block_id=0, save_dir=".research/iteration1/images"):
    """Compute eigenvalues for the block and plot histogram, saving as a PDF."""
    eigenvalues, _ = eig(block_fim_matrix)
    eigenvalues = np.real(eigenvalues)
    
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Eigenvalue Distribution for Block {block_id}', fontsize=14, fontweight='bold')
    plt.xlabel('Eigenvalue', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{save_dir}/eigen_spectrum_block{block_id}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    stats = {
        'mean': np.mean(eigenvalues),
        'std': np.std(eigenvalues),
        'min': np.min(eigenvalues),
        'max': np.max(eigenvalues)
    }
    print(f"[Experiment 3] Block {block_id} eigenvalue stats: {stats}")
    print(f"Saved eigenvalue plot to {filename}")
    return eigenvalues, stats

def calculate_backward_transfer(task_accuracies_baseline, task_accuracies_tg):
    """Calculate backward transfer improvement."""
    if len(task_accuracies_baseline) < 2 or len(task_accuracies_tg) < 2:
        return 0.0
    
    bwt_baseline = task_accuracies_baseline[0] - task_accuracies_baseline[-1]
    bwt_tg = task_accuracies_tg[0] - task_accuracies_tg[-1]
    
    improvement = bwt_tg - bwt_baseline
    print(f"Backward Transfer - Baseline: {bwt_baseline:.2f}%, TG-SMP-BDF: {bwt_tg:.2f}%")
    print(f"BWT Improvement: {improvement:.2f}%")
    return improvement

def evaluate_memory_usage():
    """Check GPU memory usage for Tesla T4 compatibility."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        if memory_allocated > 14.0:  # Leave 2GB buffer for Tesla T4
            print("WARNING: Memory usage approaching Tesla T4 limit (16GB)")
            return False
        return True
    else:
        print("CUDA not available, running on CPU")
        return True
