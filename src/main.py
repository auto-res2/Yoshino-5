import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from preprocess import prepare_experiment_data
from train import train_models
from evaluate import evaluate_model, run_ablation_study, evaluate_scene_complexity

def experiment_1(models, prompts, gt_images, save_dir):
    """Experiment 1: Benchmark Comparison for Text-to-Image Synthesis."""
    print("\n=== Experiment 1: Benchmark Comparison for Text-to-Image Synthesis ===")
    
    model_cdmad, model_baseline, model_teacher, _ = models
    
    fid_cdmad, clip_cdmad, time_cdmad = evaluate_model(model_cdmad, prompts, gt_images)
    fid_baseline, clip_baseline, time_baseline = evaluate_model(model_baseline, prompts, gt_images)
    fid_teacher, clip_teacher, time_teacher = evaluate_model(model_teacher, prompts, gt_images)

    print(f"CDMAD 2.0 -> FID: {fid_cdmad:.4f}, CLIP: {clip_cdmad:.4f}, Avg Gen Time: {time_cdmad:.4f} sec")
    print(f"Baseline  -> FID: {fid_baseline:.4f}, CLIP: {clip_baseline:.4f}, Avg Gen Time: {time_baseline:.4f} sec")
    print(f"Teacher   -> FID: {fid_teacher:.4f}, CLIP: {clip_teacher:.4f}, Avg Gen Time: {time_teacher:.4f} sec")

    models_names = ['CDMAD 2.0', 'Baseline', 'Teacher']
    fid_scores = [fid_cdmad, fid_baseline, fid_teacher]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models_names, y=fid_scores, palette='viridis')
    plt.title('FID Comparison for Text-to-Image Synthesis', fontsize=14)
    plt.ylabel('FID (Lower is better)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fid_comparison.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    clip_scores = [clip_cdmad, clip_baseline, clip_teacher]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models_names, y=clip_scores, palette='magma')
    plt.title('CLIP Score Comparison for Text-to-Image Synthesis', fontsize=14)
    plt.ylabel('CLIP Score (Higher is better)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clip_comparison.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Experiment 1 completed. Plots saved to:", save_dir)

def experiment_2(models, prompts, gt_images, save_dir):
    """Experiment 2: Ablation Study on the Dynamic CGM Module."""
    print("\n=== Experiment 2: Ablation Study on the CGM Module ===")
    
    model_cdmad, _, _, model_static = models
    
    fid_full, clip_full = run_ablation_study(model_cdmad, prompts, gt_images)
    fid_static, clip_static = run_ablation_study(model_static, prompts, gt_images)

    print(f"Full Dynamic Model -> Avg FID: {fid_full:.4f}, Avg CLIP: {clip_full:.4f}")
    print(f"Static Fusion Model -> Avg FID: {fid_static:.4f}, Avg CLIP: {clip_static:.4f}")

    models_names = ['Full Dynamic', 'Static Fusion']
    fid_scores = [fid_full, fid_static]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models_names, y=fid_scores, palette='coolwarm')
    plt.title('Ablation Study: FID Comparison', fontsize=14)
    plt.ylabel('FID (Lower is better)', fontsize=12)
    plt.xlabel('Model Variant', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fid_ablation.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    clip_scores = [clip_full, clip_static]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models_names, y=clip_scores, palette='Spectral')
    plt.title('Ablation Study: CLIP Score Comparison', fontsize=14)
    plt.ylabel('CLIP Score (Higher is better)', fontsize=12)
    plt.xlabel('Model Variant', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clip_ablation.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Experiment 2 completed. Plots saved to:", save_dir)

def experiment_3(models, scenes, save_dir):
    """Experiment 3: Real-Time Performance and Consistency Under Varying Scene Complexity."""
    print("\n=== Experiment 3: Real-Time Performance Under Varying Scene Complexity ===")
    
    model_cdmad, _, _, _ = models
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    performance_records = evaluate_scene_complexity(model_cdmad, scenes, noise_levels)

    for rec in performance_records:
        print(f"Noise: {rec['noise']:.1f} | Scene Complexity: {rec['scene_complexity']:.1f} | "
              f"Exec Time: {rec['exec_time']:.4f} s | FID: {rec['fid']:.4f} | CLIP: {rec['clip_score']:.4f}")

    noise_vals = [rec['noise'] for rec in performance_records]
    exec_times = [rec['exec_time'] for rec in performance_records]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=noise_vals, y=exec_times, hue=noise_vals, palette='viridis', s=100)
    plt.title('Real-Time Performance: Execution Time vs Noise Level', fontsize=14)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_latency.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    fid_vals = [rec['fid'] for rec in performance_records]
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=noise_vals, y=fid_vals, marker='o', linewidth=2, markersize=8)
    plt.title('FID vs Noise Level', fontsize=14)
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('FID (Lower is better)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fid_vs_noise.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Experiment 3 completed. Plots saved to:", save_dir)

def main():
    """Main experiment pipeline for CDMAD 2.0."""
    print("===== CDMAD 2.0 Consistency-guided Dual-Modal Adaptive Diffusion Experiments =====")
    print("Starting comprehensive evaluation of the proposed framework...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = ".research/iteration1/images"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print("\n--- Step 1: Data Preprocessing ---")
        prompts, gt_images, scenes = prepare_experiment_data()
        
        print("\n--- Step 2: Model Initialization ---")
        models = train_models()
        
        print("\n--- Step 3: Running Experiments ---")
        
        experiment_1(models, prompts, gt_images, save_dir)
        
        experiment_2(models, prompts, gt_images, save_dir)
        
        experiment_3(models, scenes, save_dir)
        
        print("\n===== All Experiments Completed Successfully =====")
        print(f"Results and plots saved to: {save_dir}")
        print("Generated PDF files:")
        for file in os.listdir(save_dir):
            if file.endswith('.pdf'):
                print(f"  - {file}")
        
        status_enum = "stopped"
        print(f"\nExperiment status: {status_enum}")
        
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
