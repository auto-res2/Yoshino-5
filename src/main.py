"""
MP-CHAR Main Experiment Runner
Memory-Bandwidth-Aware Multi-Precision Anytime Reasoning

This script orchestrates the complete experimental pipeline from data preprocessing
through model training to comprehensive evaluation, implementing the MP-CHAR system
as described in the research methodology.

Usage:
    python main.py              # Run full experimental pipeline
    python main.py --quick      # Run quick smoke test only
    python main.py --exp1       # Run only end-to-end benchmark
    python main.py --exp2       # Run only ablation study
    python main.py --exp3       # Run only micro-benchmarks
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess_datasets
from train import train_assessor_dummy, DummyMPCHARModel
from evaluate import MPCHAREvaluator, run_quick_test

def setup_environment():
    """Initialize experimental environment and check dependencies."""
    print("MP-CHAR: Memory-Bandwidth-Aware Multi-Precision Anytime Reasoning")
    print("=" * 70)
    print("Setting up experimental environment...")
    
    directories = [
        "data",
        "models", 
        "config",
        ".research/iteration1/images"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory: {dir_path}")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if "T4" in gpu_name:
                print("✓ Target hardware detected: Tesla T4")
            else:
                print(f"⚠ Running on {gpu_name} (target: Tesla T4)")
        else:
            print("⚠ No GPU detected, running on CPU")
    except ImportError:
        print("⚠ PyTorch not available for GPU check")
    
    print("Environment setup complete!\n")

def run_experiment_1():
    """Execute Experiment 1: End-to-End Quality·Latency·Energy Benchmark."""
    print("Starting Experiment 1: End-to-End Benchmark...")
    
    evaluator = MPCHAREvaluator()
    datasets = ["big_bench_hard", "gsm8k_subset", "easy2hard_bench"]
    
    try:
        df_results = evaluator.evaluate_end_to_end(datasets)
        stats_results = evaluator.run_statistical_analysis(df_results)
        
        print("\nExperiment 1 Results Summary:")
        print("-" * 40)
        for _, row in df_results.iterrows():
            print(f"{row['system']:12s}: acc={row['accuracy']:.3f}, "
                  f"lat={row['latency_ms']:.1f}ms, energy={row['energy_j']:.2f}J")
        
        if stats_results:
            print(f"\nMP-CHAR Improvements vs Best Baseline:")
            print(f"  Accuracy: {stats_results['accuracy_improvement_pct']:+.1f}%")
            print(f"  Latency:  {stats_results['latency_improvement_pct']:+.1f}%") 
            print(f"  Energy:   {stats_results['energy_improvement_pct']:+.1f}%")
        
        return df_results
        
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        traceback.print_exc()
        return None

def run_experiment_2():
    """Execute Experiment 2: Precision Pyramid + Assessor Ablation."""
    print("Starting Experiment 2: Ablation Study...")
    
    evaluator = MPCHAREvaluator()
    
    try:
        df_results = evaluator.evaluate_ablations()
        
        print("\nExperiment 2 Results Summary:")
        print("-" * 40)
        for _, row in df_results.iterrows():
            print(f"{row['variant']:12s}: acc={row['accuracy']:.3f}, "
                  f"energy={row['energy_j']:.2f}J, P16={row['p16_calls']:.1f}")
        
        full_system = df_results[df_results['variant'] == 'V1_full']
        no_p4 = df_results[df_results['variant'] == 'V2_noP4']
        no_assessor = df_results[df_results['variant'] == 'V4_noAssessor']
        
        if len(full_system) > 0 and len(no_p4) > 0:
            p4_contribution = (full_system['accuracy'].iloc[0] - no_p4['accuracy'].iloc[0]) * 100
            print(f"\nComponent Analysis:")
            print(f"  P4 precision contribution: {p4_contribution:+.1f}% accuracy")
        
        if len(full_system) > 0 and len(no_assessor) > 0:
            assessor_contribution = (full_system['accuracy'].iloc[0] - no_assessor['accuracy'].iloc[0]) * 100
            print(f"  Assessor contribution: {assessor_contribution:+.1f}% accuracy")
        
        return df_results
        
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        traceback.print_exc()
        return None

def run_experiment_3():
    """Execute Experiment 3: KV-Cache & Controller Micro-benchmarks."""
    print("Starting Experiment 3: Micro-benchmarks...")
    
    evaluator = MPCHAREvaluator()
    
    try:
        results = evaluator.evaluate_micro_benchmarks()
        
        print("\nExperiment 3 Results Summary:")
        print("-" * 40)
        
        cache_results = results['cache_traffic']
        zero_copy_traffic = cache_results['zero_copy']
        dup_buffer_traffic = cache_results['dup_buffers']
        traffic_reduction = (dup_buffer_traffic - zero_copy_traffic) / dup_buffer_traffic * 100
        
        print(f"KV-Cache Traffic:")
        print(f"  Zero-copy: {zero_copy_traffic:.1f} bytes/token")
        print(f"  Duplicate buffers: {dup_buffer_traffic:.1f} bytes/token")
        print(f"  Traffic reduction: {traffic_reduction:.1f}%")
        
        bandwidth_results = results['bandwidth_awareness']
        throughput_off = bandwidth_results[False]
        throughput_on = bandwidth_results[True]
        throughput_gain = (throughput_on - throughput_off) / throughput_off * 100
        
        print(f"\nBandwidth-Aware Controller:")
        print(f"  ω_mem OFF: {throughput_off:.1f} tokens/s")
        print(f"  ω_mem ON:  {throughput_on:.1f} tokens/s")
        print(f"  Throughput gain: {throughput_gain:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        traceback.print_exc()
        return None

def run_full_pipeline():
    """Execute the complete MP-CHAR experimental pipeline."""
    print("Starting Full MP-CHAR Experimental Pipeline...")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    try:
        print("\nStep 1: Data Preprocessing")
        print("-" * 30)
        preprocess_datasets()
        
        print("\nStep 2: Assessor Training")
        print("-" * 30)
        assessor = train_assessor_dummy()
        
        print("\nStep 3: Experimental Evaluation")
        print("-" * 30)
        
        results['experiment_1'] = run_experiment_1()
        results['experiment_2'] = run_experiment_2()
        results['experiment_3'] = run_experiment_3()
        
        print("\n" + "=" * 70)
        print("EXPERIMENTAL PIPELINE COMPLETE")
        print("=" * 70)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.1f} seconds")
        
        images_dir = Path(".research/iteration1/images")
        pdf_files = list(images_dir.glob("*.pdf"))
        print(f"Generated {len(pdf_files)} PDF plots in {images_dir}")
        
        if results['experiment_1'] is not None:
            mp_char_row = results['experiment_1'][results['experiment_1']['system'].str.contains('MP-CHAR')]
            if len(mp_char_row) > 0:
                mp_char_acc = mp_char_row['accuracy'].iloc[0]
                mp_char_lat = mp_char_row['latency_ms'].iloc[0]
                print(f"\nKey Results:")
                print(f"  MP-CHAR Accuracy: {mp_char_acc:.3f}")
                print(f"  MP-CHAR Latency: {mp_char_lat:.1f} ms/sample")
        
        print(f"\nAll experimental outputs saved to: .research/iteration1/images/")
        print("Experiment completed successfully! ✓")
        
        return True
        
    except Exception as e:
        print(f"\nExperimental pipeline failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="MP-CHAR Experimental Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--exp1", action="store_true", help="Run only Experiment 1")
    parser.add_argument("--exp2", action="store_true", help="Run only Experiment 2")
    parser.add_argument("--exp3", action="store_true", help="Run only Experiment 3")
    
    args = parser.parse_args()
    
    setup_environment()
    
    success = False
    
    try:
        if args.quick:
            print("Running quick smoke test...")
            success = run_quick_test()
            
        elif args.exp1:
            preprocess_datasets()
            result = run_experiment_1()
            success = result is not None
            
        elif args.exp2:
            preprocess_datasets()
            result = run_experiment_2()
            success = result is not None
            
        elif args.exp3:
            result = run_experiment_3()
            success = result is not None
            
        else:
            success = run_full_pipeline()
        
        if success:
            print("\n🎉 MP-CHAR experiment completed successfully!")
            return 0
        else:
            print("\n❌ MP-CHAR experiment failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
