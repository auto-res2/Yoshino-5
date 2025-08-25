import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import DummyDiffusionModel, NoAIBDiffusionModel
from evaluate import experiment_quality_vs_memory, experiment_edge_inference, experiment_ablation
from preprocess import test_experiments

sns.set(style='whitegrid')

def main():
    print('Starting AIB-2Q-Diff Experiments...\n')
    
    results_quality = experiment_quality_vs_memory()
    print()
    results_edge = experiment_edge_inference()
    print()
    results_ablation = experiment_ablation()
    print('\nAll experiments completed.')
    
    print('\n----- Summary of Results -----')
    print('Experiment 1 (Quality vs Memory):')
    for model in results_quality.keys():
        print(f"  {model:8s}: Time = {results_quality[model]['time']:.2f}s, FID = {results_quality[model]['FID']:.2f}")

    print('\nExperiment 2 (Edge Inference):')
    print(f"  Avg Inference Time = {results_edge['avg_time']:.2f}s, Avg Memory = {results_edge['avg_memory']:.2f}bit")

    print('\nExperiment 3 (Ablation Study):')
    for model in results_ablation.keys():
        print(f"  {model:6s}: Time = {results_ablation[model]['time']:.2f}s, FID = {results_ablation[model]['FID']:.2f}")

if __name__ == '__main__':
    print('--- Running Quick Test ---')
    test_experiments()
    print('\n--- Starting Full Experiments ---')
    main()
