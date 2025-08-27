import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

try:
    from fvcore.nn import FlopCountAnalysis
    _has_fvcore = True
except ImportError:
    _has_fvcore = False

def count_flops(module: nn.Module, *args):
    if _has_fvcore:
        return FlopCountAnalysis(module, args).total()
    return sum(p.numel() for p in module.parameters() if p.requires_grad) * 2

def experiment1(model: nn.Module, loader, device: str = "cpu"):
    print("===== EXPERIMENT 1: ACCURACY × FLOPS BENCHMARK =====")
    model.to(device)
    model.eval()
    
    acc_s = tot_s = acc_d = tot_d = 0
    flop_s = flop_d = 0
    
    for vid, q, ans, is_static in tqdm(loader, desc="Exp-1 Eval"):
        vid, q, ans = vid.to(device), q.to(device), ans.to(device)
        with torch.no_grad():
            pred, rank, T = model(vid, q)
        
        correct = (pred == ans).sum().item()
        flops_est = model.estimate_flops(rank, T)
        
        for i, sta in enumerate(is_static):
            if sta:
                acc_s += correct / len(ans)
                flop_s += flops_est
                tot_s += 1
            else:
                acc_d += correct / len(ans)
                flop_d += flops_est
                tot_d += 1
    
    print(f"STATIC   – Acc {acc_s/tot_s*100:.1f}% | FLOPs {flop_s/tot_s/1e6:.1f}M")
    print(f"DYNAMIC  – Acc {acc_d/tot_d*100:.1f}% | FLOPs {flop_d/tot_d/1e6:.1f}M")
    
    return {
        'static_acc': acc_s/tot_s,
        'dynamic_acc': acc_d/tot_d,
        'static_flops': flop_s/tot_s,
        'dynamic_flops': flop_d/tot_d
    }

def experiment2(output_dir: str = ".research/iteration1/images"):
    print("===== EXPERIMENT 2: COMPONENT ABLATION =====")
    from train import HyperSTConfig, HyperSTModel
    from preprocess import create_dataloader
    
    variants = {
        'Full': HyperSTConfig(True, True, True, True),
        '-SG': HyperSTConfig(False, True, True, True),
        '-HR': HyperSTConfig(True, False, True, True),
        '-TP': HyperSTConfig(True, True, False, True),
        '-DoRA': HyperSTConfig(True, True, True, False)
    }
    
    results = {}
    loader = create_dataloader(120, 16)
    
    for name, cfg in variants.items():
        model = HyperSTModel(cfg)
        acc = flops = 0
        
        for vid, q, ans, _ in loader:
            pred, rank, T = model(vid, q)
            acc += (pred == ans).float().mean().item()
            flops += model.estimate_flops(rank, T)
        
        acc /= len(loader)
        flops /= len(loader)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
        
        results[name] = (acc, flops, trainable)
        print(f"{name:5s} | Acc {acc*100:4.1f}% | FLOPs {flops/1e6:6.1f}M | trainable {trainable:.1f}k")
    
    names, accs, fls, _ = zip(*[(k, *v) for k, v in results.items()])
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(names), y=list(accs))
    plt.ylabel("Accuracy")
    plt.title("Component Ablation Study")
    plt.xticks(rotation=45)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/accuracy_ablation.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    
    return results

def experiment3(output_dir: str = ".research/iteration1/images"):
    print("===== EXPERIMENT 3: BUDGET-ADAPTIVE CURVE =====")
    from train import HyperSTConfig, HyperSTModel
    from preprocess import create_dataloader
    
    budgets = [20e6, 40e6, 60e6, 80e6, 100e6]
    model = HyperSTModel(HyperSTConfig())
    loader = create_dataloader(160, 16, static_prob=0.3)
    
    curve = {}
    for B in budgets:
        acc_cnt = tot = 0
        for vid, q, ans, _ in loader:
            pred, _, _ = model(vid, q, budget=B)
            acc_cnt += (pred == ans).sum().item()
            tot += len(ans)
        
        curve[B] = acc_cnt / tot
        print(f"Budget {B/1e6:3.0f}M – Acc {curve[B]*100:5.1f}%")
    
    plt.figure(figsize=(8, 6))
    plt.plot([b/1e6 for b in budgets], [curve[b] for b in budgets], 
             marker='o', label='HyperST-LoRA', linewidth=2, markersize=8)
    plt.xlabel('Budget (MFLOPs)')
    plt.ylabel('Accuracy')
    plt.title('Budget-Adaptive Degradation Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/accuracy_budget_curve.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    
    return curve
