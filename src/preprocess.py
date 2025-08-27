"""
MP-CHAR Data Preprocessing Module
Generates synthetic datasets with multiple difficulty patterns for testing
the Memory-Bandwidth-Aware Multi-Precision Anytime Reasoning system.
"""
import random
import json
from typing import List, Dict, Any
from pathlib import Path

class ToySample(dict):
    """Minimal container for synthetic samples with .in/.out fields."""
    pass

def generate_synthetic_dataset(name: str, n: int, seed: int = 42) -> List[ToySample]:
    """
    Creates synthetic datasets with three difficulty bands (easy/medium/hard)
    to demonstrate different MP-CHAR promote/prune behaviors.
    
    Args:
        name: Dataset identifier
        n: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of ToySample objects with 'in', 'out', and 'difficulty' fields
    """
    random.seed(seed)
    samples: List[ToySample] = []
    
    for i in range(n):
        difficulty = random.choices(
            ["easy", "medium", "hard"], 
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        prompt_len = {"easy": 20, "medium": 60, "hard": 120}[difficulty]
        
        answer = {
            "easy": "42",
            "medium": "Hello, world!",
            "hard": "The integral equals 2π/3."
        }[difficulty]
        
        prompt = "? " * prompt_len
        
        samples.append(ToySample({
            "in": prompt,
            "out": answer,
            "difficulty": difficulty,
            "sample_id": f"{name}_{i:04d}"
        }))
    
    return samples

def save_dataset(samples: List[ToySample], filepath: str) -> None:
    """Save dataset to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(samples, f, indent=2)

def load_dataset(filepath: str) -> List[ToySample]:
    """Load dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [ToySample(item) for item in data]

def preprocess_datasets():
    """Generate and save all experimental datasets."""
    print("Generating synthetic datasets for MP-CHAR experiments...")
    
    datasets = {
        "big_bench_hard": 60,    # End-to-end benchmark
        "gsm8k_subset": 40,      # Ablation studies  
        "easy2hard_bench": 30,   # Micro-benchmarks
        "test_quick": 10         # Quick smoke test
    }
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for name, size in datasets.items():
        samples = generate_synthetic_dataset(name, size)
        filepath = data_dir / f"{name}.json"
        save_dataset(samples, str(filepath))
        print(f"Generated {name}: {size} samples -> {filepath}")
    
    print("Dataset preprocessing complete!")

if __name__ == "__main__":
    preprocess_datasets()
