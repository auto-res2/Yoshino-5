"""
HAPIQ Data Preprocessing Module
------------------------------
Modular preprocessing component for HAPIQ experiments.
This module handles data loading, cleaning, and preparation.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

def load_and_clean_dataset(dataset_name: str, split: str = "train", max_samples: Optional[int] = None):
    """Load and clean a dataset for training/evaluation."""
    print(f"[PREPROCESS] Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        if max_samples:
            dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
        
        print(f"[PREPROCESS] Loaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"[WARNING] Failed to load {dataset_name}: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    
    text = text.strip()
    
    return text

def tokenize_dataset(dataset, tokenizer, max_length: int = 512, text_column: str = "text"):
    """Tokenize a dataset for training."""
    def tokenize_function(examples):
        if text_column in examples:
            texts = examples[text_column]
        elif "question" in examples:
            texts = examples["question"]
        elif "ctx_a" in examples and "ctx_b" in examples:
            texts = [a + " " + b for a, b in zip(examples["ctx_a"], examples["ctx_b"])]
        else:
            for key, values in examples.items():
                if isinstance(values[0], str):
                    texts = values
                    break
            else:
                texts = [""] * len(list(examples.values())[0])
        
        texts = [clean_text(text) for text in texts]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def create_synthetic_dataset(size: int = 1000, tokenizer=None) -> Dataset:
    """Create a synthetic dataset for testing."""
    print(f"[PREPROCESS] Creating synthetic dataset with {size} samples")
    
    templates = [
        "The {adjective} {noun} {verb} {adverb} in the {location}.",
        "Machine learning {verb} {adjective} {noun} for {application}.",
        "HAPIQ provides {adjective} precision for {noun} {application}.",
        "Quantization {verb} {noun} while maintaining {adjective} performance.",
        "Deep learning models require {adjective} {noun} methods.",
        "The research shows that {adjective} {noun} {verb} significantly.",
        "In artificial intelligence, {noun} {verb} {adverb} with {adjective} results.",
        "Neural networks use {adjective} {noun} to {verb} complex patterns."
    ]
    
    adjectives = ["efficient", "advanced", "novel", "robust", "scalable", "adaptive", "hierarchical", "optimal"]
    nouns = ["algorithms", "methods", "techniques", "models", "systems", "approaches", "frameworks", "solutions"]
    verbs = ["optimize", "enhance", "improve", "accelerate", "streamline", "transform", "revolutionize", "advance"]
    adverbs = ["efficiently", "effectively", "significantly", "dramatically", "substantially", "remarkably"]
    locations = ["datacenter", "cloud", "edge device", "mobile platform", "server", "workstation"]
    applications = ["inference", "training", "deployment", "optimization", "acceleration", "compression"]
    
    texts = []
    for i in range(size):
        template = np.random.choice(templates)
        text = template.format(
            adjective=np.random.choice(adjectives),
            noun=np.random.choice(nouns),
            verb=np.random.choice(verbs),
            adverb=np.random.choice(adverbs),
            location=np.random.choice(locations),
            application=np.random.choice(applications)
        )
        texts.append(text)
    
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset

def prepare_evaluation_datasets():
    """Prepare standard evaluation datasets."""
    eval_datasets = {}
    
    dataset_configs = [
        ("wikitext", "wikitext-2-raw-v1", "test"),
        ("lambada", None, "test"),
        ("piqa", None, "validation"),
    ]
    
    for name, config, split in dataset_configs:
        try:
            if config:
                dataset = load_dataset(name, config, split=split)
            else:
                dataset = load_dataset(name, split=split)
            
            if len(dataset) > 1000:
                dataset = dataset.shuffle(seed=42).select(range(1000))
            
            eval_datasets[name] = dataset
            print(f"[PREPROCESS] Loaded {name}: {len(dataset)} samples")
            
        except Exception as e:
            print(f"[WARNING] Could not load {name}: {e}")
    
    if not eval_datasets:
        print("[PREPROCESS] Creating synthetic evaluation dataset")
        eval_datasets["synthetic"] = create_synthetic_dataset(500)
    
    return eval_datasets

def preprocess_for_hapiq(dataset_name: Optional[str] = None, tokenizer_name: str = "sshleifer/tiny-gpt2", 
                        max_samples: int = 1000, max_length: int = 512):
    """Main preprocessing function for HAPIQ experiments."""
    print("[PREPROCESS] Starting HAPIQ preprocessing pipeline")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if dataset_name:
        dataset = load_and_clean_dataset(dataset_name, max_samples=max_samples)
        if dataset is None:
            print("[PREPROCESS] Falling back to synthetic dataset")
            dataset = create_synthetic_dataset(max_samples, tokenizer)
    else:
        dataset = create_synthetic_dataset(max_samples, tokenizer)
    
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "preprocessed_data.json"
    
    processed_data = {
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "num_samples": len(tokenized_dataset),
        "sample_data": [
            {
                "input_ids": tokenized_dataset[i]["input_ids"].tolist(),
                "attention_mask": tokenized_dataset[i]["attention_mask"].tolist(),
                "labels": tokenized_dataset[i]["labels"].tolist()
            }
            for i in range(min(10, len(tokenized_dataset)))  # Save first 10 samples as examples
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"[PREPROCESS] Preprocessed data saved to {output_path}")
    print(f"[PREPROCESS] Dataset size: {len(tokenized_dataset)} samples")
    print(f"[PREPROCESS] Max sequence length: {max_length}")
    
    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    dataset, tokenizer = preprocess_for_hapiq(
        dataset_name=None,  # Use synthetic data
        max_samples=500,
        max_length=256
    )
    
    print(f"[PREPROCESS] Preprocessing complete. Dataset ready with {len(dataset)} samples.")
