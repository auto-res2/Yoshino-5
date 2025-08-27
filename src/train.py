"""
HAPIQ Training Module
--------------------
Modular training component for HAPIQ experiments.
This module provides training functionality that can be used by main.py
or run independently for model training tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
from pathlib import Path
from typing import Optional

def setup_model_and_tokenizer(model_name: str, quantization_config: Optional[str] = None):
    """Setup model and tokenizer with optional quantization."""
    load_kwargs = {}
    if quantization_config:
        config_path = Path(__file__).parent.parent / "config" / quantization_config
        if config_path.exists():
            load_kwargs["quantization_config"] = str(config_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train_model(model_name: str, dataset_name: Optional[str] = None, output_dir: str = "models/", 
                quantization_config: Optional[str] = None, max_steps: int = 100):
    """Train a model with HAPIQ quantization."""
    print(f"[TRAIN] Setting up model: {model_name}")
    
    model, tokenizer = setup_model_and_tokenizer(model_name, quantization_config)
    
    if dataset_name:
        try:
            dataset = load_dataset(dataset_name, split="train[:1000]")
        except:
            print(f"[WARNING] Could not load {dataset_name}, using synthetic data")
            dataset = None
    else:
        dataset = None
    
    if dataset is None:
        synthetic_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "HAPIQ provides hierarchical adaptive precision for LLMs.",
            "Quantization reduces memory usage while maintaining performance.",
            "Deep learning models require efficient inference methods."
        ] * 20
        
        class SyntheticDataset:
            def __init__(self, texts, tokenizer):
                self.texts = texts
                self.tokenizer = tokenizer
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(text, truncation=True, padding="max_length", 
                                        max_length=128, return_tensors="pt")
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze()
                }
        
        dataset = SyntheticDataset(synthetic_texts, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print(f"[TRAIN] Starting training for {max_steps} steps...")
    trainer.train()
    
    trainer.save_model()
    print(f"[TRAIN] Model saved to {output_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    model_name = "sshleifer/tiny-gpt2"  # Use small model for testing
    train_model(model_name, quantization_config="hapiq.json", max_steps=50)
