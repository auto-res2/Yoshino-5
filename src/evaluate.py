"""
HAPIQ Evaluation Module
----------------------
Modular evaluation component for HAPIQ experiments.
This module provides evaluation functionality for trained models.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from pathlib import Path
import json
from typing import Optional

def load_model_for_evaluation(model_path: str, quantization_config: Optional[str] = None):
    """Load a trained model for evaluation."""
    load_kwargs = {}
    if quantization_config:
        config_path = Path(__file__).parent.parent / "config" / quantization_config
        if config_path.exists():
            load_kwargs["quantization_config"] = str(config_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_perplexity(model, tokenizer, texts: list, max_length: int = 512):
    """Evaluate model perplexity on given texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=max_length).to(model.device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def evaluate_generation_quality(model, tokenizer, prompts: list, max_new_tokens: int = 50):
    """Evaluate generation quality and speed."""
    model.eval()
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.perf_counter()
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = end_time - start_time
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second
        })
    
    return results

def evaluate_memory_usage(model):
    """Evaluate model memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randint(0, 1000, (1, 128)).to(model.device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1e6    # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1e6   # MB
        
        return {
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "peak_memory_mb": peak_memory
        }
    else:
        return {"memory_allocated_mb": 0, "memory_reserved_mb": 0, "peak_memory_mb": 0}

def comprehensive_evaluation(model_path: str, quantization_config: Optional[str] = None, 
                           save_results: bool = True):
    """Run comprehensive evaluation of a model."""
    print(f"[EVAL] Loading model from {model_path}")
    
    model, tokenizer = load_model_for_evaluation(model_path, quantization_config)
    
    if torch.cuda.is_available():
        model.to("cuda")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning has revolutionized artificial intelligence research.",
        "Quantization techniques reduce model size while preserving accuracy.",
        "Large language models require efficient inference methods.",
        "HAPIQ provides hierarchical adaptive precision for neural networks."
    ]
    
    test_prompts = [
        "The future of AI is",
        "Machine learning can help",
        "The most important aspect of"
    ]
    
    print("[EVAL] Evaluating perplexity...")
    perplexity = evaluate_perplexity(model, tokenizer, test_texts)
    
    print("[EVAL] Evaluating generation quality...")
    generation_results = evaluate_generation_quality(model, tokenizer, test_prompts)
    
    print("[EVAL] Evaluating memory usage...")
    memory_stats = evaluate_memory_usage(model)
    
    avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in generation_results])
    avg_generation_time = np.mean([r["generation_time"] for r in generation_results])
    
    evaluation_results = {
        "model_path": model_path,
        "quantization_config": quantization_config,
        "perplexity": perplexity,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_generation_time": avg_generation_time,
        "memory_stats": memory_stats,
        "generation_examples": generation_results[:2]  # Save first 2 examples
    }
    
    print(f"[EVAL] Results:")
    print(f"  Perplexity: {perplexity:.3f}")
    print(f"  Avg tokens/sec: {avg_tokens_per_second:.2f}")
    print(f"  Peak memory: {memory_stats['peak_memory_mb']:.1f} MB")
    
    if save_results:
        results_path = Path(__file__).parent.parent / ".research" / "iteration1" / "images" / "evaluation_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"[EVAL] Results saved to {results_path}")
    
    return evaluation_results

if __name__ == "__main__":
    model_path = "sshleifer/tiny-gpt2"  # Use small model for testing
    comprehensive_evaluation(model_path, quantization_config="hapiq.json")
