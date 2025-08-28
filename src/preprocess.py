import os
import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------ General utilities ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_dir() -> str:
    # All figures must be saved here
    return os.path.join('.research', 'iteration3', 'images')


# ------------------------------ Data prep ------------------------------

def default_warmup_texts() -> List[str]:
    return [
        "The capital of France is",  # knowledge
        "Solve: 17 + 29 =",          # arithmetic
        "Explain why the sky is blue.",  # reasoning
        "Translate: Hello -> Spanish:",  # translation
        "In biology, mitochondria are",  # science
        "Newton's second law states that",  # physics
        "Python: Write a function to reverse a list.",  # code-like
        "Q: What is the capital of Italy? A:",
    ] * 2  # keep small by default


def build_warmup_loader(tokenizer, texts: List[str], max_len: int = 96, batch_size: int = 1):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    ds = [{'input_ids': enc['input_ids'][i], 'attention_mask': enc['attention_mask'][i]} for i in range(enc['input_ids'].size(0))]
    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)


# ------------------------------ Model loading ------------------------------

def load_model_and_tokenizer(model_name: str, device: str = 'cuda') -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    Load HF model and tokenizer in a device-friendly manner.
    - Uses float16 on CUDA, float32 on CPU for broad compatibility.
    - Returns (model, tokenizer, chosen_device)
    """
    use_cuda = torch.cuda.is_available() and (device == 'cuda')
    dtype = torch.float16 if use_cuda else torch.float32
    chosen_device = 'cuda' if use_cuda else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(chosen_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, chosen_device
