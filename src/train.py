# -*- coding: utf-8 -*-
"""
Training utilities for EXACT: Executable-Aware Select-and-Compress Distillation
- Weighted CE on verified core tokens
- Unlikelihood loss on teacher-rejected tokens
- Curriculum-progressive compression (masking verified non-core steps across epochs)

This module exposes:
  - get_tokenizer_and_model
  - train_exact_language_model

All imports inside src use relative imports per project rules.
"""
from typing import Dict, List, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model  # optional
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Support running as a module or as a script
try:
    from .preprocess import ExactDataset, collate_exact  # type: ignore
except Exception:  # noqa: E722
    from preprocess import ExactDataset, collate_exact  # type: ignore


def get_tokenizer_and_model(model_name: str, device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


def _maybe_enable_lora(model: nn.Module, enable: bool = False) -> nn.Module:
    if not enable:
        return model
    if not PEFT_AVAILABLE:
        print("[LoRA] peft not available; skipping LoRA.")
        return model
    # Heuristic target module names
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]):
                short = name.split(".")[-1]
                target_modules.add(short)
    if not target_modules:
        target_modules = {"c_attn", "c_proj"}
    try:
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=list(target_modules))
        model = get_peft_model(model, cfg)
        print(f"[LoRA] Enabled for targets: {sorted(list(target_modules))}")
    except Exception as e:
        print(f"[LoRA] Failed to enable due to: {e}; continuing without LoRA.")
    return model


def _unlikelihood_loss(logits: torch.Tensor, target_ids: torch.Tensor, rej_mask: torch.Tensor) -> torch.Tensor:
    # Shift for causal LM
    logits = logits[:, :-1]
    target_ids = target_ids[:, 1:]
    rej_mask = rej_mask[:, 1:]
    if rej_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    p = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).exp().clamp(min=1e-6, max=1-1e-6)
    ul = -torch.log(1 - p)
    loss = (ul * rej_mask).sum() / (rej_mask.sum() + 1e-6)
    return loss


def _weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, core_mask: torch.Tensor, core_weight: float = 2.0) -> torch.Tensor:
    # Shifted
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    core_mask = core_mask[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    valid = (labels != -100).float()
    weights = torch.ones_like(valid)
    weights = torch.where(core_mask > 0, weights * core_weight, weights)
    nll = nll * valid
    denom = (weights * valid).sum() + 1e-6
    loss = (nll * weights).sum() / denom
    return loss


def train_exact_language_model(
    train_examples: List[Dict],
    tokenizer: AutoTokenizer,
    model_name: str = "sshleifer/tiny-gpt2",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 5e-5,
    core_weight: float = 2.0,
    lambda_neg: float = 0.5,
    curriculum_schedule: List[float] = None,
    use_lora: bool = False,
    device: str = "cpu",
) -> Tuple[AutoModelForCausalLM, Dict]:
    """Train a causal LM with EXACT objective on provided examples.
    Returns model and logs dict containing training losses per step.
    """
    if curriculum_schedule is None:
        curriculum_schedule = [0.0] * epochs
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model = _maybe_enable_lora(model.to(device), enable=use_lora)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses: List[float] = []
    for ep in range(epochs):
        ratio = curriculum_schedule[ep] if ep < len(curriculum_schedule) else curriculum_schedule[-1]
        ds = ExactDataset(train_examples, tokenizer, curriculum_ratio=ratio)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_exact(b, tokenizer.pad_token_id))
        model.train()
        ep_losses = []
        t0 = time.time()
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            core_mask = batch["core_mask"].to(device)
            rej_mask = batch["rej_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss_pos = _weighted_ce_loss(logits, labels, core_mask, core_weight=core_weight)
            loss_neg = _unlikelihood_loss(logits, labels, rej_mask)
            loss = loss_pos + lambda_neg * loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val = float(loss.detach().cpu().item())
            train_losses.append(val)
            ep_losses.append(val)
        dt = time.time() - t0
        print(f"[Train] Epoch {ep+1}/{epochs} hide={int(100*ratio)}% steps={len(dl)} avg_loss={np.mean(ep_losses):.4f} time={dt:.1f}s")
    return model, {"train_losses": train_losses}
