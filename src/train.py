"""train.py – model loading (LoRA) and task-level training utilities for AGSC
==========================================================================
The code is extracted from the original monolithic script without any logical
changes, only minimal clean-ups and additional comments / guards for safer
execution on various environments.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

logger = logging.getLogger("agsc.train")

# ---------------------------------------------------------------------------
# Safe AdamW -----------------------------------------------------------------
#   (accepts learning rate passed as string from YAML grid-search parameters)
# ---------------------------------------------------------------------------
from torch.optim import AdamW as _AdamWOrig  # noqa: E402  (import after torch)


class _AdamWSafe(_AdamWOrig):
    """AdamW that gracefully converts string lr values to float."""

    def __init__(self, params, lr: float | str = 1e-3, *a, **kw):
        if isinstance(lr, str):
            lr = float(lr)
        super().__init__(params, lr=lr, *a, **kw)


# Monkey-patch so that every subsequent import in other modules sees it.
import torch.optim.adamw as _m

_m.AdamW = _AdamWSafe  # type: ignore[attr-defined]
torch.optim.AdamW = _AdamWSafe  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model helpers --------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_llama_lora(hf_id: str, r: int, alpha: int, *, bnb_8bit: bool = True):
    """Load a LLaMA/OPT/BLOOM-style causal LM with a LoRA adapter on top."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_cfg = None
    if bnb_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        quantization_config=bnb_cfg,
    )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg).to(device)

    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=False)
    tok.pad_token_id = tok.eos_token_id  # avoid warning with causal LM
    return model, tok


# ---------------------------------------------------------------------------
# Training loop --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    *,
    mixed: bool = True,
) -> Tuple[float, float]:
    """Train *model* for one epoch, return (mean loss, epoch seconds)."""

    model.train()
    t0 = time.time()
    total = 0.0
    step = -1
    for step, batch in enumerate(dl):
        for k in batch:
            batch[k] = batch[k].to(model.device)
        with autocast(enabled=mixed):
            out = model(**batch)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        total += float(loss.item())

    return total / (step + 1), time.time() - t0


# ---------------------------------------------------------------------------
# Gradient-conflict probe ----------------------------------------------------
# ---------------------------------------------------------------------------

def _grab_grad(model: torch.nn.Module, buf: Iterator[Dict[str, torch.Tensor]]):
    model.zero_grad(set_to_none=True)
    batch = next(buf)
    for k in batch:
        batch[k] = batch[k].to(model.device)
    out = model(**batch)
    out.loss.backward()
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads)


def probe_gradient_conflict(
    model: torch.nn.Module,
    buf_a: Iterator[Dict[str, torch.Tensor]],
    buf_b: Iterator[Dict[str, torch.Tensor]],
) -> float:
    """Return cosine similarity between mean gradients of two mini-batches."""

    g1 = _grab_grad(model, buf_a)
    g2 = _grab_grad(model, buf_b)
    return float(torch.nn.functional.cosine_similarity(g1, g2, dim=0).item())
