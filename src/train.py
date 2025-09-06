"""
train.py – model construction and single-task training utilities
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# -----------------------------------------------------------------------------
# Monkey-patch -----------------------------------------------------------------
# -----------------------------------------------------------------------------
# A lot of YAML/CLI configs specify the learning-rate as a *string* (e.g. "5e-5").
# The stock `torch.optim.AdamW` expects a float and will crash otherwise.  We
# therefore wrap AdamW so it silently converts string LRs to floats.  To be extra
# safe we replace *both* torch.optim.AdamW **and** torch.optim.adamw.AdamW so
# that whatever import style a downstream module uses it will pick up the safe
# version.

from torch.optim import AdamW as _AdamWOrig  # noqa: E402 – import after torch


class _AdamWSafe(_AdamWOrig):  # type: ignore[misc]
    """Drop-in replacement for AdamW with string-to-float LR conversion."""

    def __init__(self, params, lr=1e-3, *args, **kwargs):  # noqa: ANN001
        # If the LR comes in as a string ("5e-5", "0.0001", …) convert it.
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError as exc:  # pragma: no cover – defensive
                raise ValueError(f"AdamW received an invalid lr string: {lr!r}") from exc
        super().__init__(params, lr=float(lr), *args, **kwargs)


# Expose original impl. just in case and patch *both* attribute locations.
#   1) torch.optim.AdamW                     (most common)
#   2) torch.optim.adamw.AdamW              (imported via `from … import AdamW`)

torch.optim._AdamWOrig = _AdamWOrig  # type: ignore[attr-defined]

torch.optim.AdamW = _AdamWSafe  # type: ignore[assignment]
import torch.optim.adamw as _adamw_mod  # noqa: E402  – after patch
_adamw_mod.AdamW = _AdamWSafe  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Logger ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
logger = logging.getLogger("agsc.train")

# -----------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------


def _safe_device() -> torch.device:
    """Return CUDA device if available otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_llama_lora(model_id: str, r: int, alpha: int, *, bnb_8bit: bool = True):
    """Load a causal-LM together with a LoRA adapter.

    Args:
        model_id:    HuggingFace identifier.
        r:           LoRA rank.
        alpha:       LoRA alpha.
        bnb_8bit:    If true load model with 8-bit quantisation (bitsandbytes).

    Returns:
        model (torch.nn.Module) and corresponding tokenizer.
    """
    device = _safe_device()

    bnb_cfg = None
    if bnb_8bit:
        try:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        except Exception as exc:  # pragma: no cover – CPU fallback
            logger.warning("bitsandbytes not available – falling back to fp16: %s", exc)
            bnb_cfg = None

    logger.info("Loading base model %s …", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        quantization_config=bnb_cfg,
    )

    logger.info("Applying LoRA (r=%d, alpha=%d).", r, alpha)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# -----------------------------------------------------------------------------
# Single-task training loop ----------------------------------------------------
# -----------------------------------------------------------------------------


def train_one_task(
    model: torch.nn.Module,
    tokenizer,  # transformers tokenizer – kept for future use
    dataloader,  # PyTorch DataLoader yielding *tokenised* batches
    optim: torch.optim.Optimizer,
    cfg: Dict[str, Any],
):
    """Minimal training loop for one continual-learning task."""

    scaler = GradScaler(enabled=cfg.get("mixed_precision", True) and torch.cuda.is_available())
    grad_acc_steps = int(cfg.get("grad_acc_steps", 1))

    model.train()
    total_loss: float = 0.0
    step: int = 0

    for batch in dataloader:
        step += 1
        optim.zero_grad()
        with autocast(enabled=scaler.is_enabled()):
            out = model(**batch)
            loss = out.loss / grad_acc_steps
        scaler.scale(loss).backward()
        if step % grad_acc_steps == 0:
            scaler.step(optim)
            scaler.update()
        total_loss += loss.item()

    return total_loss / max(step, 1)
