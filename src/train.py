import os
import math
import json
import time
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Matplotlib for training curves (saved as vector PDF)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------------
# Synthetic Data
# -------------------------------

class SyntheticSeqDataset(Dataset):
    """
    Generates synthetic token sequences for three patterns:
    - pattern 'copy': next token repeats previous one with small noise.
    - pattern 'sum': next token is sum of two earlier tokens modulo vocab.
    - pattern 'brackets': frequent punctuation-style tokens, with occasional bracket pairs.

    These patterns are mixed to produce robustness and reasoning-like spikes.
    """
    def __init__(self, n_samples=256, seq_len=64, vocab_size=128, mix=True, pattern=None, seed=123):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.samples = []
        punct_ids = list(range(vocab_size-8, vocab_size))  # last 8 tokens as punctuation
        for _ in range(n_samples):
            if mix:
                pat = rng.choice(["copy", "sum", "brackets"])  # mix all patterns
            else:
                pat = pattern if pattern is not None else "copy"
            x = rng.integers(low=0, high=vocab_size-8, size=seq_len, dtype=np.int64)
            if pat == "copy":
                for t in range(1, seq_len):
                    if rng.random() < 0.8:
                        x[t] = x[t-1]
                    else:
                        x[t] = int((x[t-1] + rng.integers(1, 7)) % (vocab_size-8))
            elif pat == "sum":
                for t in range(2, seq_len):
                    if rng.random() < 0.5:
                        x[t] = int((x[t-1] + x[t-2]) % (vocab_size-8))
            elif pat == "brackets":
                for t in range(1, seq_len):
                    if t % 7 == 0:
                        x[t] = rng.choice(punct_ids)
            # stash
            self.samples.append(torch.tensor(x, dtype=torch.long))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ids = self.samples[idx]
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


def collate_batch(batch):
    ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attn = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return {"input_ids": ids, "attention_mask": attn}


# -------------------------------
# Quantization helpers
# -------------------------------

@dataclass
class QParams:
    scale: torch.Tensor
    zero: torch.Tensor


def affine_quant_per_row(x: torch.Tensor, n_bits: int = 8, symmetric: bool = False) -> Tuple[torch.Tensor, QParams]:
    """
    Per-row affine quantization of 2D weight matrices.
    x: [out_features, in_features]
    Returns q_int (int tensor) and (scale, zero)
    """
    assert x.dim() == 2
    x_fp32 = x.float()
    if symmetric:
        qmin, qmax = -(2**(n_bits-1)), 2**(n_bits-1) - 1
        max_abs = x_fp32.abs().amax(dim=1, keepdim=True) + 1e-8
        scale = (max_abs / qmax).clamp(min=1e-8)
        zero = torch.zeros_like(scale)
        q = torch.clamp((x_fp32 / scale).round(), qmin, qmax).to(torch.int8 if n_bits==8 else torch.int16)
    else:
        qmin, qmax = 0, 2**n_bits - 1
        x_min = x_fp32.amin(dim=1, keepdim=True)
        x_max = x_fp32.amax(dim=1, keepdim=True)
        scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp(min=1e-8)
        zero = (qmin - x_min/scale).round()
        q = torch.clamp((x_fp32/scale + zero).round(), qmin, qmax)
        q = q.to(torch.uint8)
    return q, QParams(scale=scale, zero=zero)


def affine_dequant_per_row(q: torch.Tensor, qp: QParams, symmetric: bool = False) -> torch.Tensor:
    if symmetric:
        return (q.float() * qp.scale)
    else:
        return (q.float() - qp.zero) * qp.scale


def quantize_4bit_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, QParams]:
    # Unsigned [0, 15] per-row affine
    q, qp = affine_quant_per_row(x, n_bits=4, symmetric=False)
    return q, qp


def quantize_8bit_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, QParams]:
    # Signed int8 symmetric
    q, qp = affine_quant_per_row(x, n_bits=8, symmetric=True)
    return q, qp


# Nibble pack/unpack for 4-bit unsigned representations

def pack_nibbles(x_u8: torch.Tensor) -> torch.Tensor:
    assert x_u8.dtype == torch.uint8
    if x_u8.shape[-1] % 2 == 1:
        x_u8 = F.pad(x_u8, (0,1))
    even = x_u8[..., 0::2]
    odd = x_u8[..., 1::2]
    packed = (odd << 4) | (even & 0x0F)
    return packed.contiguous()


def unpack_nibbles(packed: torch.Tensor, out_last_dim: int) -> torch.Tensor:
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    out = torch.stack([even, odd], dim=-1).flatten(-2)
    return out[..., :out_last_dim].contiguous()


# 2-bit header packing for token-wise s_t in {0,1,2,3}

def pack_s_headers(s_tokens: torch.Tensor) -> torch.Tensor:
    # packs per 4 tokens into 1 byte (2 bits per token)
    assert s_tokens.dim() == 1
    T = s_tokens.shape[0]
    pad = (4 - (T % 4)) % 4
    x = F.pad(s_tokens.to(torch.uint8), (0, pad))
    x = x.view(-1, 4)
    b = (x[:,0] & 0x03) | ((x[:,1] & 0x03) << 2) | ((x[:,2] & 0x03) << 4) | ((x[:,3] & 0x03) << 6)
    return b.contiguous()


def unpack_s_headers(bytes_u8: torch.Tensor, T: int) -> torch.Tensor:
    bits = torch.stack([
        bytes_u8 & 0x03,
        (bytes_u8 >> 2) & 0x03,
        (bytes_u8 >> 4) & 0x03,
        (bytes_u8 >> 6) & 0x03,
    ], dim=-1).flatten()
    return bits[:T].to(torch.int32)


# -------------------------------
# Quantized Linear with dual palettes (P4, P8)
# -------------------------------

class QuantizedLinearPalettes(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight_fp = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        # Buffers for palettes
        self.register_buffer("w4", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("w8", torch.empty(0, dtype=torch.int8), persistent=False)
        self.qp4: Optional[QParams] = None
        self.qp8: Optional[QParams] = None

    def prepare_palettes(self):
        with torch.no_grad():
            q4, qp4 = quantize_4bit_rowwise(self.weight_fp.data)
            q8, qp8 = quantize_8bit_rowwise(self.weight_fp.data)
            self.w4 = q4.clone()
            self.w8 = q8.clone()
            self.qp4 = qp4
            self.qp8 = qp8

    def forward_with_mask(self, x: torch.Tensor, eight_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]; eight_mask: [B] boolean selecting 8-bit palette for that row
        We split by mask, do two matmuls with dequantized weights, then scatter back.
        """
        B, _ = x.shape
        device = x.device
        out = torch.empty(B, self.weight_fp.shape[0], device=device, dtype=x.dtype)
        if eight_mask.any():
            w8 = affine_dequant_per_row(self.w8, self.qp8, symmetric=True)
            y8 = x[eight_mask] @ w8.t()
            out[eight_mask] = y8
        if (~eight_mask).any():
            w4 = affine_dequant_per_row(self.w4, self.qp4, symmetric=False)
            y4 = x[~eight_mask] @ w4.t()
            out[~eight_mask] = y4
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_all_fp(self, x: torch.Tensor):
        return F.linear(x, self.weight_fp, self.bias)

    def forward_all_4bit(self, x: torch.Tensor):
        w4 = affine_dequant_per_row(self.w4, self.qp4, symmetric=False)
        return F.linear(x, w4, self.bias)

    def forward_all_8bit(self, x: torch.Tensor):
        w8 = affine_dequant_per_row(self.w8, self.qp8, symmetric=True)
        return F.linear(x, w8, self.bias)


# -------------------------------
# Toy Transformer (decoder-only)
# -------------------------------

class ToySelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = QuantizedLinearPalettes(d_model, d_model, bias=False)
        self.k_proj = QuantizedLinearPalettes(d_model, d_model, bias=False)
        self.v_proj = QuantizedLinearPalettes(d_model, d_model, bias=False)
        self.o_proj = QuantizedLinearPalettes(d_model, d_model, bias=False)

    def prepare_palettes(self):
        self.q_proj.prepare_palettes()
        self.k_proj.prepare_palettes()
        self.v_proj.prepare_palettes()
        self.o_proj.prepare_palettes()

    def _split_heads(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        return x

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1,2).contiguous().view(B, T, H*D)

    def forward(self, x, method="fp", eight_mask=None):
        # x: [B,T,C]
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        if method == "fp":
            Q = self.q_proj.forward_all_fp(x_flat).view(B, T, C)
            K = self.k_proj.forward_all_fp(x_flat).view(B, T, C)
            V = self.v_proj.forward_all_fp(x_flat).view(B, T, C)
        elif method == "4b":
            Q = self.q_proj.forward_all_4bit(x_flat).view(B, T, C)
            K = self.k_proj.forward_all_4bit(x_flat).view(B, T, C)
            V = self.v_proj.forward_all_4bit(x_flat).view(B, T, C)
        elif method == "8b":
            Q = self.q_proj.forward_all_8bit(x_flat).view(B, T, C)
            K = self.k_proj.forward_all_8bit(x_flat).view(B, T, C)
            V = self.v_proj.forward_all_8bit(x_flat).view(B, T, C)
        elif method == "hatq":
            assert eight_mask is not None, "eight_mask required for hatq"
            Q = self.q_proj.forward_with_mask(x_flat, eight_mask.repeat_interleave(T)).view(B, T, C)
            K = self.k_proj.forward_with_mask(x_flat, eight_mask.repeat_interleave(T)).view(B, T, C)
            V = self.v_proj.forward_with_mask(x_flat, eight_mask.repeat_interleave(T)).view(B, T, C)
        else:
            raise ValueError("Unknown method")

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        B, H, T, D = Q.shape
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.tril(torch.ones(T, T, device=attn_scores.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = attn_scores.softmax(dim=-1)
        y = torch.matmul(attn_probs, V)

        y = self._merge_heads(y)
        if method == "fp":
            out = self.o_proj.forward_all_fp(y.reshape(B*T, C)).view(B, -1, C)
        elif method == "4b":
            out = self.o_proj.forward_all_4bit(y.reshape(B*T, C)).view(B, -1, C)
        elif method == "8b":
            out = self.o_proj.forward_all_8bit(y.reshape(B*T, C)).view(B, -1, C)
        elif method == "hatq":
            out = self.o_proj.forward_with_mask(y.reshape(B*T, C), eight_mask.repeat_interleave(y.shape[1])).view(B, -1, C)
        return out


class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ToySelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = QuantizedLinearPalettes(d_model, d_model*mlp_ratio)
        self.fc2 = QuantizedLinearPalettes(d_model*mlp_ratio, d_model)
        self.act = nn.GELU()

    def prepare_palettes(self):
        self.attn.prepare_palettes()
        self.fc1.prepare_palettes()
        self.fc2.prepare_palettes()

    def forward(self, x, method="fp", eight_mask=None):
        h = self.ln1(x)
        h = self.attn(h, method=method, eight_mask=eight_mask)
        x = x + h
        h2 = self.ln2(x)
        B, T, C = h2.shape
        if method == "fp":
            y = self.fc2.forward_all_fp(self.act(self.fc1.forward_all_fp(h2.reshape(B*T, C)))).view(B, T, C)
        elif method == "4b":
            y = self.fc2.forward_all_4bit(self.act(self.fc1.forward_all_4bit(h2.reshape(B*T, C)))).view(B, T, C)
        elif method == "8b":
            y = self.fc2.forward_all_8bit(self.act(self.fc1.forward_all_8bit(h2.reshape(B*T, C)))).view(B, T, C)
        elif method == "hatq":
            y_int = self.fc1.forward_with_mask(h2.reshape(B*T, C), eight_mask.repeat_interleave(T))
            y_act = self.act(y_int)
            y = self.fc2.forward_with_mask(y_act, eight_mask.repeat_interleave(T)).view(B, T, C)
        return x + y


class ToyTransformerLM(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, n_layers=2, n_heads=4, mlp_ratio=4, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([ToyTransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = QuantizedLinearPalettes(d_model, vocab_size, bias=False)
        self.max_seq = max_seq

    def prepare_palettes(self):
        for b in self.blocks:
            b.prepare_palettes()
        self.lm_head.prepare_palettes()

    def forward(self, input_ids, method="fp", eight_mask=None, return_hidden=False):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x, method=method, eight_mask=eight_mask)
        x = self.ln_f(x)
        B, T, C = x.shape
        if method == "fp":
            logits = self.lm_head.forward_all_fp(x.reshape(B*T, C)).view(B, T, -1)
        elif method == "4b":
            logits = self.lm_head.forward_all_4bit(x.reshape(B*T, C)).view(B, T, -1)
        elif method == "8b":
            logits = self.lm_head.forward_all_8bit(x.reshape(B*T, C)).view(B, T, -1)
        elif method == "hatq":
            logits = self.lm_head.forward_with_mask(x.reshape(B*T, C), eight_mask.repeat_interleave(T)).view(B, T, -1)
        return (logits, x) if return_hidden else logits


# -------------------------------
# Importance predictor + thresholds (STE)
# -------------------------------

class LinearProbe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
    def forward(self, h):  # h: [B,T,H] or [N,H]
        if h.dim() == 3:
            return self.proj(h).squeeze(-1)
        else:
            return self.proj(h)


class STEGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Thresholds(nn.Module):
    def __init__(self, init_vals=(-0.5, 0.0, 0.5)):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(list(init_vals), dtype=torch.float32))
    def forward(self, s_hat: torch.Tensor) -> torch.Tensor:
        t1, t2, t3 = self.tau[0], self.tau[1], self.tau[2]
        g1 = STEGate.apply(s_hat - t1)
        g2 = STEGate.apply(s_hat - t2)
        g3 = STEGate.apply(s_hat - t3)
        s_t = g1 + g2 + g3
        return s_t.clamp(0,3)


# -------------------------------
# Training and evaluation helpers shared
# -------------------------------

@torch.no_grad()
def teacher_forced_loss(model, data_loader, device, method="fp", eight_mask=None):
    ce_losses = []
    top1_acc = []
    for batch in data_loader:
        ids = batch["input_ids"].to(device)
        logits = model(ids, method=method, eight_mask=eight_mask)
        logits = logits[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        ce_losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
        top1_acc.append(acc)
    return float(np.mean(ce_losses)), float(np.mean(top1_acc))


def collect_hidden_and_ce(model_fp, model_nf4, data_loader, device):
    H_list = []
    Y_list = []
    scores = []
    with torch.no_grad():
        for batch in data_loader:
            ids = batch["input_ids"].to(device)
            logits_fp, h_fp = model_fp(ids, method="fp", eight_mask=None, return_hidden=True)
            logits_4b = model_nf4(ids, method="4b", eight_mask=None)
            # Per-token CE
            ce_fp = F.cross_entropy(logits_fp[:, :-1, :].reshape(-1, logits_fp.size(-1)), ids[:, 1:].reshape(-1), reduction="none").view(ids.size(0), -1)
            ce_4b = F.cross_entropy(logits_4b[:, :-1, :].reshape(-1, logits_4b.size(-1)), ids[:, 1:].reshape(-1), reduction="none").view(ids.size(0), -1)
            y = (ce_4b - ce_fp > 0.15).to(torch.int64)
            H_list.append(h_fp[:, :-1, :].cpu())
            Y_list.append(y.cpu())
            scores.append((ce_4b - ce_fp).cpu())
    H = torch.cat(H_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    S = torch.cat(scores, dim=0)
    return H, Y, S


def calibrate_thresholds(model: ToyTransformerLM, probe: LinearProbe, th: Thresholds, calib_loader, device, budget_B=1.2, steps=50, lr=5e-3):
    opt = torch.optim.Adam([th.tau], lr=lr)
    model.eval(); probe.eval()
    for step, batch in zip(range(steps), calib_loader):
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _, h_fp = model(ids, method="fp", return_hidden=True)
        s_hat = probe(h_fp[:, :-1, :])  # [B,T-1]
        s_t = th(s_hat).detach()
        eight_mask = (s_t[:, :].reshape(-1) >= 2)
        logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
        ce = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
        E_s = s_t.float().mean()
        loss = ce + 0.1 * torch.clamp(E_s - budget_B, min=0)
        opt.zero_grad(); loss.backward(); opt.step()
    return th


def per_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    ent = -(probs * (probs+1e-9).log()).sum(dim=-1)
    return ent


# -------------------------------
# Training routine for the toy LM
# -------------------------------

def train_toy_model(device: str = "cpu", images_dir: Optional[str] = None, cfg: Optional[dict] = None, verbose: bool = True):
    set_seed(123)
    vocab_size = (cfg or {}).get("vocab_size", 128)
    d_model = (cfg or {}).get("d_model", 64)
    n_layers = (cfg or {}).get("n_layers", 2)
    n_heads = (cfg or {}).get("n_heads", 4)
    mlp_ratio = (cfg or {}).get("mlp_ratio", 4)
    max_seq = (cfg or {}).get("max_seq", 512)
    epochs = (cfg or {}).get("epochs", 3)
    batch_size = (cfg or {}).get("batch_size", 16)
    n_train = (cfg or {}).get("n_train", 256)
    n_valid = (cfg or {}).get("n_valid", 64)
    lr = (cfg or {}).get("lr", 3e-3)

    model = ToyTransformerLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, mlp_ratio=mlp_ratio, max_seq=max_seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Data
    train_ds = SyntheticSeqDataset(n_samples=n_train, seq_len=64, vocab_size=vocab_size, mix=True, seed=100)
    valid_ds = SyntheticSeqDataset(n_samples=n_valid, seq_len=64, vocab_size=vocab_size, mix=True, seed=101)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Train a few epochs for sanity
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            logits = model(ids, method="fp")
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        model.eval()
        v_loss, v_acc = teacher_forced_loss(model, valid_loader, device, method="fp")
        train_losses.append(epoch_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        if verbose:
            print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={v_loss:.4f}, val_acc={v_acc:.4f}")

    # Prepare palettes for quantized paths
    model.prepare_palettes()

    # Save training curves
    if images_dir is not None:
        ensure_dir(images_dir)
        plt.figure(figsize=(5,3))
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="valid")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training/Validation Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "training_loss_hatq.pdf"), bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(5,3))
        plt.plot(val_accs, label="valid_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "accuracy_hatq.pdf"), bbox_inches="tight")
        plt.close()

    return model, train_loader, valid_loader


# -------------------------------
# Quick predictor training helper (used by evaluation)
# -------------------------------

def train_probe_on_calib(model: ToyTransformerLM, calib_loader, device: str, images_dir: Optional[str] = None):
    H, Y, S = collect_hidden_and_ce(model, model, calib_loader, device)
    probe = LinearProbe(H.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    # Train a few steps
    probe.train()
    for _ in range(5):
        idx = torch.randint(0, H.shape[0], (8,))
        h = H[idx].to(device)
        y = Y[idx].to(device)
        logits = probe(h)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1).float())
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate AUC/PR on calib set
    with torch.no_grad():
        scores = probe(H.to(device)).cpu().view(-1)
    y_true = Y.view(-1).numpy()
    y_score = scores.numpy()
    try:
        auc = roc_auc_score(y_true, y_score)
        pr = average_precision_score(y_true, y_score)
    except Exception:
        auc, pr = float("nan"), float("nan")
    print(f"Predictor ROC-AUC={auc:.3f}, PR-AUC={pr:.3f}")

    # Plot probe score distribution
    if images_dir is not None:
        ensure_dir(images_dir)
        plt.figure(figsize=(5,3))
        try:
            sns.kdeplot(y_score[y_true==0], label="easy", fill=True)
            sns.kdeplot(y_score[y_true==1], label="hard", fill=True)
        except Exception:
            # fallback to hist
            plt.hist(y_score[y_true==0], bins=30, alpha=0.5, label="easy")
            plt.hist(y_score[y_true==1], bins=30, alpha=0.5, label="hard")
        plt.xlabel("Probe score s_hat")
        plt.ylabel("Density")
        plt.title("Probe score distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "predictor_roc.pdf"), bbox_inches="tight")
        plt.close()

    return probe, {"auc": float(auc), "pr": float(pr)}
