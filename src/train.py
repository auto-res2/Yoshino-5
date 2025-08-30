import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import get_device


# ------------------------------
# HATQ Bit-plane Linear
# ------------------------------
class HATQLinear(nn.Module):
    """Bit-plane linear layer with per-token dynamic precision.
    - Stores planes as int8 tensors with values in {-1, +1} for prototyping.
    - Has trainable per-plane per-out scale (small parameter set).
    - Accepts token_bits specifying how many most significant planes to use per token.
    """
    def __init__(self, in_f: int, out_f: int, planes: int = 8, min_msb: int = 3, device: Optional[torch.device] = None):
        super().__init__()
        self.in_f, self.out_f, self.planes = in_f, out_f, planes
        self.min_msb = min_msb
        device = device if device is not None else get_device()
        self.register_buffer('B', torch.empty(planes, out_f, in_f, dtype=torch.int8, device=device))
        self.scales = nn.Parameter(torch.ones(planes, out_f, dtype=torch.float32, device=device))
        self.register_buffer('bit_counter', torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def init_random_planes(self):
        self.B.copy_((torch.randint(0, 2, self.B.shape, device=self.B.device, dtype=torch.int8) * 2 - 1))
        self.scales.data.copy_(torch.linspace(0.5, 1.0, steps=self.planes, device=self.scales.device).unsqueeze(1).expand(self.planes, self.out_f))

    @torch.no_grad()
    def load_planes(self, B_planes: torch.Tensor, scales: Optional[torch.Tensor] = None):
        assert B_planes.shape == self.B.shape
        self.B.copy_(B_planes)
        if scales is not None:
            assert scales.shape == self.scales.shape
            self.scales.copy_(scales)

    def forward(self, x: torch.Tensor, token_bits: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            B, T, Din = x.shape
            x_flat = x.reshape(B*T, Din)
        elif x.dim() == 2:
            x_flat = x
            B = None
            T = None
        else:
            raise ValueError("HATQLinear expects x of shape [B,T,D] or [N,D]")

        if token_bits is None:
            token_bits = torch.full((x_flat.size(0),), 4, device=x_flat.device, dtype=torch.int32)
        else:
            if token_bits.dim() == 2:
                token_bits = token_bits.reshape(-1)
            token_bits = token_bits.to(dtype=torch.int32, device=x_flat.device)
        token_bits = torch.clamp(token_bits, min=self.min_msb, max=self.planes)

        sorted_bits, sort_idx = torch.sort(token_bits)
        unique_bits = torch.unique_consecutive(sorted_bits)

        x_sorted = x_flat.index_select(0, sort_idx)
        out_sorted = torch.empty(x_sorted.size(0), self.out_f, device=x_sorted.device, dtype=torch.float32)

        start = 0
        for k in unique_bits.tolist():
            count = int((sorted_bits == k).sum().item())
            end = start + count
            xk = x_sorted[start:end]
            acc = None
            for p in range(self.planes - k, self.planes):
                Bp = self.B[p].to(dtype=torch.float32)
                sp = self.scales[p]
                contrib = torch.matmul(xk, Bp.t())
                contrib = contrib * sp
                acc = contrib if acc is None else acc + contrib
            out_sorted[start:end] = acc
            start = end

        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
        out = out_sorted.index_select(0, inv_idx)

        self.bit_counter += token_bits.sum().detach().cpu()

        if original_shape[0] is not None and len(original_shape) == 3:
            return out.reshape(original_shape[0], original_shape[1], self.out_f)
        return out


# ------------------------------
# Tiny HATQ Transformer
# ------------------------------
class TinyHATQBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, planes: int = 8, device: Optional[torch.device] = None):
        super().__init__()
        device = device if device is not None else get_device()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_model % self.n_heads == 0
        self.q_proj = HATQLinear(d_model, d_model, planes=planes, device=device)
        self.k_proj = HATQLinear(d_model, d_model, planes=planes, device=device)
        self.v_proj = HATQLinear(d_model, d_model, planes=planes, device=device)
        self.o_proj = HATQLinear(d_model, d_model, planes=planes, device=device)
        self.fc1 = HATQLinear(d_model, d_ff, planes=planes, device=device)
        self.fc2 = HATQLinear(d_ff, d_model, planes=planes, device=device)
        self.ln1 = nn.LayerNorm(d_model).to(device)
        self.ln2 = nn.LayerNorm(d_model).to(device)
        for mod in [self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.fc1, self.fc2]:
            mod.init_random_planes()

    def self_attn(self, x: torch.Tensor, token_bits: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        q = self.q_proj(x, token_bits)
        k = self.k_proj(x, token_bits)
        v = self.v_proj(x, token_bits)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, H)
        out = self.o_proj(context, token_bits)
        return out

    def forward(self, x: torch.Tensor, token_bits_layer: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), token_bits_layer)
        y = self.fc2(F.gelu(self.fc1(self.ln2(x), token_bits_layer), approximate='tanh'), token_bits_layer)
        x = x + y
        return x


class TinyHATQTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, d_ff: int = 256,
                 planes: int = 8, max_len: int = 512, device: Optional[torch.device] = None):
        super().__init__()
        device = device if device is not None else get_device()
        self.config = type('cfg', (), {})()
        self.config.num_hidden_layers = n_layers
        self.device_ = device
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model, device=device))
        self.blocks = nn.ModuleList([
            TinyHATQBlock(d_model, n_heads, d_ff, planes=planes, device=device) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model).to(device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device)
        self.current_bits_btl: Optional[torch.Tensor] = None
        self.forced_bits: int = 4

    def _set_token_bits(self, bits_btl: torch.Tensor):
        self.current_bits_btl = bits_btl

    def _force_bits(self, k: int):
        self.current_bits_btl = None
        self.forced_bits = k

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids) + self.pos_embed[:, :T, :]
        bits_btl = self.current_bits_btl
        for l, block in enumerate(self.blocks):
            if bits_btl is not None:
                token_bits_layer = bits_btl[:, :, l]
            else:
                token_bits_layer = torch.full((B, T), getattr(self, 'forced_bits', 4), device=device, dtype=torch.int32)
            x = block(x, token_bits_layer)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_hidden:
            return logits, x
        return logits


# ------------------------------
# Controllers
# ------------------------------
class GlobalBudgetRNN(nn.Module):
    def __init__(self, n_layers: int, hist: int = 8, hidden: int = 64, min_bits: int = 3, max_bits: int = 8, device: Optional[torch.device] = None):
        super().__init__()
        device = device if device is not None else get_device()
        self.gru = nn.GRU(input_size=n_layers, hidden_size=hidden, batch_first=True).to(device)
        self.out = nn.Linear(hidden, n_layers).to(device)
        self.min_bits, self.max_bits = min_bits, max_bits

    def forward(self, kv_stats_seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(kv_stats_seq)
        logits = self.out(h[:, -1])
        probs = torch.sigmoid(logits)
        cont = self.min_bits + (self.max_bits - self.min_bits) * probs
        hard = torch.clamp(cont.round(), self.min_bits, self.max_bits)
        return hard.detach() + (cont - cont.detach())


class TokenMaskBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 2, device: Optional[torch.device] = None):
        super().__init__()
        device = device if device is not None else get_device()
        self.ln = nn.LayerNorm(hidden_size).to(device)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=heads, batch_first=True).to(device)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size), nn.GELU(), nn.Linear(4*hidden_size, 1)).to(device)

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        x = self.ln(last_hidden)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        logits = self.mlp(attn_out).squeeze(-1)
        return logits


class BudgetHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, global_bits_bt: torch.Tensor, token_logits_bt: torch.Tensor, enable_mask: bool = True) -> torch.Tensor:
        if not enable_mask:
            return torch.zeros_like(token_logits_bt)
        probs = torch.sigmoid(token_logits_bt)
        if self.training:
            noise = torch.empty_like(probs).uniform_(-0.5, 0.5)
            hard01 = torch.clamp((probs + noise).round(), 0, 1)
        else:
            hard01 = torch.clamp(probs.round(), 0, 1)
        return hard01.detach() + (probs - probs.detach())


# ------------------------------
# Training
# ------------------------------
@dataclass
class TrainConfig:
    steps: int = 200
    lr: float = 2e-3
    lambda_nll: float = 0.8
    grad_clip: float = 1.0


def train_controller_and_scales(model: TinyHATQTransformer,
                                data: List[dict],
                                controller_modules: Tuple[GlobalBudgetRNN, TokenMaskBlock, BudgetHead],
                                cfg: TrainConfig,
                                print_every: int = 50) -> List[float]:
    model.train()
    gb_rnn, mask_block, budget_head = controller_modules
    gb_rnn.train(); mask_block.train(); budget_head.train()
    params = list(gb_rnn.parameters()) + list(mask_block.parameters()) + list(budget_head.parameters())
    for m in model.modules():
        if isinstance(m, HATQLinear):
            params.append(m.scales)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-2)

    losses = []
    for step in range(cfg.steps):
        batch = data[step % len(data)]
        input_ids = batch["input_ids"]
        B, _T = input_ids.shape
        device = input_ids.device
        kv_stats = torch.zeros(B, 8, model.config.num_hidden_layers, device=device)
        logits, hidden = model(input_ids, return_hidden=True)
        gbits = gb_rnn(kv_stats)
        token_logits = mask_block(hidden)
        extra_mask = budget_head(gbits, token_logits, enable_mask=True)
        bits_btl = torch.clamp(3 + extra_mask.unsqueeze(-1).expand(-1, -1, model.config.num_hidden_layers), 3, 8)
        model._set_token_bits(bits_btl)
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='mean')
        lat = bits_btl.float().mean()
        loss = cfg.lambda_nll * nll + (1 - cfg.lambda_nll) * lat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % print_every == 0 or step == 0:
            print(f"[Train step {step+1}/{cfg.steps}] loss={loss.item():.4f}, nll={nll.item():.4f}, latency_proxy={lat.item():.4f}")
    return losses


def build_model_and_controller(vocab_size: int,
                               d_model: int,
                               n_layers: int,
                               n_heads: int,
                               d_ff: int,
                               planes: int,
                               max_len: int,
                               device: Optional[torch.device] = None):
    device = device if device is not None else get_device()
    model = TinyHATQTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, planes=planes, max_len=max_len, device=device)
    gb_rnn = GlobalBudgetRNN(n_layers=n_layers, hist=8, hidden=max(32, d_model//2), device=device)
    mask_block = TokenMaskBlock(hidden_size=d_model, heads=min(4, n_heads), device=device)
    budget_head = BudgetHead()
    return model, (gb_rnn, mask_block, budget_head)
