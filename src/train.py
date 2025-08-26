import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple
import time

class CLEABlock(nn.Module):
    
    def __init__(self, d_in: int, rank: int = 4):
        super().__init__()
        self.d_in = d_in
        self.W0 = nn.Linear(d_in, d_in, bias=False)
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, d_in) * 0.02)

    def forward(self, x):
        W = self.W0.weight + self.U @ self.V
        return torch.einsum("btd,df->btf", x, W)

class AORAttention(nn.Module):
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.alpha = nn.Parameter(torch.zeros(num_heads))
        
        self.register_buffer("pos_abs", self._positional_encoding(64, embed_dim), persistent=False)
        self.register_buffer("pos_rel", self._relative_encoding(64, embed_dim), persistent=False)

    @staticmethod
    def _positional_encoding(max_len: int, d: int):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    @staticmethod
    def _relative_encoding(max_len: int, d: int):
        rel = torch.zeros(max_len, d)
        for i in range(max_len):
            rel[i] = (torch.arange(max_len) - i).abs()[:d]
        rel = rel / rel.max() if rel.max() > 0 else rel
        return rel

    def forward(self, x):
        B, T, D = x.shape
        
        sigma_alpha = torch.sigmoid(self.alpha)
        pos_abs = self.pos_abs[:T]
        pos_rel = self.pos_rel[:T]
        
        pos_mixed = (sigma_alpha.view(1, -1, 1) * pos_abs.unsqueeze(1) + 
                    (1 - sigma_alpha.view(1, -1, 1)) * pos_rel.unsqueeze(1))
        pos_mixed = pos_mixed.mean(1)
        
        x_pos = x + pos_mixed.unsqueeze(0)
        
        attn_out, attn_weights = self.mha(x_pos, x_pos, x_pos, need_weights=True)
        return attn_out, attn_weights

class PIVOTXRTransformer(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 2, 
                 num_layers: int = 2, use_aor: bool = True, use_clea: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_aor = use_aor
        self.use_clea = use_clea
        
        self.emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        
        self.clea = CLEABlock(d_model) if use_clea else nn.Identity()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': AORAttention(d_model, num_heads) if use_aor else 
                       nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, vocab_size + 1)
        
        self.attention_weights = []

    def forward(self, x):
        h = self.emb(x)
        h = self.clea(h)
        
        self.attention_weights = []
        
        for layer in self.layers:
            if self.use_aor:
                attn_out, attn_w = layer['attn'](h)
                self.attention_weights.append(attn_w)
            else:
                attn_out, attn_w = layer['attn'](h, h, h, need_weights=True)
                self.attention_weights.append(attn_w)
            
            h = layer['norm1'](h + attn_out)
            
            ff_out = layer['ff'](h)
            h = layer['norm2'](h + ff_out)
        
        logits = self.out_proj(h)
        return logits, self.attention_weights

def train_epoch(model: PIVOTXRTransformer, loader: DataLoader, 
                optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for q, a in loader:
        q, a = q.to(device), a.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(q)
        
        loss = F.cross_entropy(logits[:, 0, :], a[:, 0])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def create_model_variants(vocab_size: int):
    models = {
        "Base-FT": PIVOTXRTransformer(vocab_size, use_aor=False, use_clea=False),
        "PIVOT-++": PIVOTXRTransformer(vocab_size, use_aor=False, use_clea=True),
        "PIVOT-XR": PIVOTXRTransformer(vocab_size, use_aor=True, use_clea=True),
    }
    return models
