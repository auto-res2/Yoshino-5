import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class HyperSTConfig:
    use_spectral_gate: bool = True
    use_hyper_rank: bool = True
    use_token_prune: bool = True
    use_dora_sign: bool = True
    max_rank: int = 16
    keep_ratio: float = 0.1

class SpectralGate(nn.Module):
    def __init__(self, keep_ratio: float = 0.1):
        super().__init__()
        self.keep_ratio = keep_ratio
    
    @staticmethod
    def _dct(x):
        return torch.real(torch.fft.rfft(x, dim=1))
    
    @staticmethod
    def _idct(c, T):
        return torch.fft.irfft(c, n=T, dim=1)
    
    def forward(self, video: torch.Tensor):
        B, T, *rest = video.shape
        flat = video.view(B, T, -1)
        coeff = self._dct(flat)
        k = max(1, int(coeff.size(1) * self.keep_ratio))
        
        topk = torch.topk(coeff.abs().mean(-1), k=k, dim=1).indices
        mask = torch.zeros_like(coeff)
        mask.scatter_(1, topk.unsqueeze(-1).expand_as(coeff), 1.)
        masked = coeff * mask
        recon = self._idct(masked, T).view_as(video)
        
        return recon, topk

class HyperLoRA(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, max_rank: int = 16):
        super().__init__()
        self.max_rank = max_rank
        self.spectrum_proj = nn.Linear(224*224*3, 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, max_rank + 2)
        )
    
    def forward(self, spectrum: torch.Tensor, q_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = spectrum.mean(dim=1)
        if pooled.dim() > 2:
            pooled = pooled.view(pooled.size(0), -1)
        pooled = self.spectrum_proj(pooled)
        if q_embed.dim() > 2:
            q_embed = q_embed.view(q_embed.size(0), -1)
        x = torch.cat([pooled, q_embed], dim=-1)
        out = self.mlp(x)
        rank_logits = out[:, :self.max_rank]
        rank = torch.argmax(rank_logits, dim=-1) + 1
        lora_delta = out[:, -2:]
        return rank, lora_delta

class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(B * T, C, H, W)
        feat = self.pool(F.relu(self.conv(x))).view(B, T, 16)
        return feat.mean(1)

class TinyLanguage(nn.Module):
    def __init__(self, vocab: int = 30522, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
    
    def forward(self, tok):
        emb = self.embed(tok)
        _, h = self.rnn(emb)
        return h.squeeze(0)

class HyperSTModel(nn.Module):
    def __init__(self, cfg: HyperSTConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = TinyVision()
        self.lang = TinyLanguage()
        self.sg = SpectralGate(cfg.keep_ratio)
        self.hyper = HyperLoRA(in_dim=16 + 32, max_rank=cfg.max_rank)
        self.clf = nn.Linear(16 + 32, 5)
    
    def estimate_flops(self, rank: int, T: int):
        return 1e6 + rank * 1e5 + T * 2e4
    
    def forward(self, video: torch.Tensor, question_tok: torch.Tensor, budget: Optional[float] = None):
        B, T, _, _, _ = video.shape
        
        if self.cfg.use_spectral_gate:
            recon, topk = self.sg(video)
        else:
            recon = video
        
        if self.cfg.use_token_prune:
            k = max(1, int(T * 0.5))
            recon = recon[:, :k]
            T = k
        
        vis_feat = self.vision(recon)
        q_emb = self.lang(question_tok)
        
        if self.cfg.use_hyper_rank:
            rank, lora = self.hyper(recon, q_emb)
            rank = rank[0].item()
        else:
            rank, lora = 8, None
        
        if budget is not None:
            while self.estimate_flops(rank, T) > budget and rank > 1:
                rank -= 1
        
        if self.cfg.use_dora_sign:
            sign = torch.sign(torch.tensor(rank).float())
            vis_feat = vis_feat * sign
        
        fused = torch.cat([vis_feat, q_emb], dim=-1)
        logits = self.clf(fused)
        pred = logits.argmax(-1)
        
        return pred, rank, T
