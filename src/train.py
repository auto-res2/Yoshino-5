import math
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess import collate_batch
from evaluate import evaluate_em, measure_inference


# -------------------------
# DoRA/LoRA style layers
# -------------------------

class DoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 use_dora: bool = True, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.use_dora = use_dora
        # Frozen base weight simulating quantized backbone; here kept as fp32 but requires_grad=False
        W0 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(W0, a=math.sqrt(5))
        self.W0 = nn.Parameter(W0, requires_grad=False)
        # Low-rank update
        self.A = nn.Parameter(torch.zeros(out_features, r))
        self.B = nn.Parameter(torch.zeros(r, in_features))
        nn.init.normal_(self.A, std=1e-3)
        nn.init.normal_(self.B, std=1e-3)
        # DoRA learned per-row magnitude
        self.magnitude = nn.Parameter(torch.ones(out_features)) if use_dora else None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def effective_weight(self) -> torch.Tensor:
        W = self.W0 + self.A @ self.B
        if self.use_dora and self.magnitude is not None:
            row_norm = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
            W_dir = W / row_norm
            W = self.magnitude[:, None] * W_dir
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.effective_weight()
        y = x @ W.t()
        if self.bias is not None:
            y = y + self.bias
        return y


class SelfAttentionMAGMA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int = 16,
                 r: int = 8, use_dora: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.q_proj = DoRALinear(d_model, d_model, r=r, use_dora=use_dora)
        self.k_proj = DoRALinear(d_model, d_model, r=r, use_dora=use_dora)
        self.v_proj = DoRALinear(d_model, d_model, r=r, use_dora=use_dora)
        self.o_proj = DoRALinear(d_model, d_model, r=r, use_dora=use_dora)
        self._last_q: Optional[torch.Tensor] = None
        self._last_k: Optional[torch.Tensor] = None

    def get_row_norms(self, proj_name: str) -> torch.Tensor:
        proj: DoRALinear = getattr(self, proj_name)
        if proj.magnitude is not None:
            mag = proj.magnitude.detach()  # [d_model]
            mag = mag.view(self.n_heads, self.head_dim)
            return mag.norm(dim=1)  # [H]
        return torch.ones(self.n_heads, device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                magma_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,S,D]
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()  # [B,H,S,Dh]
        k = self.k_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()

        self._last_q = q.detach()
        self._last_k = k.detach()

        device = x.device
        out = torch.empty_like(q)
        KB = (S + self.block_size - 1) // self.block_size
        if magma_mask is None:
            magma_mask = torch.ones(H, KB, dtype=torch.bool, device=device)

        # Precompute kept token indices per head
        kept_tok_idx: List[torch.Tensor] = []
        for h in range(H):
            kb_idx = torch.nonzero(magma_mask[h], as_tuple=False).squeeze(-1)
            if kb_idx.numel() == 0:
                kb_idx = torch.tensor([0], device=device)
            tok_idx = (kb_idx[:, None] * self.block_size + torch.arange(self.block_size, device=device)[None, :]).reshape(-1)
            tok_idx = tok_idx[tok_idx < S]
            kept_tok_idx.append(tok_idx)

        scale = 1.0 / math.sqrt(Dh)
        for b in range(B):
            valid_len = int(attn_mask[b].sum().item()) if attn_mask is not None else S
            for h in range(H):
                Q = q[b, h, :valid_len, :]  # [Sv,Dh]
                K = k[b, h, kept_tok_idx[h], :]
                V = v[b, h, kept_tok_idx[h], :]
                Kvalid_mask = kept_tok_idx[h] < valid_len
                if Kvalid_mask.sum() == 0:
                    Kvalid_mask = torch.ones_like(kept_tok_idx[h], dtype=torch.bool)
                K = K[Kvalid_mask]
                V = V[Kvalid_mask]
                Sk = K.shape[0]
                # causal mask mapped to compacted keys
                causal_comp = torch.zeros(Q.shape[0], Sk, dtype=torch.bool, device=device)
                orig_k_pos = kept_tok_idx[h][Kvalid_mask]
                for qi in range(Q.shape[0]):
                    causal_comp[qi] = orig_k_pos <= qi
                logits = (Q @ K.t()) * scale
                logits = logits.masked_fill(~causal_comp, -1e9)
                attn = torch.softmax(logits, dim=-1)
                O = attn @ V
                out_bh = torch.zeros(S, Dh, device=device, dtype=O.dtype)
                out_bh[:Q.shape[0]] = O
                out[b, h] = out_bh
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.o_proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, dim_ff: int = 128, block_size: int = 16,
                 r: int = 8, use_dora: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(32768, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': SelfAttentionMAGMA(d_model, n_heads, block_size=block_size, r=r, use_dora=use_dora),
                'ln2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.masks: Optional[torch.Tensor] = None  # [L,H,KB]

    def set_masks(self, M_lh: Optional[torch.Tensor]):
        self.masks = M_lh

    def get_last_qk(self) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        qk = []
        for l in range(self.n_layers):
            attn: SelfAttentionMAGMA = self.layers[l]['attn']
            qk.append((attn._last_q, attn._last_k))
        return qk

    def get_head_magnitudes(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        mQ, mK = [], []
        for l in range(self.n_layers):
            attn: SelfAttentionMAGMA = self.layers[l]['attn']
            mQ.append(attn.get_row_norms('q_proj'))
            mK.append(attn.get_row_norms('k_proj'))
        return mQ, mK

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.emb(input_ids) + self.pos(pos_ids)
        for l in range(self.n_layers):
            h = self.layers[l]
            x = x + h['attn'](h['ln1'](x), attn_mask=attention_mask,
                               magma_mask=(None if self.masks is None else self.masks[l]))
            x = x + h['ff'](h['ln2'](x))
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 16) -> torch.Tensor:
        self.eval()
        out_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(out_ids, attention_mask=(out_ids != 0).long())
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_token], dim=1)
        return out_ids


# ---------------------
# MAGMA components
# ---------------------

@dataclass
class EMATable:
    n_layers: int
    n_heads: int
    n_blocks: int
    decay: float = 0.95

    def __post_init__(self):
        self.T = torch.zeros(self.n_layers, self.n_heads, self.n_blocks, dtype=torch.float32)
        self.N = torch.zeros_like(self.T)

    @torch.no_grad()
    def update_layer(self, layer_id: int, head_scores: torch.Tensor):
        # head_scores: [H, KB]
        self.T[layer_id] = self.decay * self.T[layer_id] + (1.0 - self.decay) * head_scores.cpu()
        self.N[layer_id] += 1

    def get_scores(self) -> torch.Tensor:
        return self.T / self.N.clamp_min(1.0)


@torch.no_grad()
def block_scores_from_qk(q: torch.Tensor, k: torch.Tensor,
                         mQ_norm: torch.Tensor, mK_norm: torch.Tensor,
                         block_size: int = 16) -> torch.Tensor:
    # q,k: [B,H,S,D]; mQ_norm,mK_norm: [H]
    qn = torch.linalg.vector_norm(q.float(), dim=-1)  # [B,H,S]
    kn = torch.linalg.vector_norm(k.float(), dim=-1)  # [B,H,S]
    scale = (mQ_norm.to(q.device)[:, None] * mK_norm.to(q.device)[:, None])  # [H,1]
    s = qn * kn * scale  # [B,H,S]
    B, H, S = s.shape
    KB = (S + block_size - 1) // block_size
    pad = KB * block_size - S
    if pad > 0:
        s = F.pad(s, (0, pad))
    s = s.view(B, H, KB, block_size).mean(dim=-1)  # [B,H,KB]
    return s.mean(dim=0).cpu()  # [H,KB]


@torch.no_grad()
def allocate_masks(T_scores: torch.Tensor, rho: float) -> torch.Tensor:
    # T_scores: [L,H,KB]
    L, H, KB = T_scores.shape
    total = L * H * KB
    keep = max(1, int((1.0 - rho) * total))
    flat = T_scores.reshape(-1)
    topk = torch.topk(flat, keep)
    M = torch.zeros_like(flat, dtype=torch.bool)
    M[topk.indices] = True
    M = M.view(L, H, KB)
    # ensure at least one block per head
    for l in range(L):
        for h in range(H):
            if not M[l, h].any():
                M[l, h, 0] = True
    return M


@torch.no_grad()
def random_masks_like(T_scores: torch.Tensor, rho: float, seed: int = 0) -> torch.Tensor:
    L, H, KB = T_scores.shape
    gen = torch.Generator()
    gen.manual_seed(seed)
    M = torch.rand(L, H, KB, generator=gen) < (1.0 - rho)
    for l in range(L):
        for h in range(H):
            if not M[l, h].any():
                M[l, h, 0] = True
    return M


class MAGMAController:
    def __init__(self, model: TinyTransformer, seq_len: int, rho_target: float = 0.9,
                 block_size: int = 16, decay: float = 0.95, mask_update_every: int = 50,
                 sparsity_warmup_steps: int = 200):
        self.model = model
        self.seq_len = seq_len
        self.block_size = block_size
        self.KB = (seq_len + block_size - 1) // block_size
        self.ema = EMATable(model.n_layers, model.n_heads, self.KB, decay=decay)
        self.rho_target = rho_target
        self.mask_update_every = mask_update_every
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.step = 0
        self.M_lh = torch.ones(model.n_layers, model.n_heads, self.KB, dtype=torch.bool)

    def rho_at_step(self) -> float:
        if self.step >= self.sparsity_warmup_steps:
            return self.rho_target
        return 0.7 + (self.rho_target - 0.7) * (self.step / max(1, self.sparsity_warmup_steps))

    @torch.no_grad()
    def update_from_model(self):
        qk_list = self.model.get_last_qk()
        mQ, mK = self.model.get_head_magnitudes()
        for l, (q, k) in enumerate(qk_list):
            if q is None or k is None:
                continue
            head_scores = block_scores_from_qk(q, k, mQ[l], mK[l], block_size=self.block_size)
            self.ema.update_layer(l, head_scores)

    @torch.no_grad()
    def maybe_update_masks(self, use_random: bool = False, rng_seed: int = 0):
        self.step += 1
        if self.step % self.mask_update_every != 0:
            return self.M_lh
        T_scores = self.ema.get_scores()
        rho = self.rho_at_step()
        if use_random:
            self.M_lh = random_masks_like(T_scores, rho, seed=rng_seed)
        else:
            self.M_lh = allocate_masks(T_scores, rho)
        self.model.set_masks(self.M_lh.to(next(self.model.parameters()).device))
        return self.M_lh


# ---------------------
# Training API
# ---------------------

@dataclass
class TrainConfig:
    use_dora: bool = True
    use_magma: bool = True
    random_sparse: bool = False
    rho: float = 0.9
    lr: float = 3e-3
    batch_size: int = 8
    steps: int = 200
    mask_update_every: int = 50
    warmup_steps: int = 100
    block_size: int = 16
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    images_dir: str = '.research/iteration2/images'


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(model: TinyTransformer, train_loader: DataLoader, cfg: TrainConfig,
              valid_set, tokenizer, tag: str = 'magma') -> Dict:
    os.makedirs(cfg.images_dir, exist_ok=True)
    set_seed(cfg.seed)
    model.to(cfg.device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    losses: List[float] = []

    ctrl: Optional[MAGMAController] = None
    if cfg.use_magma:
        ctrl = MAGMAController(model, seq_len=train_loader.dataset.max_len,
                               rho_target=cfg.rho, block_size=cfg.block_size,
                               mask_update_every=cfg.mask_update_every, sparsity_warmup_steps=cfg.warmup_steps)
        model.set_masks(torch.ones(model.n_layers, model.n_heads, ctrl.KB, dtype=torch.bool, device=cfg.device))
    else:
        model.set_masks(None)

    print(f"[Train] tag={tag} use_dora={cfg.use_dora} use_magma={cfg.use_magma} random_sparse={cfg.random_sparse} rho={cfg.rho} steps={cfg.steps}")
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        if step >= cfg.steps:
            break
        input_ids = batch['input_ids'].to(cfg.device)
        labels = batch['labels'].to(cfg.device)
        attn_mask = batch['attention_mask'].to(cfg.device)
        _, loss = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"  step={step+1:4d}/{cfg.steps}, loss={loss.item():.4f}")
        if cfg.use_magma:
            ctrl.update_from_model()
            ctrl.maybe_update_masks(use_random=cfg.random_sparse, rng_seed=cfg.seed)
    dt = time.time() - t0
    print(f"[Train] tag={tag} done in {dt:.2f}s, mean loss={np.mean(losses):.4f}")

    # Save loss curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(5,3))
        plt.plot(losses, label=tag)
        plt.xlabel('Step'); plt.ylabel('Loss'); plt.legend()
        fpath = os.path.join(cfg.images_dir, f'training_loss_{tag}.pdf')
        plt.savefig(fpath, bbox_inches='tight'); plt.close()
        print(f"[Plot] Saved {fpath}")
    except Exception as e:
        print(f"[Plot][Warn] Could not save training loss plot: {e}")

    # Evaluate EM on validation set and measure throughput
    em, preds, golds = evaluate_em(model, valid_set, tokenizer, device=cfg.device)
    print(f"[Eval] tag={tag} EM={em*100:.2f}% on {len(valid_set)} samples")

    prompts = [ex['prompt'] for ex in valid_set.samples[:32]]
    eff = measure_inference(model, tokenizer, prompts, device=cfg.device)
    print(f"[Eff] tag={tag} throughput={eff['throughput_toks_per_s']:.1f} toks/s, peak_mem_gb={eff['peak_gb']:.3f}")

    M_lh = model.masks.detach().cpu() if model.masks is not None else None
    T_scores = (ctrl.ema.get_scores() if ctrl is not None else None)

    return {
        'losses': losses,
        'em': em,
        'preds': preds,
        'golds': golds,
        'throughput': eff,
        'M_lh': M_lh,
        'T_scores': T_scores,
        'controller': ctrl,
    }
