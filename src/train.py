import os
import math
import pathlib
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 32, alpha: int = 16):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling

class GatedBlock(nn.Module):
    def __init__(self, host_dim: int, rank: int = 32):
        super().__init__()
        self.adapter = LoRAAdapter(host_dim, rank)
        self.gate = nn.Parameter(torch.ones(host_dim))

    def forward(self, x):
        adapted = self.adapter(x)
        gated = adapted * self.gate
        return x + gated

class SapStudent(nn.Module):
    def __init__(self, vocab: int = 1000, hidden: int = 128, layers: int = 4, rank: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([GatedBlock(hidden, rank) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.proj = nn.Linear(hidden, 64, bias=False)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h

class SapCotdLoss(nn.Module):
    def __init__(self, lambda_kl=1.0, beta_sup=2.0, gamma_nce=0.1, temperature=0.07):
        super().__init__()
        self.lm_kl = nn.KLDivLoss(reduction="batchmean")
        self.lambda_kl = lambda_kl
        self.beta_sup = beta_sup
        self.gamma_nce = gamma_nce
        self.temperature = temperature

    def forward(self, student_logits, student_hidden, batch, model):
        B = len(batch["z_t"])
        rep_loss = 0.0
        kl_loss = 0.0
        sup_loss = 0.0
        nce_loss = 0.0
        
        for i in range(B):
            z_t = batch["z_t"][i].to(student_hidden.device)
            m_t = batch["m_t"][i].to(student_hidden.device)
            p_t = batch["p_t"][i].to(student_hidden.device)
            
            S = min(z_t.size(0), student_hidden.size(1))
            h_s = student_hidden[i, :S]
            proj = model.proj(h_s)
            rep_loss += F.mse_loss(proj, z_t[:S])
            
            kl_loss += self.lm_kl(
                F.log_softmax(student_logits[i, :S], dim=-1),
                F.softmax(p_t[:S].detach(), dim=-1)
            )
            
            if m_t.size(-1) == h_s.size(-1):
                sup_loss += ((1 - m_t[:S]) * h_s.abs()).mean()
            
            z = F.normalize(z_t[:S], dim=-1)
            sims = torch.matmul(z, z.t()) / self.temperature
            nce_loss += (-torch.diag(F.log_softmax(sims, dim=-1)).mean())
        
        rep_loss = rep_loss / B
        kl_loss = kl_loss / B
        sup_loss = sup_loss / B
        nce_loss = nce_loss / B
        
        total = rep_loss + self.lambda_kl * kl_loss + self.beta_sup * sup_loss + self.gamma_nce * nce_loss
        return total, {
            "rep": rep_loss.item() if hasattr(rep_loss, 'item') else float(rep_loss),
            "kl": kl_loss.item() if hasattr(kl_loss, 'item') else float(kl_loss),
            "sup": sup_loss.item() if hasattr(sup_loss, 'item') else float(sup_loss),
            "nce": nce_loss.item() if hasattr(nce_loss, 'item') else float(nce_loss),
        }

def experiment1_train(train_loader, model, device, epochs=1, lr=2e-4, 
                     lambda_kl=1, beta_sup=2, gamma_nce=0.1, out_dir="exp1_ckpt"):
    os.makedirs(out_dir, exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    criterion = SapCotdLoss(lambda_kl, beta_sup, gamma_nce)
    
    losses, steps = [], []
    step = 0
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            step += 1
            inp = batch["input_ids"].to(device)
            logits, hidden = model(inp)
            loss, parts = criterion(logits, hidden, batch, model)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            losses.append(loss.item())
            steps.append(step)
            
            if step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        torch.save(model.state_dict(), f"{out_dir}/student_epoch{epoch+1}.pt")
    
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=steps, y=losses)
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.title("SAP-CoTD training loss")
    pdf_path = pathlib.Path(out_dir) / "training_loss.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    print(f"[Exp-1] training finished, curve saved to {pdf_path}")
    return losses, str(pdf_path)
