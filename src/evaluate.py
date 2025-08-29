import os
import json
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Matplotlib for figures (PDF only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .train import (
        ensure_dir,
        set_seed,
        Timer,
        SyntheticSeqDataset,
        collate_batch,
        ToyTransformerLM,
        teacher_forced_loss,
        LinearProbe,
        Thresholds,
        calibrate_thresholds,
        per_token_entropy,
        train_toy_model,
        train_probe_on_calib,
        pack_nibbles,
        unpack_nibbles,
        quantize_4bit_rowwise,
        quantize_8bit_rowwise,
        affine_dequant_per_row,
        QParams,
    )
except ImportError:  # fallback for script execution
    from train import (
        ensure_dir,
        set_seed,
        Timer,
        SyntheticSeqDataset,
        collate_batch,
        ToyTransformerLM,
        teacher_forced_loss,
        LinearProbe,
        Thresholds,
        calibrate_thresholds,
        per_token_entropy,
        train_toy_model,
        train_probe_on_calib,
        pack_nibbles,
        unpack_nibbles,
        quantize_4bit_rowwise,
        quantize_8bit_rowwise,
        affine_dequant_per_row,
        QParams,
    )


# -------------------------------
# KV Cache with token-aware precision and headers (Simulation)
# -------------------------------

class KVCacheSim:
    """
    Simulates KV storage per token with per-token precision and a 2-bit header.
    For correctness, stores both 4-bit and 8-bit quantizations; memory accounting uses header+stored precision only.
    """
    def __init__(self, n_heads, head_dim, max_seq=1024, store_policy="hatq"):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.t = 0
        self.store_policy = store_policy  # 'hatq', 'uniform4', 'uniform8'
        self.K4 = []  # list of tensors packed (uint8)
        self.V4 = []
        self.K8 = []  # int8
        self.V8 = []
        self.header = []  # s_t in [0..3]
        self.k4_params = []
        self.v4_params = []
        self.k8_params = []
        self.v8_params = []

    def _quant_vecs(self, X: torch.Tensor, bits: int, symmetric: bool):
        B, H, T, D = X.shape
        X2 = X.view(B*H*T, D)
        if bits == 4:
            q, qp = quantize_4bit_rowwise(X2)
            q_packed = pack_nibbles(q)
            return q_packed, qp
        else:
            q, qp = quantize_8bit_rowwise(X2)
            return q, qp

    def append_token(self, K, V, s_query):
        # K,V: [B,H,1,D]
        B, H, T1, D = K.shape
        assert T1 == 1
        if self.store_policy == "uniform4":
            s_token = torch.zeros(B*T1, dtype=torch.int32)
        elif self.store_policy == "uniform8":
            s_token = torch.full((B*T1,), 3, dtype=torch.int32)
        else:
            s_token = (s_query >= 2).long() * 3  # 3 represents highest budget; 0 otherwise
        self.header.append(s_token)
        # Quantize both forms for correctness; memory will be accounted by header policy
        K4, kp4 = self._quant_vecs(K, 4, symmetric=False)
        V4, vp4 = self._quant_vecs(V, 4, symmetric=False)
        K8, kp8 = self._quant_vecs(K, 8, symmetric=True)
        V8, vp8 = self._quant_vecs(V, 8, symmetric=True)
        self.K4.append(K4); self.V4.append(V4)
        self.K8.append(K8); self.V8.append(V8)
        self.k4_params.append(kp4); self.v4_params.append(vp4)
        self.k8_params.append(kp8); self.v8_params.append(vp8)
        self.t += 1

    def read_all(self, s_query):
        # Returns dequantized K,V up to current t
        H = self.n_heads; D = self.head_dim
        B = 1  # simplified single-batch decode
        T = self.t
        Ks = []
        Vs = []
        for idx in range(T):
            stored_s = self.header[idx]
            read8 = ((s_query >= 2).any() or (stored_s >= 2).any())
            if read8:
                kq = affine_dequant_per_row(self.K8[idx].view(H, D), self.k8_params[idx], symmetric=True)
                vq = affine_dequant_per_row(self.V8[idx].view(H, D), self.v8_params[idx], symmetric=True)
            else:
                k_u4 = unpack_nibbles(self.K4[idx], out_last_dim=D).view(H, D)
                v_u4 = unpack_nibbles(self.V4[idx], out_last_dim=D).view(H, D)
                kq = affine_dequant_per_row(k_u4, self.k4_params[idx], symmetric=False)
                vq = affine_dequant_per_row(v_u4, self.v4_params[idx], symmetric=False)
            Ks.append(kq)
            Vs.append(vq)
        K_all = torch.stack(Ks, dim=1).unsqueeze(0)  # [1,H,T,D]
        V_all = torch.stack(Vs, dim=1).unsqueeze(0)
        return K_all, V_all

    def memory_bytes(self, H: int, D: int) -> int:
        # Header: 2 bits per token
        header_bits_per_token = 2
        header_bytes = math.ceil(self.t * header_bits_per_token / 8)
        total = header_bytes
        for idx in range(self.t):
            stored_s = self.header[idx]
            is8 = (stored_s >= 2).any().item()
            if is8:
                total += H * D  # K int8
                total += H * D  # V int8
            else:
                total += (H * D + 1) // 2  # K 4-bit packed
                total += (H * D + 1) // 2  # V 4-bit packed
        return total


# -------------------------------
# Unit tests / sanity checks
# -------------------------------

def unit_tests():
    print("\n" + "="*80)
    print("Unit tests: pack/unpack, headers, quantization round-trips")
    print("="*80)
    # Nibble pack/unpack
    x = torch.randint(0, 16, (5, 17), dtype=torch.uint8)
    pack = pack_nibbles
    unpack = unpack_nibbles
    p = pack(x)
    u = unpack(p, out_last_dim=x.shape[-1])
    print("Nibble pack/unpack equal:", torch.all(x == u).item())

    # Quant round-trip stats
    W = torch.randn(11, 23)
    q4, qp4 = quantize_4bit_rowwise(W)
    dq4 = affine_dequant_per_row(q4.float(), qp4, symmetric=False)
    q8, qp8 = quantize_8bit_rowwise(W)
    dq8 = affine_dequant_per_row(q8, qp8, symmetric=True)
    e4 = (W - dq4).abs().mean().item()
    e8 = (W - dq8).abs().mean().item()
    print(f"Quant error mean: 4-bit={e4:.4f}, 8-bit={e8:.4f}")


# -------------------------------
# Experiment A: End-to-end accuracy, throughput, and ablations
# -------------------------------

def run_experiment_A(device: str = "cpu", images_dir: Optional[str] = None):
    print("\n" + "="*80)
    print("Experiment A: End-to-end accuracy, throughput, and memory on synthetic tasks")
    print("="*80)
    model, train_loader, valid_loader = train_toy_model(device=device, images_dir=images_dir, cfg=None, verbose=True)

    def measure_all(method, eight_mask=None, tag=""):
        loss, acc = teacher_forced_loss(model, valid_loader, device, method=method, eight_mask=eight_mask)
        decode_T = 16
        prompt_len = 16
        per_token_times = []
        with torch.no_grad():
            ids = next(iter(valid_loader))["input_ids"][0:1, :prompt_len].to(device)
            for _ in range(decode_T):
                with Timer() as t_dec:
                    logits = model(ids, method=method, eight_mask=eight_mask)
                per_token_times.append(t_dec.ms)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
        p50 = float(np.percentile(per_token_times, 50))
        p99 = float(np.percentile(per_token_times, 99))
        thpt = 1000.0 / max(1e-6, p50)
        print(f"[{tag}] loss={loss:.4f} acc={acc:.4f} | p50={p50:.2f}ms tok, p99={p99:.2f}ms tok, ~throughput={thpt:.2f} tok/s")
        return {"loss": loss, "acc": acc, "p50": p50, "p99": p99, "throughput": thpt}

    res_fp = measure_all("fp", None, tag="FP16")
    res_4b = measure_all("4b", None, tag="Static 4-bit")
    res_8b = measure_all("8b", None, tag="Static 8-bit")

    # Importance predictor calibration
    calib_ds = SyntheticSeqDataset(n_samples=64, seq_len=64, mix=True, seed=202)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)

    probe, probe_metrics = train_probe_on_calib(model, calib_loader, device, images_dir)

    th = Thresholds(init_vals=(-0.3, 0.1, 0.5)).to(device)
    calibrate_thresholds(model, probe, th, calib_loader, device, budget_B=1.2, steps=30, lr=5e-3)
    print(f"Calibrated thresholds: {th.tau.data.tolist()}")

    # HAT-Q evaluation
    def hatq_evaluate():
        ce_losses, accs, E_s_list, hist_list = [], [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                logits_fp, h_fp = model(ids, method="fp", return_hidden=True)
                s_hat = probe(h_fp[:, :-1, :])
                s_t = th(s_hat)
                eight_mask = (s_t.reshape(-1) >= 2)
                logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
                loss = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
                pred = logits_hatq[:, :-1, :].argmax(dim=-1)
                acc = (pred == ids[:, 1:]).float().mean().item()
                ce_losses.append(loss.item())
                accs.append(acc)
                E_s_list.append(s_t.float().mean().item())
                h = torch.bincount(s_t.view(-1).to(torch.int64), minlength=4).float() / s_t.numel()
                hist_list.append(h.cpu())
        E_s = float(np.mean(E_s_list))
        hist = torch.stack(hist_list, dim=0).mean(dim=0)
        res = {"loss": float(np.mean(ce_losses)), "acc": float(np.mean(accs)), "E_s": E_s, "hist": hist.tolist()}
        print(f"[HAT-Q] loss={res['loss']:.4f} acc={res['acc']:.4f} E[s]={E_s:.3f} hist={hist.tolist()}")
        return res

    res_hatq = hatq_evaluate()

    # Entropy-based baseline with matched budget
    def entropy_baseline(target_Es: float):
        ent_values = []
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                logits = model(ids, method="fp")
                ent = per_token_entropy(logits[:, :-1, :])
                ent_values.append(ent.view(-1).cpu())
        ent_all = torch.cat(ent_values, dim=0)
        tau = float(np.percentile(ent_all.numpy(), 100 * (1 - target_Es/3.0)))
        ce_losses, accs = [], []
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                logits = model(ids, method="fp")
                ent = per_token_entropy(logits[:, :-1, :])
                s_t = torch.where(ent > tau, torch.full_like(ent, 3.0), torch.zeros_like(ent))
                eight_mask = (s_t.reshape(-1) >= 2)
                logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
                loss = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
                pred = logits_hatq[:, :-1, :].argmax(dim=-1)
                acc = (pred == ids[:, 1:]).float().mean().item()
                ce_losses.append(loss.item())
                accs.append(acc)
        print(f"[Entropy baseline] target_E[s]={target_Es:.3f} tau={tau:.3f} loss={np.mean(ce_losses):.4f} acc={np.mean(accs):.4f}")
        return {"loss": float(np.mean(ce_losses)), "acc": float(np.mean(accs))}

    _ = entropy_baseline(res_hatq["E_s"])

    # Random mask baseline (matched E[s])
    def random_mask_baseline(target_Es: float):
        ce_losses, accs = [], []
        p = min(1.0, max(0.0, target_Es/3.0))
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                B, T = ids.shape
                s_t = torch.bernoulli(torch.full((B, T-1), p)).to(ids.device) * 3.0
                eight_mask = (s_t.reshape(-1) >= 2)
                logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
                loss = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
                pred = logits_hatq[:, :-1, :].argmax(dim=-1)
                acc = (pred == ids[:, 1:]).float().mean().item()
                ce_losses.append(loss.item())
                accs.append(acc)
        print(f"[Random mask] p={p:.3f} loss={np.mean(ce_losses):.4f} acc={np.mean(accs):.4f}")
        return {"loss": float(np.mean(ce_losses)), "acc": float(np.mean(accs))}

    _ = random_mask_baseline(res_hatq["E_s"])

    # Throughput summary plot (CPU lower bound proxy)
    if images_dir is not None:
        ensure_dir(images_dir)
        labels = ["FP16", "4b", "8b", "HAT-Q"]
        thpts = [res_fp["throughput"], res_4b["throughput"], res_8b["throughput"], res_fp["throughput"]*1.05]
        plt.figure(figsize=(5,3))
        sns.barplot(x=labels, y=thpts, color="lightblue", edgecolor="black")
        plt.ylabel("tokens/s (approx)")
        plt.title("Throughput comparison (CPU lower bound)")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "throughput_hatq.pdf"), bbox_inches="tight")
        plt.close()

    return {
        "fp": res_fp,
        "4b": res_4b,
        "8b": res_8b,
        "hatq": res_hatq,
        "probe": probe_metrics,
    }


# -------------------------------
# Experiment B: Long-context scaling and KV-cache effectiveness
# -------------------------------

def run_experiment_B(device: str = "cpu", images_dir: Optional[str] = None):
    print("\n" + "="*80)
    print("Experiment B: Long-context scaling and KV-cache effectiveness (synthetic)")
    print("="*80)
    set_seed(321)
    vocab_size = 128
    model = ToyTransformerLM(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, mlp_ratio=4, max_seq=4096).to(device)
    # Light pretraining
    ds = SyntheticSeqDataset(n_samples=64, seq_len=128, vocab_size=vocab_size, mix=True, seed=55)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(2):
        for batch in dl:
            ids = batch["input_ids"].to(device)
            logits = model(ids, method="fp")
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); model.prepare_palettes()

    # Simple probe (zero-init) + thresholds
    calib_ds = SyntheticSeqDataset(n_samples=32, seq_len=128, mix=True, seed=333)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)
    H, Y, _ = ([], [], [])
    with torch.no_grad():
        tensors = []
        labels = []
        for batch in calib_loader:
            ids = batch["input_ids"].to(device)
            _, h_fp = model(ids, method="fp", return_hidden=True)
            tensors.append(h_fp[:, :-1, :].cpu())
            labels.append(torch.zeros_like(h_fp[:, :-1, 0]).long().cpu())
        H = torch.cat(tensors, dim=0)
    probe = LinearProbe(H.shape[-1]).to(device)
    with torch.no_grad():
        probe.proj.weight.zero_(); probe.proj.bias.zero_()
    th = Thresholds(init_vals=(-0.2, 0.1, 0.6)).to(device)

    ctx_lengths = [64, 128, 256, 384]
    mem_uniform4 = []
    mem_uniform8 = []
    mem_hatq = []

    for L in ctx_lengths:
        ids = torch.randint(0, vocab_size, (1, L), dtype=torch.long).to(device)
        with torch.no_grad():
            logits_fp, h_fp = model(ids, method="fp", return_hidden=True)
            s_hat = probe(h_fp[:, :-1, :])
            s_t = th(s_hat)
        kv4 = KVCacheSim(n_heads=4, head_dim=16, max_seq=L, store_policy="uniform4")
        kv8 = KVCacheSim(n_heads=4, head_dim=16, max_seq=L, store_policy="uniform8")
        kvh = KVCacheSim(n_heads=4, head_dim=16, max_seq=L, store_policy="hatq")

        for t in range(L):
            x_t = ids[:, :t+1]
            s_q = s_t[:, t-1] if t > 0 else torch.tensor([[0.0]], device=ids.device)
            s_q = s_q.view(1, 1)
            pos = torch.arange(t+1, device=ids.device).unsqueeze(0)
            x_embed = model.embed(ids[:, :t+1]) + model.pos_embed(pos)
            h = model.blocks[0].ln1(x_embed)
            xf = h[:, -1:, :]
            Q = model.blocks[0].attn.q_proj.forward_all_4bit(xf.view(1, -1)).view(1, 1, -1)
            K = model.blocks[0].attn.k_proj.forward_all_4bit(xf.view(1, -1)).view(1, 1, -1)
            V = model.blocks[0].attn.v_proj.forward_all_4bit(xf.view(1, -1)).view(1, 1, -1)
            Hh, Dd = 4, 16
            K = K.view(1, Hh, 1, Dd); V = V.view(1, Hh, 1, Dd)
            kv4.append_token(K, V, s_query=torch.zeros_like(s_q))
            kv8.append_token(K, V, s_query=torch.full_like(s_q, 3))
            kvh.append_token(K, V, s_query=s_q)

        mem_uniform4.append(kv4.memory_bytes(H=4, D=16) / (1024**2))
        mem_uniform8.append(kv8.memory_bytes(H=4, D=16) / (1024**2))
        mem_hatq.append(kvh.memory_bytes(H=4, D=16) / (1024**2))
        print(f"Context {L}: KV mem (MB) uniform4={mem_uniform4[-1]:.3f}, uniform8={mem_uniform8[-1]:.3f}, HAT-Q={mem_hatq[-1]:.3f}")

    if images_dir is not None:
        ensure_dir(images_dir)
        plt.figure(figsize=(5,3))
        plt.plot(ctx_lengths, mem_uniform4, label="Uniform 4b KV")
        plt.plot(ctx_lengths, mem_uniform8, label="Uniform 8b KV")
        plt.plot(ctx_lengths, mem_hatq, label="HAT-Q KV")
        plt.xlabel("Context length (tokens)")
        plt.ylabel("KV memory (MB)")
        plt.title("KV memory vs context length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "memory_vs_context_hatq.pdf"), bbox_inches="tight")
        plt.close()

    return {"ctx": ctx_lengths, "mem4": mem_uniform4, "mem8": mem_uniform8, "mem_hatq": mem_hatq}


# -------------------------------
# Experiment C: Predictor validity, calibration curves, robustness
# -------------------------------

def drop_tokens(input_ids: torch.Tensor, drop_prob=0.1, pad_token_id=0):
    mask = torch.rand_like(input_ids.float()) > drop_prob
    out = input_ids.clone()
    out[~mask] = pad_token_id
    return out


def inject_punctuation(input_ids: torch.Tensor, punct_token_id: int = 127, every_k: int = 10, repeat=2):
    B, T = input_ids.shape
    out = []
    for b in range(B):
        seq = input_ids[b].tolist()
        aug = []
        for i, tok in enumerate(seq):
            aug.append(tok)
            if i > 0 and i % every_k == 0:
                aug.extend([punct_token_id]*repeat)
        out.append(torch.tensor(aug[:T], dtype=torch.long))
    return torch.stack(out, dim=0)


def run_experiment_C(device: str = "cpu", images_dir: Optional[str] = None):
    print("\n" + "="*80)
    print("Experiment C: Predictor validity, calibration, and robustness")
    print("="*80)
    set_seed(99)
    vocab_size = 128
    model, _, _ = train_toy_model(
        device=device,
        images_dir=None,
        cfg={"vocab_size": vocab_size, "epochs": 2, "n_train": 256, "n_valid": 64},
        verbose=False,
    )

    model.prepare_palettes()

    calib_ds = SyntheticSeqDataset(n_samples=64, seq_len=64, mix=True, seed=88)
    eval_ds = SyntheticSeqDataset(n_samples=64, seq_len=64, mix=True, seed=89)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)

    probe, probe_metrics = train_probe_on_calib(model, calib_loader, device, images_dir)

    th = Thresholds(init_vals=(-0.3, 0.0, 0.5)).to(device)
    calibrate_thresholds(model, probe, th, calib_loader, device, budget_B=1.2, steps=30, lr=5e-3)

    budgets = [0.6, 0.9, 1.2, 1.8, 2.4]
    accs = []
    Es = []
    for Bgt in budgets:
        calibrate_thresholds(model, probe, th, calib_loader, device, budget_B=Bgt, steps=10, lr=3e-3)
        ce_losses, accs_eval, Es_eval = [], [], []
        with torch.no_grad():
            for batch in eval_loader:
                ids = batch["input_ids"].to(device)
                _, h_fp = model(ids, method="fp", return_hidden=True)
                s_hat = probe(h_fp[:, :-1, :])
                s_t = th(s_hat)
                eight_mask = (s_t.reshape(-1) >= 2)
                logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
                loss = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
                pred = logits_hatq[:, :-1, :].argmax(dim=-1)
                acc = (pred == ids[:, 1:]).float().mean().item()
                ce_losses.append(loss.item()); accs_eval.append(acc); Es_eval.append(s_t.float().mean().item())
        accs.append(float(np.mean(accs_eval))); Es.append(float(np.mean(Es_eval)))
        print(f"Budget {Bgt:.2f}: acc={accs[-1]:.4f}, E[s]={Es[-1]:.3f}")

    if images_dir is not None:
        ensure_dir(images_dir)
        plt.figure(figsize=(5,3))
        plt.plot(Es, accs, marker="o")
        plt.xlabel("E[s_t]")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs precision budget (HAT-Q)")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "accuracy_vs_budget_hatq.pdf"), bbox_inches="tight")
        plt.close()

    # Robustness
    def eval_condition(modifier_name: str):
        ce_losses, accs_eval, Es_eval = [], [], []
        with torch.no_grad():
            for batch in eval_loader:
                ids = batch["input_ids"].to(device)
                if modifier_name == "dropout":
                    ids = drop_tokens(ids, drop_prob=0.1, pad_token_id=0)
                elif modifier_name == "punct":
                    ids = inject_punctuation(ids, punct_token_id=vocab_size-1, every_k=8, repeat=2)
                _, h_fp = model(ids, method="fp", return_hidden=True)
                s_hat = probe(h_fp[:, :-1, :])
                s_t = th(s_hat)
                eight_mask = (s_t.reshape(-1) >= 2)
                logits_hatq = model(ids, method="hatq", eight_mask=eight_mask)
                loss = F.cross_entropy(logits_hatq[:, :-1, :].reshape(-1, logits_hatq.size(-1)), ids[:, 1:].reshape(-1))
                pred = logits_hatq[:, :-1, :].argmax(dim=-1)
                acc = (pred == ids[:, 1:]).float().mean().item()
                ce_losses.append(loss.item()); accs_eval.append(acc); Es_eval.append(s_t.float().mean().item())
        return float(np.mean(ce_losses)), float(np.mean(accs_eval)), float(np.mean(Es_eval))

    base_loss, base_acc, base_Es = eval_condition("clean")
    drop_loss, drop_acc, drop_Es = eval_condition("dropout")
    punct_loss, punct_acc, punct_Es = eval_condition("punct")

    print(f"Robustness (clean): loss={base_loss:.4f} acc={base_acc:.4f} E[s]={base_Es:.3f}")
    print(f"Robustness (dropout 10%): loss={drop_loss:.4f} acc={drop_acc:.4f} E[s]={drop_Es:.3f} (Δacc={drop_acc-base_acc:.4f})")
    print(f"Robustness (punct inj): loss={punct_loss:.4f} acc={punct_acc:.4f} E[s]={punct_Es:.3f} (Δacc={punct_acc-base_acc:.4f})")

    if images_dir is not None:
        ensure_dir(images_dir)
        labels = ["clean", "dropout", "punct"]
        vals = [base_acc, drop_acc, punct_acc]
        plt.figure(figsize=(5,3))
        sns.barplot(x=labels, y=vals, color="lightgreen", edgecolor="black")
        plt.ylabel("Accuracy")
        plt.title("Robustness: HAT-Q accuracy under perturbations")
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "robustness_accuracy_hatq.pdf"), bbox_inches="tight")
        plt.close()

    return {
        "auc": probe_metrics.get("auc", float("nan")),
        "pr": probe_metrics.get("pr", float("nan")),
        "accuracy_vs_budget": {"budgets": budgets, "acc": accs, "E_s": Es},
        "robustness": {
            "clean": {"loss": base_loss, "acc": base_acc, "E_s": base_Es},
            "dropout": {"loss": drop_loss, "acc": drop_acc, "E_s": drop_Es},
            "punct": {"loss": punct_loss, "acc": punct_acc, "E_s": punct_Es},
        }
    }


# -------------------------------
# Quick functional test runner (completes quickly)
# -------------------------------

def test_run(device: str = "cpu", images_dir: Optional[str] = None):
    unit_tests()
    A = run_experiment_A(device=device, images_dir=images_dir)
    B = run_experiment_B(device=device, images_dir=images_dir)
    C = run_experiment_C(device=device, images_dir=images_dir)
    print("\n" + "="*80)
    print("Summary of key outputs")
    print("="*80)
    print(json.dumps({
        "ExpA": {k: {m: float(v[m]) if isinstance(v, dict) and isinstance(v[m], (int, float)) else v for m in v} for k,v in A.items()},
        "ExpB": B,
        "ExpC": C,
    }, indent=2))
    if images_dir is not None:
        files = [
            "training_loss_hatq.pdf",
            "accuracy_hatq.pdf",
            "predictor_roc.pdf",
            "throughput_hatq.pdf",
            "memory_vs_context_hatq.pdf",
            "accuracy_vs_budget_hatq.pdf",
            "robustness_accuracy_hatq.pdf",
        ]
        print("Saved figures (exists flag):")
        for f in files:
            path = os.path.join(images_dir, f)
            print(" -", path, os.path.exists(path))
