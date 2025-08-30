import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import roc_auc_score, confusion_matrix
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    from .preprocess import get_device, now_ms, measure_memory_bytes
    from .train import TinyHATQTransformer, GlobalBudgetRNN, TokenMaskBlock, BudgetHead
except ImportError:  # fallback when running as scripts
    from preprocess import get_device, now_ms, measure_memory_bytes
    from train import TinyHATQTransformer, GlobalBudgetRNN, TokenMaskBlock, BudgetHead


class BitStats:
    def __init__(self):
        self.total_planes = 0
        self.total_tokens = 0
    def update(self, token_bits: torch.Tensor):
        if token_bits is None:
            return
        self.total_planes += int(token_bits.sum().item())
        self.total_tokens += int(token_bits.numel())
    def avg_bits(self) -> float:
        if self.total_tokens == 0:
            return float('nan')
        return self.total_planes / self.total_tokens
    def reset(self):
        self.total_planes = 0
        self.total_tokens = 0


@torch.no_grad()
def evaluate_ppl(model: TinyHATQTransformer, data: List[Dict[str, torch.Tensor]],
                 controller_modules: Optional[Tuple[GlobalBudgetRNN, TokenMaskBlock, BudgetHead]] = None,
                 ablation: str = "hatq") -> float:
    model.eval()
    gb_rnn = mask_block = budget_head = None
    if controller_modules is not None:
        gb_rnn, mask_block, budget_head = controller_modules
        gb_rnn.eval(); mask_block.eval(); budget_head.eval()

    total_nll = 0.0
    count = 0
    for batch in data:
        input_ids = batch["input_ids"]
        B, T = input_ids.shape
        if ablation == "hatq" and controller_modules is not None:
            # Clear any stale token-bit assignments before the first pass
            model._force_bits(4)
            kv_stats = torch.zeros(B, 8, model.config.num_hidden_layers, device=input_ids.device)
            logits, hidden = model(input_ids, return_hidden=True)
            gbits = gb_rnn(kv_stats)
            token_logits = mask_block(hidden)
            extra_mask = budget_head(gbits, token_logits, enable_mask=True)
            bits_btl = torch.clamp(3 + extra_mask.unsqueeze(-1).expand(-1, -1, model.config.num_hidden_layers), 3, 8)
            model._set_token_bits(bits_btl)
            logits = model(input_ids)
        elif ablation == "global_only" and controller_modules is not None:
            kv_stats = torch.zeros(B, 8, model.config.num_hidden_layers, device=input_ids.device)
            gbits = gb_rnn(kv_stats)
            bits_btl = torch.clamp(gbits.unsqueeze(1).expand(-1, T, -1), 3, 8)
            model._set_token_bits(bits_btl)
            logits = model(input_ids)
        elif ablation == "controller_off":
            model._force_bits(4)
            logits = model(input_ids)
        else:
            model._force_bits(4)
            logits = model(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='mean')
        total_nll += float(nll.item())
        count += 1
    ppl = float(np.exp(total_nll / max(1, count)))
    return ppl


@torch.no_grad()
def decode_throughput(model: TinyHATQTransformer,
                      prompts: torch.Tensor,
                      controller_modules: Optional[Tuple[GlobalBudgetRNN, TokenMaskBlock, BudgetHead]] = None,
                      max_new_tokens: int = 64,
                      ablation: str = "hatq") -> Tuple[float, int, float]:
    model.eval()
    gb_rnn = mask_block = budget_head = None
    if controller_modules is not None:
        gb_rnn, mask_block, budget_head = controller_modules
        gb_rnn.eval(); mask_block.eval(); budget_head.eval()

    B, T = prompts.shape
    ids = prompts.clone()
    bit_stats = BitStats()

    def step(ids_):
        B_, T_ = ids_.shape
        # Clear any stale per-token bit assignments before the first pass
        model._force_bits(4)
        if ablation == 'hatq' and controller_modules is not None:
            kv_stats = torch.zeros(B_, 8, model.config.num_hidden_layers, device=ids_.device)
            logits, hidden = model(ids_, return_hidden=True)
            gbits = gb_rnn(kv_stats)
            token_logits = mask_block(hidden)
            extra_mask = budget_head(gbits, token_logits, enable_mask=True)
            bits_btl = torch.clamp(3 + extra_mask.unsqueeze(-1).expand(-1, -1, model.config.num_hidden_layers), 3, 8)
            model._set_token_bits(bits_btl)
            bit_stats.update(bits_btl.reshape(-1))
            logits = model(ids_)
        elif ablation == 'global_only' and controller_modules is not None:
            kv_stats = torch.zeros(B_, 8, model.config.num_hidden_layers, device=ids_.device)
            gbits = gb_rnn(kv_stats)
            bits_btl = torch.clamp(gbits.unsqueeze(1).expand(-1, T_, -1), 3, 8)
            model._set_token_bits(bits_btl)
            bit_stats.update(bits_btl.reshape(-1))
            logits = model(ids_)
        elif ablation == 'controller_off':
            model._force_bits(4)
            logits = model(ids_)
            bit_stats.update(torch.full((B_*T_*model.config.num_hidden_layers,), 4, device=ids_.device))
        else:
            model._force_bits(4)
            logits = model(ids_)
            bit_stats.update(torch.full((B_*T_*model.config.num_hidden_layers,), 4, device=ids_.device))
        return logits

    device = ids.device
    for _ in range(4):
        _ = step(ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max_new_tokens):
            logits = step(ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        mem_bytes = torch.cuda.max_memory_allocated()
    else:
        t0 = now_ms()
        for _ in range(max_new_tokens):
            logits = step(ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        elapsed_ms = now_ms() - t0
        mem_bytes = measure_memory_bytes(device)

    total_tokens = B * max_new_tokens
    tps = total_tokens / (elapsed_ms / 1000.0)
    return float(tps), int(mem_bytes), float(bit_stats.avg_bits())


# ------------------------------
# Systems micro-benchmarks
# ------------------------------
@torch.no_grad()
def microbench_gemm(M: int = 128, K: int = 512, N: int = 512,
                    mix: Dict[int, float] = {3:0.7, 4:0.25, 6:0.05},
                    mode: str = 'hatq',
                    device: Optional[torch.device] = None,
                    iters: int = 50) -> float:
    device = device if device is not None else get_device()
    planes = 8
    dtype = torch.float32 if device.type == 'cpu' else torch.float16
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = (torch.randint(0, 2, (planes, N, K), device=device, dtype=torch.int8) * 2 - 1).to(dtype=dtype)
    scales = torch.ones(planes, N, device=device, dtype=dtype)

    keys = sorted(list(mix.keys()))
    probs = torch.tensor([mix[k] for k in keys], device=device, dtype=torch.float32)
    probs = probs / probs.sum()
    choice_idx = torch.multinomial(probs, num_samples=M, replacement=True)
    row_bits = torch.tensor([keys[i] for i in choice_idx.tolist()], device=device, dtype=torch.int32)

    def run_once():
        if mode == 'static4':
            k = 4
            acc = None
            for p in range(planes - k, planes):
                contrib = A @ B[p].t()
                acc = contrib * scales[p] if acc is None else acc + contrib * scales[p]
            C = acc
        elif mode == 'hatq':
            uniq = torch.unique(row_bits)
            outs = []
            for k in uniq.tolist():
                idx = (row_bits == k).nonzero(as_tuple=False).squeeze(-1)
                Ak = A.index_select(0, idx)
                acc = None
                for p in range(planes - k, planes):
                    contrib = Ak @ B[p].t()
                    acc = contrib * scales[p] if acc is None else acc + contrib * scales[p]
                outs.append((idx, acc))
            C = torch.empty(M, N, device=device, dtype=outs[0][1].dtype)
            for idx, out in outs:
                C.index_copy_(0, idx, out)
        elif mode == 'naive':
            C = torch.empty(M, N, device=device, dtype=dtype)
            for i in range(M):
                k = int(row_bits[i].item())
                acc = None
                for p in range(planes - k, planes):
                    contrib = A[i:i+1] @ B[p].t()
                    acc = contrib * scales[p] if acc is None else acc + contrib * scales[p]
                C[i:i+1] = acc
        elif mode == 'fetch_all_then_mask':
            acc_all = None
            for p in range(0, planes):
                contrib = A @ B[p].t()
                acc_all = contrib * scales[p] if acc_all is None else acc_all + contrib * scales[p]
            C = acc_all
        else:
            raise ValueError("Unknown mode")
        return C

    for _ in range(5):
        _ = run_once()

    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = run_once()
        end.record(); end.synchronize()
        ms = start.elapsed_time(end) / iters
    else:
        t0 = now_ms()
        for _ in range(iters):
            _ = run_once()
        ms = (now_ms() - t0) / iters
    return float(ms)


def run_experiment2_microbench(images_dir: str, device: Optional[torch.device] = None):
    os.makedirs(images_dir, exist_ok=True)
    device = device if device is not None else get_device()
    mixes = {
        'mixA': {3:0.7, 4:0.25, 6:0.05},
        'mixB': {3:0.85, 4:0.10, 7:0.05},
        'mixC': {3:0.10, 4:0.60, 6:0.30},
    }
    Ms = [32, 128, 512]
    K = 512
    N = 512
    results = {}
    for mix_name, mix in mixes.items():
        print(f"[Experiment 2] Benchmarking {mix_name} on device={device} ...")
        data = { 'static4': [], 'hatq': [], 'naive': [], 'fetch_all_then_mask': [] }
        for M in Ms:
            for mode in data.keys():
                ms = microbench_gemm(M=M, K=K, N=N, mix=mix, mode=mode, device=device, iters=30)
                data[mode].append(ms)
                print(f"  M={M:>4} mode={mode:<20} avg_time={ms:.3f} ms")
        results[mix_name] = data
        plt.figure(figsize=(6,4))
        for mode, ys in data.items():
            plt.plot(Ms, ys, marker='o', label=mode)
        plt.xlabel('M (tokens)')
        plt.ylabel('Kernel time (ms, avg)')
        plt.title(f'Systems micro-bench: {mix_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f"kernel_time_{mix_name}.pdf"), bbox_inches="tight")
        plt.close()
    print("[Experiment 2] Completed. Saved per-mix kernel_time_*.pdf")
    return results


# ------------------------------
# Controller diagnostics
# ------------------------------
@torch.no_grad()
def compute_offline_labels(model: TinyHATQTransformer, data: List[Dict[str, torch.Tensor]], tau: float = 0.01) -> torch.Tensor:
    labels = []
    for batch in data:
        x = batch["input_ids"]
        model._force_bits(3)
        logits3 = model(x)
        nll3 = F.cross_entropy(logits3[:, :-1].reshape(-1, logits3.size(-1)), x[:, 1:].reshape(-1), reduction='none')
        nll3 = nll3.view(x.size(0), -1)
        model._force_bits(5)
        logits5 = model(x)
        nll5 = F.cross_entropy(logits5[:, :-1].reshape(-1, logits5.size(-1)), x[:, 1:].reshape(-1), reduction='none')
        nll5 = nll5.view(x.size(0), -1)
        dnll = (nll5 - nll3)
        y = (dnll < -tau).int()
        labels.append(y.cpu())
    return torch.cat(labels, dim=0)


def heuristic_policy(last_hidden: torch.Tensor, logits: torch.Tensor, mode: str = 'norm', rate: float = 0.15) -> torch.Tensor:
    B, T, H = last_hidden.shape
    if mode == 'norm':
        s = last_hidden.norm(dim=-1)
    elif mode == 'entropy':
        probs = torch.softmax(logits.detach(), dim=-1)
        s = -(probs * (probs + 1e-9).log()).sum(-1)
    elif mode == 'random':
        s = torch.rand(B, T, device=last_hidden.device)
    else:
        s = last_hidden.norm(dim=-1)
    k = max(1, int(rate * T))
    thresh = torch.kthvalue(s, max(1, T - k), dim=1).values.unsqueeze(1)
    mask = (s >= thresh).int()
    return mask


def run_experiment3_controller(model: TinyHATQTransformer, data_small: List[Dict[str, torch.Tensor]],
                               controller_modules: Tuple[GlobalBudgetRNN, TokenMaskBlock, BudgetHead],
                               images_dir: str,
                               labels_offline: Optional[torch.Tensor] = None):
    os.makedirs(images_dir, exist_ok=True)
    gb_rnn, mask_block, budget_head = controller_modules
    model.eval(); gb_rnn.eval(); mask_block.eval(); budget_head.eval()

    batch = data_small[0]
    x = batch["input_ids"]
    # Clear any stale token-bit assignments before the first pass
    model._force_bits(4)
    logits, hidden = model(x, return_hidden=True)

    kv_stats = torch.zeros(x.size(0), 8, model.config.num_hidden_layers, device=x.device)
    gbits = gb_rnn(kv_stats)
    token_logits = mask_block(hidden)
    ctrl_mask = budget_head(gbits, token_logits, enable_mask=True)

    if labels_offline is None:
        labels_offline = compute_offline_labels(model, data_small)
    # offline labels are for next-token prediction, so they have length T-1
    y = labels_offline[:x.size(0), :x.size(1)-1]

    # Align controller scores and predictions to T-1 as well
    y_true = y.reshape(-1).cpu().numpy()
    y_score = torch.sigmoid(token_logits[:, :-1]).reshape(-1).detach().cpu().numpy()
    y_pred = (ctrl_mask[:, :-1].reshape(-1) > 0.5).int().cpu().numpy()

    auc = None
    if _HAVE_SKLEARN:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    print(f"[Experiment 3] Controller ROC-AUC vs offline labels: {auc if auc is not None else 'N/A (sklearn not available)'}")

    if _HAVE_SKLEARN and auc is not None:
        thresholds = np.linspace(0, 1, 50)
        tpr = []
        fpr = []
        P = y_true.sum(); N = len(y_true) - P
        for th in thresholds:
            yp = (y_score >= th).astype(np.int32)
            tp = int(((yp == 1) & (y_true == 1)).sum())
            fp = int(((yp == 1) & (y_true == 0)).sum())
            tpr.append(tp / max(1, P))
            fpr.append(fp / max(1, N))
        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr, label=f'Controller (AUC={auc:.2f})')
        plt.plot([0,1],[0,1], 'k--', label='random')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Controller ROC')
        plt.legend()
        plt.savefig(os.path.join(images_dir, "controller_roc_hatq.pdf"), bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(6,4))
        sns.kdeplot(y_score[y_true==1], label='positives', fill=True)
        sns.kdeplot(y_score[y_true==0], label='negatives', fill=True)
        plt.title('Controller score distributions')
        plt.legend()
        plt.savefig(os.path.join(images_dir, "controller_roc_hatq.pdf"), bbox_inches="tight")
        plt.close()

    if x.device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = now_ms(); _ = gb_rnn(kv_stats); t1 = now_ms()
    t2 = now_ms(); _ = mask_block(hidden); t3 = now_ms()
    ov_ms = (t1 - t0) + (t3 - t2)
    print(f"[Experiment 3] Controller overhead ~ {ov_ms:.3f} ms (GRU + token block)")
    plt.figure(figsize=(4,3))
    plt.bar(["global", "token"], [t1 - t0, t3 - t2])
    plt.ylabel('ms')
    plt.title('Controller overhead components')
    plt.savefig(os.path.join(images_dir, "controller_overhead.pdf"), bbox_inches="tight")
    plt.close()

    return auc


@torch.no_grad()
def evaluate_classification(model: TinyHATQTransformer,
                            X: torch.Tensor,
                            y: torch.Tensor,
                            controller_modules: Optional[Tuple[GlobalBudgetRNN, TokenMaskBlock, BudgetHead]] = None,
                            ablation: str = 'hatq') -> Tuple[float, Optional[np.ndarray]]:
    """Evaluate a simple binary classification using LM perplexity as a score.
    - Compute mean token NLL per sequence; predict positive if score above median.
    - Supports the same ablation modes as other evaluators.
    Returns: (accuracy, confusion_matrix_or_None)
    """
    model.eval()
    device = X.device
    B, T = X.shape

    gb_rnn = mask_block = budget_head = None
    if controller_modules is not None:
        gb_rnn, mask_block, budget_head = controller_modules
        gb_rnn.eval(); mask_block.eval(); budget_head.eval()

    # Prepare logits according to ablation
    if ablation == 'hatq' and controller_modules is not None:
        model._force_bits(4)
        kv_stats = torch.zeros(B, 8, model.config.num_hidden_layers, device=device)
        logits_first, hidden = model(X, return_hidden=True)
        gbits = gb_rnn(kv_stats)
        token_logits = mask_block(hidden)
        extra_mask = budget_head(gbits, token_logits, enable_mask=True)
        bits_btl = torch.clamp(3 + extra_mask.unsqueeze(-1).expand(-1, -1, model.config.num_hidden_layers), 3, 8)
        model._set_token_bits(bits_btl)
        logits = model(X)
    elif ablation == 'global_only' and controller_modules is not None:
        kv_stats = torch.zeros(B, 8, model.config.num_hidden_layers, device=device)
        gbits = gb_rnn(kv_stats)
        bits_btl = torch.clamp(gbits.unsqueeze(1).expand(-1, T, -1), 3, 8)
        model._set_token_bits(bits_btl)
        logits = model(X)
    else:  # controller_off or fallback
        model._force_bits(4)
        logits = model(X)

    # Per-sequence mean NLL
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = X[:, 1:].contiguous()
    token_nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
    token_nll = token_nll.view(B, -1)
    seq_nll = token_nll.mean(dim=1)
    score = -seq_nll  # higher is "better"

    # Threshold by median
    thresh = score.median()
    y_pred = (score >= thresh).long()

    y_true = y.long().detach().cpu().numpy()
    yhat = y_pred.detach().cpu().numpy()
    acc = float((yhat == y_true).mean())
    cm = None
    if _HAVE_SKLEARN:
        try:
            cm = confusion_matrix(y_true, yhat)
        except Exception:
            cm = None
    return acc, cm
