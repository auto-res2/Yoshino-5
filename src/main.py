import os
import math
from dataclasses import dataclass
from typing import Dict

import torch
import yaml

# Plotting setup for high-quality PDFs
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import CharTokenizer, MathReasoningDataset, collate_batch
from .train import TinyTransformer, TrainConfig, train_one
from .evaluate import evaluate_em, save_barplot_pdf, save_scatter_pdf, spearman_corr


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def load_config(path: str = 'config/experiment.yaml') -> Dict:
    if not os.path.exists(path):
        # default minimal config
        return {
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'images_dir': '.research/iteration1/images'
        }
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def experiment1_core(images_dir: str, seed: int = 42, device: str = 'cpu'):
    print('===== Experiment 1 — Core efficacy under sparsity =====')
    tok = CharTokenizer()
    train_ds = MathReasoningDataset(n_samples=256, tokenizer=tok, max_len=256, long_context=False, seed=seed)
    valid_ds = MathReasoningDataset(n_samples=64, tokenizer=tok, max_len=256, long_context=False, seed=seed+1)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_batch)

    # Dense baseline (LoRA-like, no MAGMA)
    model_dense = TinyTransformer(vocab_size=len(tok.stoi), d_model=64, n_layers=2, n_heads=4,
                                  dim_ff=128, block_size=16, r=8, use_dora=False)
    cfg_dense = TrainConfig(use_dora=False, use_magma=False, random_sparse=False, rho=0.0,
                            steps=60, batch_size=8, lr=3e-3, seed=seed, device=device, images_dir=images_dir)
    out_dense = train_one(model_dense, train_loader, cfg_dense, valid_ds, tok, tag='dense')

    # Random sparse baseline
    model_rand = TinyTransformer(vocab_size=len(tok.stoi), d_model=64, n_layers=2, n_heads=4,
                                 dim_ff=128, block_size=16, r=8, use_dora=False)
    cfg_rand = TrainConfig(use_dora=False, use_magma=True, random_sparse=True, rho=0.9,
                           steps=60, batch_size=8, lr=3e-3, seed=seed, device=device,
                           mask_update_every=20, warmup_steps=40, images_dir=images_dir)
    out_rand = train_one(model_rand, train_loader, cfg_rand, valid_ds, tok, tag='random')

    # MAGMA (DoRA + magnitude-guided)
    model_magma = TinyTransformer(vocab_size=len(tok.stoi), d_model=64, n_layers=2, n_heads=4,
                                  dim_ff=128, block_size=16, r=8, use_dora=True)
    cfg_magma = TrainConfig(use_dora=True, use_magma=True, random_sparse=False, rho=0.9,
                            steps=60, batch_size=8, lr=3e-3, seed=seed, device=device,
                            mask_update_every=20, warmup_steps=40, images_dir=images_dir)
    out_magma = train_one(model_magma, train_loader, cfg_magma, valid_ds, tok, tag='magma')

    # Plots
    names = ['dense', 'random', 'magma']
    ems = [out_dense['em'] * 100, out_rand['em'] * 100, out_magma['em'] * 100]
    save_barplot_pdf(names, ems, 'Exact Match (%)', 'EM comparison (Exp1)', os.path.join(images_dir, 'accuracy_magma_vs_random.pdf'))

    thr = [out_dense['throughput']['throughput_toks_per_s'], out_rand['throughput']['throughput_toks_per_s'], out_magma['throughput']['throughput_toks_per_s']]
    save_barplot_pdf(names, thr, 'Tokens/sec', 'Throughput (Exp1)', os.path.join(images_dir, 'throughput_magma_vs_dense.pdf'))

    if out_magma['M_lh'] is not None:
        keep_rate = float(out_magma['M_lh'].float().mean().item())
        print(f"[Mask] MAGMA keep-rate ~ {keep_rate*100:.2f}% with rho target={cfg_magma.rho*100:.1f}% drop")

    return {
        'tok': tok,
        'valid_ds': valid_ds,
        'models': {'dense': model_dense, 'random': model_rand, 'magma': model_magma},
        'artifacts': {'dense': out_dense, 'random': out_rand, 'magma': out_magma}
    }


def extrapolate_mask(M8: torch.Tensor, S8: int, S32: int, block_size: int = 16, mode: str = 'tile') -> torch.Tensor:
    L, H, KB8 = M8.shape
    KB32 = (S32 + block_size - 1) // block_size
    if KB8 == 0:
        return torch.ones(L, H, KB32, dtype=torch.bool)
    if mode == 'tile':
        reps = math.ceil(KB32 / KB8)
        M = M8.repeat(1, 1, reps)[..., :KB32]
    elif mode == 'uniform':
        keep_rate = M8.float().mean(dim=-1, keepdim=True)
        M = (torch.rand(L, H, KB32) < keep_rate).bool()
    else:
        reps = math.ceil(KB32 / KB8)
        M = M8.repeat(1, 1, reps)[..., :KB32]
    for l in range(L):
        for h in range(H):
            if not M[l, h].any():
                M[l, h, 0] = True
    return M


def experiment2_long_context(core: Dict, images_dir: str, seed: int = 42, device: str = 'cpu'):
    print('===== Experiment 2 — Long-context robustness =====')
    tok = core['tok']
    long_ds = MathReasoningDataset(n_samples=48, tokenizer=tok, max_len=512, long_context=True, target_chars=1500, seed=seed+123)

    model_magma: TinyTransformer = core['models']['magma']
    model_magma.eval()
    M8 = model_magma.masks.detach().cpu() if model_magma.masks is not None else None

    em_short, _, _ = evaluate_em(model_magma, core['valid_ds'], tok, device=device)
    print(f"[Eval] MAGMA EM short={em_short*100:.2f}%")

    model_long = TinyTransformer(vocab_size=len(tok.stoi), d_model=64, n_layers=2, n_heads=4,
                                 dim_ff=128, block_size=16, r=8, use_dora=True)
    model_long.load_state_dict(model_magma.state_dict())
    model_long.to(device)
    if M8 is not None:
        M32 = extrapolate_mask(M8, S8=256, S32=512, block_size=16, mode='tile')
        model_long.set_masks(M32.to(device))
    else:
        model_long.set_masks(None)

    em_long, _, _ = evaluate_em(model_long, long_ds, tok, device=device)
    print(f"[Eval] MAGMA EM long={em_long*100:.2f}% on {len(long_ds)} samples")

    save_barplot_pdf(['short','long'], [em_short*100, em_long*100], 'Exact Match (%)', 'Long-context EM (MAGMA)', os.path.join(images_dir, 'accuracy_long_context_magma.pdf'))
    return {'em_short': em_short, 'em_long': em_long}


def tag_token(ch: str) -> str:
    if ch.isdigit(): return 'digit'
    if ch in ['+', '-', '*', '/', '×', '÷', '=', '^']: return 'op'
    if ch in ['(', ')', '[', ']', '{', '}']: return 'bracket'
    if ch == '.': return 'dot'
    if ch in ['>', ':']: return 'step'
    if len(ch) == 1 and ch.isalpha(): return 'word'
    return 'word'


def block_math_density_from_text(text: str, block_size: int = 16):
    KB = (len(text) + block_size - 1) // block_size
    import numpy as np
    dens = np.zeros(KB, dtype=np.float32)
    for b in range(KB):
        seg = text[b*block_size:(b+1)*block_size]
        if len(seg) == 0:
            continue
        math_like = sum(tag_token(ch) in {'digit','op','bracket','dot','step'} for ch in seg)
        dens[b] = math_like / len(seg)
    return dens


def experiment3_faithfulness(core: Dict, images_dir: str, seed: int = 42, device: str = 'cpu'):
    print('===== Experiment 3 — Token-importance faithfulness =====')
    tok = core['tok']
    model_magma: TinyTransformer = core['models']['magma']
    art_magma = core['artifacts']['magma']
    model_rand: TinyTransformer = core['models']['random']

    M = art_magma['M_lh']  # [L,H,KB]
    if M is None:
        print('[Warn] MAGMA masks missing; skipping Experiment 3.')
        return {}
    keep_prob = M.float().mean(dim=(0,1)).numpy()

    eval_ds = MathReasoningDataset(n_samples=64, tokenizer=tok, max_len=256, long_context=False, seed=seed+7)

    # correlation per sample, then report mean
    import numpy as np
    rhos = []
    for ex in eval_ds.samples:
        text = tok.decode(ex['input_ids'].tolist())
        dens = block_math_density_from_text(text, block_size=16)
        KB = min(len(dens), len(keep_prob))
        rhos.append(spearman_corr(keep_prob[:KB], dens[:KB]))
    mean_rho = float(np.mean(rhos)); std_rho = float(np.std(rhos))
    print(f"[Corr] Spearman(mask keep prob vs math-density) = {mean_rho:.3f} ± {std_rho:.3f}")

    # scatter for one example
    ex = eval_ds.samples[0]
    text0 = tok.decode(ex['input_ids'].tolist())
    dens0 = block_math_density_from_text(text0, block_size=16)
    KB0 = min(len(dens0), len(keep_prob))
    save_scatter_pdf(dens0[:KB0], keep_prob[:KB0], 'Math-density per block', 'Keep probability', 'Mask-content correlation (MAGMA)', os.path.join(images_dir, 'correlation_mask_math_tokens_magma.pdf'))

    # Causal ablations
    T_scores = art_magma['T_scores']  # [L,H,KB]
    if T_scores is None:
        print('[Warn] Missing T_scores; skipping ablations.')
        return {'rho_mean': mean_rho, 'rho_std': std_rho}
    L,H,KBmax = T_scores.shape

    def modify_masks(M_lh: torch.Tensor, T: torch.Tensor, mode: str = 'drop_kept_10') -> torch.Tensor:
        M2 = M_lh.clone()
        for l in range(L):
            for h in range(H):
                scores = T[l, h]
                kept = torch.nonzero(M_lh[l, h], as_tuple=False).squeeze(-1)
                dropped = torch.nonzero(~M_lh[l, h], as_tuple=False).squeeze(-1)
                if mode == 'drop_kept_10' and kept.numel() > 0:
                    k = max(1, int(0.1 * kept.numel()))
                    idx = kept[torch.topk(scores[kept], k, largest=False).indices]
                    M2[l, h, idx] = False
                elif mode == 'drop_dropped_10' and dropped.numel() > 0:
                    k = max(1, int(0.1 * dropped.numel()))
                    idx = dropped[torch.topk(scores[dropped], k, largest=True).indices]
                    M2[l, h, idx] = True
        return M2

    base_em, _, _ = evaluate_em(model_magma, eval_ds, tok, device=device)
    M_base = model_magma.masks.detach().cpu()

    M_drop_kept = modify_masks(M_base, T_scores, mode='drop_kept_10')
    model_magma.set_masks(M_drop_kept.to(device))
    em_drop_kept, _, _ = evaluate_em(model_magma, eval_ds, tok, device=device)

    M_drop_dropped = modify_masks(M_base, T_scores, mode='drop_dropped_10')
    model_magma.set_masks(M_drop_dropped.to(device))
    em_drop_dropped, _, _ = evaluate_em(model_magma, eval_ds, tok, device=device)

    # Restore
    model_magma.set_masks(M_base.to(device))

    print(f"[Ablation] Base EM={base_em*100:.2f}% | drop-kept -> {em_drop_kept*100:.2f}% | drop-dropped -> {em_drop_dropped*100:.2f}%")
    save_barplot_pdf(['base','drop_kept','drop_dropped'], [base_em*100, em_drop_kept*100, em_drop_dropped*100], 'Exact Match (%)', 'Causal ablation (MAGMA)', os.path.join(images_dir, 'accuracy_ablation_magma.pdf'))

    # Adversarial number shuffling
    def shuffle_numbers(text: str, seed: int = 0) -> str:
        import random
        rnd = random.Random(seed)
        nums = []
        for i, ch in enumerate(text):
            if ch.isdigit():
                nums.append((i, ch))
        if len(nums) < 2:
            return text
        (i1, a), (i2, b) = rnd.sample(nums, 2)
        lst = list(text)
        lst[i1] = b; lst[i2] = a
        return ''.join(lst)

    adv_ds = MathReasoningDataset(n_samples=64, tokenizer=tok, max_len=256, long_context=False, seed=seed+9)
    for ex in adv_ds.samples:
        dec = tok.decode(ex['input_ids'].tolist())
        adv = shuffle_numbers(dec, seed=seed)
        ids = tok.encode(adv)
        ids = ids[:adv_ds.max_len] + [tok.pad_id] * max(0, adv_ds.max_len - len(ids))
        ex['input_ids'] = torch.tensor(ids[:-1], dtype=torch.long)
        ex['labels'] = torch.tensor(ids[1:], dtype=torch.long)

    em_magma_adv, _, _ = evaluate_em(model_magma, adv_ds, tok, device=device)
    em_rand_adv, _, _ = evaluate_em(model_rand, adv_ds, tok, device=device)
    print(f"[Adversary] EM under number-shuffle: MAGMA={em_magma_adv*100:.2f}% | Random={em_rand_adv*100:.2f}%")
    save_barplot_pdf(['MAGMA','Random'], [em_magma_adv*100, em_rand_adv*100], 'Exact Match (%)', 'Adversarial number shuffle', os.path.join(images_dir, 'accuracy_adversarial_magma_vs_random.pdf'))

    return {
        'rho_mean': mean_rho,
        'rho_std': std_rho,
        'em_base': base_em,
        'em_drop_kept': em_drop_kept,
        'em_drop_dropped': em_drop_dropped,
        'em_magma_adv': em_magma_adv,
        'em_rand_adv': em_rand_adv,
    }


def run_all():
    ensure_dirs()
    cfg = load_config()
    seed = int(cfg.get('seed', 42))
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    images_dir = cfg.get('images_dir', '.research/iteration1/images')

    core = experiment1_core(images_dir=images_dir, seed=seed, device=device)
    exp2 = experiment2_long_context(core, images_dir=images_dir, seed=seed, device=device)
    exp3 = experiment3_faithfulness(core, images_dir=images_dir, seed=seed, device=device)

    magma_em = core['artifacts']['magma']['em'] * 100
    random_em = core['artifacts']['random']['em'] * 100
    dense_em  = core['artifacts']['dense']['em'] * 100
    long_short = exp2.get('em_short', 0) * 100
    long_long  = exp2.get('em_long', 0) * 100

    # Summary plot
    save_barplot_pdf(['dense', 'random', 'magma', 'magma_short', 'magma_long'],
                     [dense_em, random_em, magma_em, long_short, long_long],
                     'Exact Match (%)', 'Summary Accuracies', os.path.join(images_dir, 'accuracy_summary_magma.pdf'))

    print('===== Finished all experiments =====')


if __name__ == '__main__':
    run_all()
