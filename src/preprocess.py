import os
import random
from typing import List, Optional, Dict

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self):
        base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digits = list("0123456789")
        ops = list("+-*/=^×÷()[]{}:\n .,>")
        extra = list(";!?_#$")
        vocab = ['<pad>', '<bos>', '<eos>'] + base_chars + digits + ops + extra
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi['<pad>']
        self.bos_id = self.stoi['<bos>']
        self.eos_id = self.stoi['<eos>']

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi[' ']))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars: List[str] = []
        for i in ids:
            ch = self.itos.get(int(i), ' ')
            if ch in ['<bos>', '<eos>', '<pad>']:
                continue
            chars.append(ch)
        return ''.join(chars)


def _apply(x, op, y):
    if op == '+': return x + y
    if op == '-': return x - y
    if op == '*': return x * y
    raise ValueError


def generate_math_problem(seed: Optional[int] = None):
    rnd = random.Random(seed) if seed is not None else random
    a = rnd.randint(1, 99)
    b = rnd.randint(1, 99)
    c = rnd.randint(0, 99)
    op1 = rnd.choice(['+', '-', '*'])
    op2 = rnd.choice(['+', '-'])
    res = _apply(_apply(a, op1, b), op2, c)
    problem = f"Problem: Compute {a} {op1} {b} {op2} {c}.\n"
    steps = (
        f"Step 1: Compute {a} {op1} {b} = {_apply(a, op1, b)}.\n"
        f"Step 2: Then {_apply(a, op1, b)} {op2} {c} = {res}.\n"
        f"Thus, the final result is shown below.\n"
    )
    answer = f"Answer: {res}\n"
    return problem, problem + steps + answer, res


def make_distractor_paragraph(seed: int = 0, repeats: int = 4) -> str:
    rnd = random.Random(seed)
    words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "lorem", "ipsum", "dolor", "sit", "amet"
    ]
    s = []
    for _ in range(repeats):
        sent = " ".join(rnd.choice(words) for _ in range(50))
        s.append(f"Note: {sent}.")
    return "\n".join(s) + "\n"


def synthesize_sample(long_context: bool = False, target_chars: int = 512, seed: Optional[int] = None):
    problem, full, ans = generate_math_problem(seed=seed)
    if not long_context:
        distractor = make_distractor_paragraph(seed=seed or 0, repeats=2)
        text = problem + distractor + full
        return problem, text, ans
    rnd = random.Random(seed or 0)
    text = problem
    while len(text) < target_chars:
        text += make_distractor_paragraph(seed=rnd.randint(0, 10**6), repeats=4)
        if rnd.random() < 0.4:
            text += f"Hint: Remember PEMDAS. {rnd.randint(10,99)} + {rnd.randint(1,9)} = {rnd.randint(11,108)}.\n"
    _, full, ans = generate_math_problem(seed=seed)
    text += full
    return problem, text, ans


class MathReasoningDataset(Dataset):
    def __init__(self, n_samples: int = 1000, tokenizer: Optional[CharTokenizer] = None,
                 max_len: int = 256, long_context: bool = False, target_chars: int = 2048,
                 seed: int = 0):
        self.tok = tokenizer or CharTokenizer()
        self.max_len = max_len
        self.samples: List[Dict] = []
        rnd = random.Random(seed)
        for _ in range(n_samples):
            prob, text, ans = synthesize_sample(long_context=long_context, target_chars=target_chars,
                                                seed=rnd.randint(0, 10**9))
            ids = self.tok.encode(text)
            ids = ids[:max_len]
            pad_len = max(0, max_len - len(ids))
            if pad_len > 0:
                ids = ids + [self.tok.pad_id] * pad_len
            inp = ids[:-1]
            tgt = ids[1:]
            self.samples.append({
                'input_ids': torch.tensor(inp, dtype=torch.long),
                'labels': torch.tensor(tgt, dtype=torch.long),
                'prompt': prob,
                'answer': str(ans)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch: List[Dict]):
    input_ids = torch.stack([b['input_ids'] for b in batch], dim=0)
    labels = torch.stack([b['labels'] for b in batch], dim=0)
    attn_mask = (input_ids != 0).long()
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attn_mask,
        'meta': batch,
    }
