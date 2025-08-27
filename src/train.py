import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class DummyExecutor:
    def __init__(self, graph):
        self.graph = graph

    def run(self):
        try:
            return eval(str(self.graph.get("expr", "0")))
        except Exception:
            return None


class DummyGraphGenerator(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.l = nn.Linear(hidden, 3)

    def forward(self, h):
        return self.l(h)


class TinyModel(nn.Module):
    def __init__(self, with_graph_head: bool = True, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(100, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)
        self.with_graph_head = with_graph_head
        if with_graph_head:
            self.graph_head = DummyGraphGenerator(32)

    def forward_logits(self, questions):
        qs = torch.tensor([[len(q) % 100 for _ in range(3)] for q in questions])
        emb = self.embed(qs)
        h, _ = self.rnn(emb)
        logits = self.fc(h[:, -1])
        return logits, h[:, -1]

    @torch.no_grad()
    def generate(self, questions, return_graph: bool = True, prune: bool = False):
        logits, last_h = self.forward_logits(questions)
        answers = logits.argmax(-1).cpu().tolist()
        graphs = []
        if self.with_graph_head:
            gl = self.graph_head(last_h)
            for ans, _ in zip(answers, gl):
                graphs.append({"expr": f"{ans}"})
        else:
            graphs = [{} for _ in answers]
        if return_graph:
            return answers, graphs
        return answers


def exact_match(pred, gt):
    return (np.array(pred) == gt.numpy()).mean()


def faithfulness(graphs, answers):
    return np.mean([DummyExecutor(g).run() == a for g, a in zip(graphs, answers)])


@dataclass
class TrainConfig:
    steps: int = 100
    lr: float = 1e-2
    with_ev: bool = True
    with_graph_head: bool = True


class TinyTrainer:
    def __init__(self, model: TinyModel, cfg: TrainConfig, loader: DataLoader):
        self.model = model
        self.cfg = cfg
        self.loader = loader
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def fit(self):
        losses, exec_pass = [], []
        it = iter(self.loader)
        for _ in range(self.cfg.steps):
            try:
                qbatch, gt = next(it)
            except StopIteration:
                it = iter(self.loader)
                qbatch, gt = next(it)
            logits, _ = self.model.forward_logits(qbatch)
            loss = F.cross_entropy(logits, gt.clamp(min=0, max=9))
            if self.cfg.with_ev:
                with torch.no_grad():
                    preds, graphs = self.model.generate(qbatch, return_graph=True)
                    ev_pen = 1 - faithfulness(graphs, preds)
                loss = loss + float(ev_pen)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                preds, graphs = self.model.generate(qbatch, return_graph=True)
                exec_pass.append(faithfulness(graphs, preds))
        return losses, exec_pass


def train(dls: dict, images_dir: str, cfg: TrainConfig) -> dict:
    os.makedirs(images_dir, exist_ok=True)
    model = TinyModel(with_graph_head=cfg.with_graph_head, seed=42)
    trainer = TinyTrainer(model, cfg, dls["train"])
    losses, exec_pass = trainer.fit()
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="train")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss (log)")
    plt.legend()
    plt.tight_layout()
    loss_pdf = f"{images_dir}/training_loss_ablation.pdf"
    plt.savefig(loss_pdf, bbox_inches="tight")
    return {"loss_pdf": loss_pdf, "final_loss": losses[-1], "exec_pass_mean": float(np.mean(exec_pass))}
