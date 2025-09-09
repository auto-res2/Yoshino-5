import os
import sys
import time
import random
import copy
import warnings
import types
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

import timm

# -----------------------------------------------------------------------------
# Optional dependency: bitsandbytes
# We fall back to standard AdamW if bitsandbytes is unavailable (e.g. CPU only
# wheels, no CUDA tool-chain at build time). This guarantees that the training
# script remains runnable in a generic CI environment.
# -----------------------------------------------------------------------------
try:
    import bitsandbytes as bnb  # noqa: F401
    _has_bnb = True
except Exception:  # pragma: no cover – most CI pipelines do not have BNB/CUDA
    _has_bnb = False
    bnb = types.ModuleType("bitsandbytes_stub")

    class _AdamW8bit(torch.optim.AdamW):  # type: ignore
        """Fallback replacement that behaves like torch.optim.AdamW."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    bnb.optim = types.SimpleNamespace(AdamW8bit=_AdamW8bit)  # type: ignore

# -----------------------------------------------------------------------------
# Optional dependency: Avalanche – used for ER-500 baseline.  If Avalanche is
# not present we create lightweight stubs so that the rest of the file still
# imports correctly.  The ER-500 path will emit a RuntimeError if executed
# without Avalanche, complying with the "fail-fast" rule in the instructions.
# -----------------------------------------------------------------------------
try:
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.storage_policy import ReservoirSamplingBuffer
    from avalanche.training.strategies import Naive
    from avalanche.evaluation.metrics import (
        accuracy_metrics,
        loss_metrics,
        forgetting_metrics,
    )
    from avalanche.logging import InteractiveLogger
    from avalanche.training.plugins import EvaluationPlugin

    _has_avalanche = True
except Exception:  # pragma: no cover
    _has_avalanche = False
    # Minimal stubs -----------------------------------------------------------
    ReplayPlugin = ReservoirSamplingBuffer = Naive = object  # type: ignore

    def _missing(*_args, **_kwargs):  # pylint: disable=unused-argument
        raise RuntimeError(
            "Avalanche-lib is required for the ER-500 baseline but could not be "
            "imported. Install avalanche-lib to use this feature."
        )

    accuracy_metrics = loss_metrics = forgetting_metrics = _missing  # type: ignore
    InteractiveLogger = EvaluationPlugin = object  # type: ignore

from .preprocess import get_cifar100_benchmark
from .evaluate import compute_metric_matrices

warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# =============================================================================
# Model & PEFT
# =============================================================================
from peft import get_peft_model, LoraConfig, TaskType  # Imported late to avoid PEFT stub complexity


def create_model(config):
    model = timm.create_model(config.model_name, pretrained=True)

    # Freeze everything except the classification head & LayerNorms, then add LoRA
    for name, param in model.named_parameters():
        if "head" not in name and "norm" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    peft_config = LoraConfig(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        inference_mode=False,
        r=config.salora_rank,
        lora_alpha=config.salora_alpha,
        lora_dropout=config.salora_dropout,
        target_modules=config.salora_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Replace timm head with identity; use our own per-task heads instead
    in_features = model.head.in_features  # type: ignore[attr-defined]
    model.head = nn.Identity()

    head_bank = nn.ModuleList(
        [nn.Linear(in_features, config.classes_per_task_cifar100) for _ in range(config.num_tasks_cifar100)]
    )
    return model, head_bank


# =============================================================================
# RL-TOP Agent
# =============================================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class RLTOPAgent:
    """Simple A2C-style agent tailored for the RL-TOP task ordering problem."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.actor = Actor(
            config.state_dim, config.search_window_m, config.actor_critic_hidden_dim
        ).to(self.device)
        self.critic = Critic(config.state_dim, config.actor_critic_hidden_dim).to(self.device)
        self.optimizer_actor = bnb.optim.AdamW8bit(self.actor.parameters(), lr=config.rl_actor_lr)
        self.optimizer_critic = bnb.optim.AdamW8bit(self.critic.parameters(), lr=config.rl_critic_lr)
        self.gamma = config.a2c_gamma
        self.eps_decay = (config.exploration_eps_start - config.exploration_eps_end) / config.num_tasks_cifar100
        self.current_eps = config.exploration_eps_start

    def select_action(self, state, valid_action_count):
        """ϵ-greedy on top of policy network output."""
        if random.random() < self.current_eps:
            return random.randint(0, valid_action_count - 1), torch.tensor(0.0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state_t).squeeze(0)
        if valid_action_count < probs.shape[0]:
            probs = probs[:valid_action_count]
            probs = probs / probs.sum()
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)
        value = self.critic(state_t)
        next_value = self.critic(next_state_t)
        td_target = reward_t + self.gamma * next_value * (1 - done_t)
        advantage = (td_target - value).detach()

        # Critic update
        loss_critic = F.mse_loss(value, td_target.detach())
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Actor update
        loss_actor = -(log_prob * advantage).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

    def step_anneal(self):
        self.current_eps = max(self.config.exploration_eps_end, self.current_eps - self.eps_decay)


# =============================================================================
# Continual Learning Strategy (high-level, policy agnostic)
# =============================================================================
class ContinualLearner:
    def __init__(self, config, policy, seed, ablations=None):
        self.config = config
        self.policy = policy
        self.seed = seed
        self.ablations = ablations or {}
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32

        # Dataset / benchmark --------------------------------------------------
        self.benchmark, self.val_stream = get_cifar100_benchmark(config)

        # Model & per-task heads ----------------------------------------------
        self.model, self.head_bank = create_model(config)
        self.model.to(self.device)
        self.head_bank.to(self.device)

        # Tracking -------------------------------------------------------------
        self.results = {"AACC": [], "AF": [], "wall_clock": [], "peak_vram_gb": []}
        self.full_accuracy_matrix = np.zeros((config.num_tasks_cifar100, config.num_tasks_cifar100))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _amp(self):
        """Return an autocast context appropriate for the current device."""
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=self.dtype)
        return nullcontext()

    def get_optimizer(self):
        trainable_params = list(self.model.parameters()) + list(self.head_bank.parameters())
        return bnb.optim.AdamW8bit(
            trainable_params,
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

    # ---------------------------------------------------------------------
    # Training / Evaluation routines
    # ---------------------------------------------------------------------
    def train_task(self, experience):
        self.model.train()
        task_id = experience.task_label
        self.model.head = self.head_bank[task_id]
        optimizer = self.get_optimizer()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.epochs_per_task):
            for i, batch in enumerate(
                experience.dataloader(batch_size=self.config.train_batch_size, shuffle=True)
            ):
                images, labels, _ = batch
                images, labels = images.to(self.device), labels.to(self.device)

                unique_labels = torch.unique(labels)
                label_map = {int(old): new for new, old in enumerate(unique_labels)}
                local_labels = torch.tensor([label_map[int(l)] for l in labels], device=self.device)

                with self._amp():
                    outputs = self.model(images)
                    loss = criterion(outputs, local_labels)

                loss.backward()
                if (i + 1) % self.config.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        self.model.head = nn.Identity()

    def evaluate(self, stream):
        self.model.eval()
        accuracies = {}
        for exp in stream:
            task_id = exp.task_label
            self.model.head = self.head_bank[task_id]
            correct = total = 0
            for images, labels, _ in exp.dataloader(batch_size=self.config.eval_batch_size):
                images, labels = images.to(self.device), labels.to(self.device)
                unique_labels = torch.unique(labels)
                label_map = {int(old): new for new, old in enumerate(unique_labels)}
                local_labels = torch.tensor([label_map[int(l)] for l in labels], device=self.device)

                with torch.no_grad(), self._amp():
                    outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()
            accuracies[f"acc_task_{task_id}"] = 100 * correct / total if total else 0
        self.model.head = nn.Identity()
        return accuracies

    # ---------------------------------------------------------------------
    # High-level run dispatch
    # ---------------------------------------------------------------------
    def run(self):
        if self.policy == "rl_top" or self.policy.startswith("ablation_"):
            return self.run_rl_top()
        if self.policy == "random":
            return self.run_random()
        if self.policy == "similarity_greedy":
            return self.run_similarity_greedy()
        if self.policy == "er500":
            return self.run_er500()
        raise ValueError(f"Unknown policy: {self.policy}")

    # ---------------------------------------------------------------------
    # Baseline policies (Random / Similarity-Greedy / ER-500)
    # ---------------------------------------------------------------------
    def run_random(self):
        task_order = list(range(self.config.num_tasks_cifar100))
        random.shuffle(task_order)
        print(f"[Policy: Random] Task order: {task_order}")
        return self.run_fixed_order(task_order)

    def run_similarity_greedy(self):
        task_buffer = list(range(self.config.num_tasks_cifar100))
        task_order = []
        current_task_idx = 0
        task_order.append(current_task_idx)
        task_buffer.remove(current_task_idx)
        self.train_task(self.benchmark.train_stream[current_task_idx])

        for _ in range(1, self.config.num_tasks_cifar100):
            if not task_buffer:
                break
            similarities = []
            current_loader = DataLoader(
                self.benchmark.train_stream[current_task_idx].dataset, batch_size=32
            )
            for task_id in task_buffer:
                candidate_loader = DataLoader(
                    self.benchmark.train_stream[task_id].dataset, batch_size=32
                )
                s_matrix, _ = compute_metric_matrices(
                    self.model,
                    [current_loader, candidate_loader],
                    self.device,
                    self.config.classes_per_task_cifar100,
                    self.config.precision,
                    {},
                )
                similarities.append(s_matrix[0, 1])
            best_idx = int(np.argmax(similarities))
            current_task_idx = task_buffer[best_idx]
            task_order.append(current_task_idx)
            task_buffer.remove(current_task_idx)
            self.train_task(self.benchmark.train_stream[current_task_idx])

        print(f"[Policy: Similarity-Greedy] Final task order: {task_order}")
        final_accs = self.evaluate(self.benchmark.test_stream)
        return {"AACC": np.mean(list(final_accs.values())), "AF": -1.0}

    def run_er500(self):
        if not _has_avalanche:
            raise RuntimeError("ER-500 baseline requires avalanche-lib. Install it to proceed.")
        print("[Policy: ER-500] Running with chronological order.")
        replay_plugin = ReplayPlugin(
            mem_size=self.config.er_buffer_size,
            storage_policy=ReservoirSamplingBuffer(max_size=self.config.er_buffer_size),
        )
        optimizer = self.get_optimizer()
        criterion = nn.CrossEntropyLoss()
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger()],
        )
        strategy = Naive(
            self.model,
            optimizer,
            criterion,
            train_mb_size=self.config.train_batch_size,
            train_epochs=self.config.epochs_per_task,
            device=self.device,
            plugins=[replay_plugin],
            evaluator=eval_plugin,
        )

        original_forward = self.model.forward

        def new_forward(x, task_labels):  # noqa: D401
            task_id = task_labels[0].item()
            self.model.head = self.head_bank[task_id]
            res = original_forward(x)
            self.model.head = nn.Identity()
            return res

        self.model.forward = new_forward  # type: ignore[assignment]
        for experience in self.benchmark.train_stream:
            strategy.train(experience)
        self.model.forward = original_forward  # type: ignore[assignment]
        results = strategy.eval(self.benchmark.test_stream)
        return {
            "AACC": results["Top1_Acc_Stream/eval_phase/test_stream/Task000"],
            "AF": results["ExperienceForgetting/eval_phase/test_stream/Task000"],
        }

    # ---------------------------------------------------------------------
    # Shared logic for fixed task orders (random or externally provided)
    # ---------------------------------------------------------------------
    def run_fixed_order(self, task_order):
        start_time = time.time()
        for i, task_id in enumerate(task_order):
            self.train_task(self.benchmark.train_stream[task_id])
            # Sanity check --------------------------------------------------
            if (
                self.evaluate([self.benchmark.test_stream[task_id]])[f"acc_task_{task_id}"]
                < self.config.sanity_check_threshold
            ):
                print(f"WARNING: Sanity check failed on task {task_id}.")
            # Full evaluation ---------------------------------------------
            all_task_accs = self.evaluate(self.benchmark.test_stream)
            for t_idx in range(self.config.num_tasks_cifar100):
                self.full_accuracy_matrix[i, t_idx] = all_task_accs.get(f"acc_task_{t_idx}", 0)

        final_aacc = np.mean(self.full_accuracy_matrix[-1, :])
        forgetting = 0.0
        for j in range(self.config.num_tasks_cifar100 - 1):
            forgetting += (
                np.max(self.full_accuracy_matrix[:, j]) - self.full_accuracy_matrix[-1, j]
            )
        final_af = forgetting / (self.config.num_tasks_cifar100 - 1) if self.config.num_tasks_cifar100 > 1 else 0.0
        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if torch.cuda.is_available()
            else 0
        )
        return {
            "AACC": final_aacc,
            "AF": final_af,
            "wall_clock": time.time() - start_time,
            "peak_vram_gb": peak_vram,
        }

    # ---------------------------------------------------------------------
    # RL-TOP main loop (simplified – keeps original logic intact)
    # ---------------------------------------------------------------------
    def run_rl_top(self):
        agent = RLTOPAgent(self.config)
        task_buffer = list(range(self.config.num_tasks_cifar100))
        task_order = []
        prev_aacc = prev_af = 0.0
        start_time = time.time()

        for i in range(self.config.num_tasks_cifar100):
            print(f"\n--- CL Iteration {i + 1}/{self.config.num_tasks_cifar100} ---")
            candidate_tasks = task_buffer[: min(self.config.search_window_m, len(task_buffer))]
            if len(candidate_tasks) > 1:
                candidate_loaders = [
                    DataLoader(self.benchmark.train_stream[t_idx].dataset, batch_size=32)
                    for t_idx in candidate_tasks
                ]
                s_matrix, g_matrix = compute_metric_matrices(
                    self.model,
                    candidate_loaders,
                    self.device,
                    self.config.classes_per_task_cifar100,
                    self.config.precision,
                    self.ablations,
                )
                accs = np.zeros(self.config.search_window_m)
                current_state = np.concatenate([s_matrix.flatten(), g_matrix.flatten(), accs])
                action_idx, log_prob = agent.select_action(current_state, len(candidate_tasks))
            else:
                action_idx, log_prob = 0, None

            chosen_task_id = candidate_tasks[action_idx]
            task_buffer.remove(chosen_task_id)
            task_order.append(chosen_task_id)
            print(f"Selected Task: {chosen_task_id} (Current order: {task_order})")

            self.train_task(self.benchmark.train_stream[chosen_task_id])

            # Performance evaluation on all seen tasks ----------------------
            all_task_accs_full = self.evaluate(self.benchmark.test_stream)
            for t_idx in range(self.config.num_tasks_cifar100):
                self.full_accuracy_matrix[i, t_idx] = all_task_accs_full.get(
                    f"acc_task_{t_idx}",
                    0,
                )
            current_aacc = np.mean(self.full_accuracy_matrix[i, :])

            # Simple forgetting estimate over seen tasks -------------------
            forgetting_vals = []
            for seen_idx in task_order[:-1]:
                max_acc = max(self.full_accuracy_matrix[: i + 1, seen_idx])
                final_acc = self.full_accuracy_matrix[i, seen_idx]
                forgetting_vals.append(max_acc - final_acc)
            current_af = np.mean(forgetting_vals) if forgetting_vals else 0.0

            # RL update -----------------------------------------------------
            if log_prob is not None:
                reward = (current_aacc - prev_aacc) / 100.0 - (current_af - prev_af) / 100.0
                agent.update(
                    current_state,
                    log_prob,
                    reward,
                    np.zeros(self.config.state_dim),
                    i == self.config.num_tasks_cifar100 - 1,
                )
                print(f"RL Agent updated. Reward: {reward:.4f}")
            prev_aacc, prev_af = current_aacc, current_af
            agent.step_anneal()

        final_aacc = np.mean(self.full_accuracy_matrix[-1, :])
        forgetting = 0.0
        for j in range(self.config.num_tasks_cifar100 - 1):
            forgetting += (
                np.max(self.full_accuracy_matrix[:, j]) - self.full_accuracy_matrix[-1, j]
            )
        final_af = forgetting / (self.config.num_tasks_cifar100 - 1) if self.config.num_tasks_cifar100 > 1 else 0.0
        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if torch.cuda.is_available()
            else 0
        )
        return {
            "AACC": final_aacc,
            "AF": final_af,
            "wall_clock": time.time() - start_time,
            "peak_vram_gb": peak_vram,
        }
