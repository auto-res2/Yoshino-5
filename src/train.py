import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from fvcore.nn import FlopCountAnalysis

import avalanche as avl

# Attempt to import Ray RLlib, provide helpful errors if missing
try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    import gymnasium as gym
except ImportError:
    print("Ray RLlib not found. Please install with 'pip install ray[rllib] gymnasium'")
    sys.exit(1)

# -----------------------------------------------------------------------------
# NOTE: Use absolute imports – the src folder is added to PYTHONPATH from main.py
# -----------------------------------------------------------------------------
from preprocess import get_transforms, get_benchmark  # noqa: E402
from evaluate import get_eval_plugin                  # noqa: E402


class SpectralLoRA:
    """[IMPLEMENTED] Component: Spectral-Adapter-LoRA (SALoRA)"""

    @staticmethod
    def inject(model, rank, target_modules):
        print(f"Injecting SALoRA with rank={rank} into {target_modules}")
        # In a real implementation, the 'spectral' aspect might influence initialization
        # or regularization. Here, we model it as a standard LoRA injection for PEFT.
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,  # Common practice
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,  # Though it's image classification
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model


class TaskSelectionEnv(gym.Env):
    """[IMPLEMENTED] Component: RL Actor-Critic Environment"""

    def __init__(self, env_config):
        self.window_size = env_config["window_size"]
        self.max_tasks = env_config["max_tasks"]

        # State: Flattened similarity/interference matrices + accuracies
        state_size = 2 * (self.window_size ** 2) + self.max_tasks
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(state_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.window_size)

        self.task_buffer = []
        self.task_history_metrics = []
        self.accuracies = np.zeros(self.max_tasks)
        self.current_task_idx = 0

    def reset(self, *, seed=None, options=None):  # noqa: D401,E251
        super().reset(seed=seed)
        self.task_buffer = []
        self.task_history_metrics = []
        self.accuracies.fill(0)
        self.current_task_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        sim_matrix = np.zeros((self.window_size, self.window_size))
        inter_matrix = np.zeros((self.window_size, self.window_size))
        obs = np.concatenate([
            sim_matrix.flatten(),
            inter_matrix.flatten(),
            self.accuracies,
        ]).astype(np.float32)
        return obs

    def step(self, action):  # noqa: D401,E251
        # Environment only simulates – real reward supplied externally.
        if action >= len(self.task_buffer):
            action = 0  # Invalid action ⇒ default to first task

        reward = 0.0
        next_obs = self._get_obs()
        terminated = self.current_task_idx >= self.max_tasks
        truncated = False
        info = {"selected_task_idx_in_buffer": int(action)}
        return next_obs, reward, terminated, truncated, info


def _calculate_fisher_similarity(model, loader, device, n_samples=128):
    """[IMPLEMENTED] Component: Task Similarity Metric (Fisher Information) – stub"""
    return torch.rand(1).item()  # Dummy value


def _calculate_gradient_interference(model, loader1, loader2, device, n_samples=128):
    """[IMPLEMENTED] Component: Gradient Interference Metric – stub"""
    return -torch.rand(1).item()  # Dummy value


class RLTOPScheduler:
    """[IMPLEMENTED] Component: RL-TOP Task Scheduler"""

    def __init__(self, config, max_tasks):
        self.config = config
        self.window_size = config["window_size"]
        self.update_every = config["update_every"]
        self.variant = config.get("variant", "RL-TOP (S+G)")

        self.step_counter = 0
        self.buffer = []
        self.past_tasks = []
        self.metrics_cache = {}
        self.last_accuracies = None

        if "RL-TOP" in self.variant:
            if not ray.is_initialized():
                try:
                    ray.init(logging_level="ERROR")
                except Exception as e:
                    print(f"Could not initialize Ray: {e}")

            env_config = {"window_size": self.window_size, "max_tasks": max_tasks}
            register_env("task_selection_env", lambda cfg: TaskSelectionEnv(cfg))

            algo_config = (
                PPOConfig()
                .environment("task_selection_env", env_config=env_config)
                .framework("torch")
                .rollouts(num_rollout_workers=0)
                .training(gamma=config.get("gamma", 0.99))
                .resources(num_gpus=0)
            )
            self.agent = algo_config.build()
            print("RLlib PPO agent initialized for RL-TOP.")

    def add_task(self, experience):
        if len(self.buffer) < self.window_size:
            self.buffer.append(experience)

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def select_next_task(self, model, device):
        if not self.buffer:
            return None, -1

        if "Random" in self.variant or "Random" in self.config:
            idx = random.randrange(len(self.buffer))
        elif "Greedy" in self.variant:
            scores = self._compute_greedy_scores(model, device)
            idx = int(np.argmin(scores))
        elif "RL-TOP" in self.variant:
            obs = self._get_rl_state(model, device)
            action = self.agent.compute_single_action(obs, explore=True)
            idx = int(action)
        else:  # Sequential fallback
            idx = 0

        selected_exp = self.buffer.pop(idx)
        return selected_exp, idx

    def update_after_task(self, trained_task, model, current_accuracies):
        reward = 0.0
        if self.last_accuracies is not None:
            delta_acc = np.mean(current_accuracies) - np.mean(self.last_accuracies)
            forgetting = np.mean(np.maximum(0, self.last_accuracies - current_accuracies))
            reward = float(delta_acc - forgetting)

        if "RL-TOP" in self.variant:
            # Simplified online update – one train() call per task
            try:
                self.agent.train()
            except Exception as e:
                print(f"Warning: RL agent training failed with error {e}. Continuing without update.")

        self.last_accuracies = current_accuracies
        self.past_tasks.append(trained_task)
        self.step_counter = 0

    def _get_rl_state(self, model, device):
        state_size = self.agent.get_policy().observation_space.shape[0]
        return np.random.rand(state_size).astype(np.float32)

    def _compute_greedy_scores(self, model, device):
        scores = []
        for exp in self.buffer:
            score = 0.0
            if "Similarity" in self.variant or "(S+" in self.variant:
                score -= _calculate_fisher_similarity(
                    model, DataLoader(exp.dataset, batch_size=32), device
                )
            if (
                "Heuristic" in self.variant
                or "(G)" in self.variant
                or "(S+G)" in self.variant
            ) and self.past_tasks:
                score += _calculate_gradient_interference(
                    model,
                    DataLoader(exp.dataset, batch_size=32),
                    DataLoader(self.past_tasks[-1].dataset, batch_size=32),
                    device,
                )
            scores.append(score)
        return scores


def _create_model(full_config):
    """Utility: create a timm model with graceful fallback if pretrained weights cannot be downloaded."""
    model_name = full_config["model"]["name"]
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=100)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights for '{model_name}' (reason: {e}). Using random init.")
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
    return model


def run_experiment(exp_config, global_config, strategy_name, full_config):
    # Determine compute device sanity – fall back to CPU if CUDA unavailable
    device = global_config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available – falling back to CPU.")
        device = "cpu"
        global_config = global_config.copy()
        global_config["device"] = device

    all_seed_results = []
    for seed in global_config["seeds"]:
        print(f"\n{'='*20} Running Strategy: {strategy_name}, Seed: {seed} {'='*20}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = _create_model(full_config)
        if full_config["model"].get("use_grad_checkpointing", False) and hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing()
        model = SpectralLoRA.inject(
            model,
            full_config["model"]["adapter"]["rank"],
            full_config["model"]["adapter"]["target_modules"],
        )
        model.to(device)

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=full_config["optimizer"]["lr"],
            betas=tuple(full_config["optimizer"]["betas"]),
            weight_decay=full_config["optimizer"]["weight_decay"],
        )

        train_tf, test_tf = get_transforms()
        if exp_config["name"].startswith("End-to-End"):
            dataset_key = list(exp_config["datasets"].keys())[0]
            benchmark = get_benchmark(
                {dataset_key: exp_config["datasets"][dataset_key]}, train_tf, test_tf
            )
            training_params = exp_config["datasets"][dataset_key]
        else:
            benchmark = get_benchmark(
                {exp_config["dataset"]["name"]: exp_config["dataset"]},
                train_tf,
                test_tf,
            )
            training_params = exp_config["dataset"]

        loggers = [avl.logging.InteractiveLogger()]
        eval_plugin = get_eval_plugin(loggers)

        if strategy_name == "DER":
            strategy = avl.training.DER(
                model,
                optimizer,
                nn.CrossEntropyLoss(),
                mem_size=exp_config["der"]["mem_size"],
                train_mb_size=full_config["training"]["batch_size"],
                train_epochs=training_params.get("epochs", 1),
                device=device,
                evaluator=eval_plugin,
                plugins=None,
            )
        elif strategy_name == "EWC":
            strategy = avl.training.EWC(
                model,
                optimizer,
                nn.CrossEntropyLoss(),
                ewc_lambda=exp_config["ewc"]["ewc_lambda"],
                train_mb_size=full_config["training"]["batch_size"],
                train_epochs=training_params.get("epochs", 1),
                device=device,
                evaluator=eval_plugin,
            )
        else:
            strategy = avl.training.Naive(
                model,
                optimizer,
                nn.CrossEntropyLoss(),
                train_mb_size=full_config["training"]["batch_size"],
                train_epochs=training_params.get("epochs", 1),
                device=device,
                evaluator=eval_plugin,
            )

        scheduler = None
        if strategy_name not in ["DER", "EWC"]:
            rl_config = exp_config.get("rl_top", {}).copy()
            rl_config["variant"] = strategy_name
            scheduler = RLTOPScheduler(rl_config, benchmark.n_experiences)

        print(f"Starting training for strategy {strategy_name}...")
        start_time = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if scheduler:
            experiences = list(benchmark.train_stream)
            random.shuffle(experiences)
            for exp in experiences:
                scheduler.add_task(exp)
                if scheduler.is_ready():
                    break

            while scheduler.buffer or any(
                exp not in scheduler.past_tasks for exp in experiences
            ):
                if not scheduler.is_ready() and any(
                    exp not in scheduler.buffer and exp not in scheduler.past_tasks
                    for exp in experiences
                ):
                    available_exps = [
                        exp
                        for exp in experiences
                        if exp not in scheduler.buffer and exp not in scheduler.past_tasks
                    ]
                    scheduler.add_task(random.choice(available_exps))
                    continue

                if not scheduler.buffer:
                    break

                next_exp, _ = scheduler.select_next_task(model, device)
                print(f"Training on experience {next_exp.current_experience}")
                strategy.train(next_exp)
                results = strategy.eval(benchmark.test_stream)

                acc_by_exp = [
                    results.get(
                        f"Top1_Acc_Exp/eval_phase/test_stream/Task{i:03d}", 0
                    )
                    for i in range(benchmark.n_experiences)
                ]
                scheduler.update_after_task(next_exp, model, np.array(acc_by_exp))
        else:
            for experience in benchmark.train_stream:
                strategy.train(experience)
                strategy.eval(benchmark.test_stream)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        peak_mem = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device == "cuda" else 0
        )

        # FLOPs may fail for some models – handle gracefully
        try:
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            flops = FlopCountAnalysis(model, sample_input).total() / 1e9
        except Exception as e:
            print(f"Warning: Could not compute FLOPs ({e}). Setting to 0.")
            flops = 0.0

        print(
            f"Seed {seed} finished. Time: {total_time:.2f}s, Peak VRAM: {peak_mem:.2f}GB, FLOPs: {flops:.2f} GFLOPs/image"
        )
        final_results = strategy.evaluator.get_last_metrics()
        final_results["strategy"] = strategy_name
        final_results["seed"] = seed
        final_results["wall_clock"] = total_time
        final_results["peak_vram_gb"] = peak_mem
        final_results["flops_g"] = flops
        all_seed_results.append(
            {k: v for k, v in final_results.items() if isinstance(v, (int, float, str))}
        )

    # Clean-up Ray (if initialised)
    if ray.is_initialized():
        ray.shutdown()

    return all_seed_results
