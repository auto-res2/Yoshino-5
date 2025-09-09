import torch
import torch.nn as nn
import timm
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _has_peft = True
except ImportError:
    _has_peft = False

try:
    import bitsandbytes as bnb
    _has_bnb = True
except ImportError:
    _has_bnb = False


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _autocast_dtype(requested_precision: str) -> torch.dtype:
    """Return the correct autocast dtype, handling hardware limitations.

    If bf16 is requested but the current CUDA device does not support it, we
    fall back to fp16 transparently while printing a clear warning so the user
    knows what happened.
    """
    if requested_precision.lower() == 'bf16':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Fallback
        print("[WARN] Current CUDA Device does not support bfloat16 – falling "
              "back to float16 for mixed-precision training.")
        return torch.float16
    if requested_precision.lower() in {'fp16', 'float16'}:
        return torch.float16
    return torch.float32


def _grad_scaler_enabled(dtype: torch.dtype, device: torch.device) -> bool:
    """GradScaler is only useful/valid for fp16 on CUDA."""
    return dtype == torch.float16 and device.type == 'cuda'


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------

def get_model(config):
    """Factory to create the continual learning model."""
    return ContinualVisionModel(config)


class ContinualVisionModel(nn.Module):
    """A wrapper for a timm backbone with swappable heads and optional PEFT adapters."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone_name = config.model_name
        self.image_size = config.image_size
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head
            img_size=self.image_size,
        )
        # Most timm models expose *num_features* – fallback to embed_dim when missing.
        self.embed_dim = getattr(self.backbone, 'num_features', None) or getattr(
            self.backbone, 'embed_dim', None
        )
        if self.embed_dim is None:
            raise ValueError(
                f"Could not infer embedding dimension for backbone {self.backbone_name}."
            )
        self.head_bank = nn.ModuleDict()
        self.active_task_id = None

    # ------------------------------------------------------------------
    # Forward / task-management helpers
    # ------------------------------------------------------------------
    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale → RGB
            x = x.repeat(1, 3, 1, 1)
        if 'mixer' in self.backbone_name and x.shape[-1] != self.image_size:
            x = nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
            )
        # timm Vision Transformers expose *forward_features* that returns either
        # a sequence (B, N+1, C) or already-pooled features (B, C).
        features = (
            self.backbone.forward_features(x)
            if hasattr(self.backbone, 'forward_features')
            else self.backbone(x)
        )
        if 'vit' in self.backbone_name and features.ndim == 3:
            features = features[:, 0]  # CLS token
        if self.active_task_id is not None:
            return self.head_bank[str(self.active_task_id)](features)
        return features

    def set_active_task(self, task_id):
        if str(task_id) not in self.head_bank:
            self.add_task_head(task_id, self.config.classes_per_task)
        self.active_task_id = task_id
        self.backbone.train(self.training)
        self.head_bank.train(self.training)

    def add_task_head(self, task_id, num_classes):
        self.head_bank[str(task_id)] = nn.Linear(self.embed_dim, num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def apply_salora_adapters(self):
        if not _has_peft:
            print("WARNING: `peft` library not found. Cannot apply SALoRA adapters.")
            return
        self.freeze_backbone()
        target_modules = (
            ["qkv", "proj", "fc1", "fc2"] if 'vit' in self.backbone_name else ["fn1", "fn2"]
        )
        peft_config = LoraConfig(
            r=self.config.salora_rank,
            lora_alpha=self.config.salora_alpha,
            lora_dropout=self.config.salora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        print("Applied SALoRA (LoRA) adapters.")
        self.backbone.print_trainable_parameters()


# -----------------------------------------------------------------------------
# RL components
# -----------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """A simple MLP Actor-Critic agent for task scheduling."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)


class A2CAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.m = config.search_window_m
        self.state_dim = self.m * self.m * 2
        self.action_dim = self.m
        self.policy = ActorCritic(self.state_dim, self.action_dim, config.actor_critic_hidden_dim).to(device)
        self.optimizer = optim.AdamW(
            self.policy.parameters(), lr=config.lr * 0.1, weight_decay=config.weight_decay
        )
        self.gamma = config.a2c_gamma
        self.log_probs, self.values, self.rewards = [], [], []

    # ------------------------------------------------------------------
    # Action selection with robust masking
    # ------------------------------------------------------------------
    def select_action(self, state, valid_actions_mask):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.policy(state)

        # Ensure logits is 1-D (action_dim,)
        logits = logits.squeeze(0)

        if valid_actions_mask is not None:
            mask = torch.tensor(valid_actions_mask, dtype=torch.bool, device=self.device)
            # Pad mask to action_dim
            if mask.numel() < self.action_dim:
                pad = torch.zeros(self.action_dim - mask.numel(), dtype=torch.bool, device=self.device)
                mask = torch.cat([mask, pad], dim=0)
            logits[~mask] = -1e9  # effectively zero probability after softmax
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        # If, by numerical accident, an invalid action is sampled, pick the first valid one.
        if valid_actions_mask is not None:
            attempts = 0
            while not mask[action] and attempts < 10:
                action = dist.sample()
                attempts += 1
            if not mask[action]:
                action = torch.where(mask)[0][0]
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        return action.item()

    # ------------------------------------------------------------------
    # A2C bookkeeping
    # ------------------------------------------------------------------
    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        if not self.rewards:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        log_probs, values = torch.cat(self.log_probs), torch.cat(self.values).squeeze()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs, self.values, self.rewards = [], [], []


# -----------------------------------------------------------------------------
# Optimiser / training helpers
# -----------------------------------------------------------------------------

def get_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    if config.optimizer == 'AdamW8bit' and _has_bnb and 'cuda' in str(config.device):
        return bnb.optim.AdamW8bit(
            params, lr=config.lr, betas=tuple(config.betas), weight_decay=config.weight_decay
        )
    return torch.optim.AdamW(
        params, lr=config.lr, betas=tuple(config.betas), weight_decay=config.weight_decay
    )


def train_one_task(model, train_loader, optimizer, device, config):
    """Train *model* for a single continual-learning task."""

    model.train()

    dtype = _autocast_dtype(config.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=_grad_scaler_enabled(dtype, device))

    num_steps = getattr(config, 'steps_per_task', len(train_loader) * config.epochs_per_task)
    progress_bar = tqdm(total=num_steps, desc='Training Task', leave=False)

    step = 0
    while step < num_steps:
        for x, y, _ in train_loader:
            if step >= num_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                outputs = model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            step += 1
    progress_bar.close()