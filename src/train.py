import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from contextlib import nullcontext
import gc
import timm
from evaluate import ContinualMetrics, evaluate

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _has_peft = True
except ImportError:
    _has_peft = False
    print("WARNING: `peft` library not found. SALoRA adapters will be disabled.")

try:
    from fvcore.nn import FlopCountAnalysis
    _has_fvcore = True
except ImportError:
    _has_fvcore = False
    print("WARNING: `fvcore` not found. FLOP/s and VRAM analysis will be disabled.")
    
try:
    import bitsandbytes as bnb
    _has_bnb = True
except ImportError:
    _has_bnb = False
    print("WARNING: `bitsandbytes` not found. 8-bit optimizers will fall back to torch.optim.AdamW.")

""" ---------------------------------------------------------------------------
Avalanche import compatibility
---------------------------------------------------------------------------
The public API of Avalanche has slightly changed across minor versions.  In
0.5.x the `ReservoirSamplingBuffer` utility lived in `avalanche.storage_policy`
whereas from 0.6.x onward it was moved to `avalanche.training.storage_policy`.
The following try / except chain keeps the code robust to both versions without
forcing a specific pin in `requirements.txt` (we currently use 0.6.x because it
is the first version with Python 3.11 wheels).
"""

_has_avalanche = False
ReplayPlugin = object
ReservoirSamplingBuffer = object

try:
    from avalanche.training.plugins import ReplayPlugin as _ReplayPlugin
    ReplayPlugin = _ReplayPlugin

    try:
        # Avalanche < 0.6
        from avalanche.storage_policy import ReservoirSamplingBuffer as _RSB
    except ImportError:
        # Avalanche ≥ 0.6
        from avalanche.training.storage_policy import ReservoirSamplingBuffer as _RSB
    ReservoirSamplingBuffer = _RSB
    _has_avalanche = True
except ImportError:
    print("WARNING: `avalanche-lib` not found. Experience replay baselines will be disabled.")


class ContinualVisionModel(nn.Module):
    """A wrapper for a timm backbone with swappable heads and optional adapters."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=True,
            num_classes=0
        )
        # Different timm models expose the embedding size under different names.
        self.embed_dim = getattr(self.backbone, 'embed_dim', None) or getattr(self.backbone, 'num_features')

        # Freeze backbone weights (we only train head or adapters)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optional LoRA adapters (a.k.a. SALoRA in the paper)
        if _has_peft:
            # ------------------------------------------------------------------
            # PEFT versions prior to 0.12 do NOT expose `TaskType.IMAGE_CLASSIFICATION`.
            # Fallback gracefully to the default FEATURE_EXTRACTION task type when
            # the newer enum entry is unavailable so that experiments keep running
            # regardless of the installed PEFT minor version.
            # ------------------------------------------------------------------
            if hasattr(TaskType, "IMAGE_CLASSIFICATION"):
                _task_type = TaskType.IMAGE_CLASSIFICATION
            else:
                _task_type = TaskType.FEATURE_EXTRACTION  # safe default across versions

            peft_config = LoraConfig(
                r=config.salora_rank,
                lora_alpha=getattr(config, 'salora_alpha', config.salora_rank * 2),
                lora_dropout=getattr(config, 'salora_dropout', 0.1),
                bias="none",
                target_modules=["qkv", "proj", "fc1", "fc2"],
                task_type=_task_type,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            print("Applied SALoRA (LoRA) adapters.")
            self.backbone.print_trainable_parameters()

        self.head_bank = nn.ModuleDict()
        self.active_head = None

    def forward(self, x):
        # Handle grayscale inputs (e.g. Permuted-MNIST)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # MLP-Mixer models in timm expect 224×224 ‑ we upsample smaller inputs
        if 'mixer' in self.config.model_name and x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        features = self.backbone(x)
        if self.active_head is not None:
            return self.head_bank[self.active_head](features)
        return features

    def add_or_set_task_head(self, task_id):
        task_key = str(task_id)
        if task_key not in self.head_bank:
            self.head_bank[task_key] = nn.Linear(self.embed_dim, self.config.classes_per_task)
        self.active_head = task_key


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
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


class A2CAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.state_dim = config.search_window_m**2 * 2 + config.search_window_m
        self.action_dim = config.search_window_m
        self.policy = ActorCritic(self.state_dim, self.action_dim, config.actor_critic_hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr * 0.1)
        self.gamma = config.a2c_gamma
        self.memory = []

    def select_action(self, state, valid_actions_mask):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs, value = self.policy(state)
        
        probs = probs * valid_actions_mask
        probs = probs / probs.sum()

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.memory.append((log_prob, value))
        return action.item()

    def update(self, rewards):
        if not self.memory:
            return

        log_probs, values = zip(*self.memory)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        advantages = returns - values
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


def get_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == 'AdamW8bit' and _has_bnb and 'cuda' in str(config.device):
        return bnb.optim.AdamW8bit(params, lr=config.lr, betas=tuple(config.betas), weight_decay=config.weight_decay)
    else:
        if config.optimizer == 'AdamW8bit':
            print("Falling back to torch.optim.AdamW (8-bit optimizer unavailable).")
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


# Remaining functions unchanged ------------------------------------------------
