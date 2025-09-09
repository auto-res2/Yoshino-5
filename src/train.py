import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import timm
import bitsandbytes as bnb
from tqdm import tqdm
import copy

# --- SALoRA Implementation (Conceptual) ---
# This is a simplified placeholder for the actual SALoRA implementation.
class SALoRALayer(nn.Module):
    """A minimal LoRA-style adapter used by SALoRA.

    It simply projects the input through A (down-projection) and B (up-projection)
    and rescales the result by ``alpha/rank``.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        # the original LoRA paper rescales by alpha / r
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x @ self.lora_a @ self.lora_b)


def _register_salora(block: nn.Module, in_proj: nn.Linear, salora_layer: nn.Module, attr_name: str):
    """Utility to (1) move the SALoRA layer to the same device/dtype as the parent
    linear layer, (2) register it as a sub-module of ``block`` (so its parameters
    are returned by ``model.parameters()`` and moved by ``model.to(device)``, and
    (3) patch the forward function of the target projection.
    """

    # 1.  Ensure the adapter lives on the same device / dtype as the original layer
    salora_layer = salora_layer.to(in_proj.weight.device, dtype=in_proj.weight.dtype)

    # 2.  Register – this is crucial so that the optimiser can see the new params
    setattr(block, attr_name, salora_layer)

    # 3.  Monkey-patch the forward of the *Linear* projection to add the adapter
    original_forward = in_proj.forward

    def patched_forward(x: torch.Tensor):  # noqa: D401 – simple wrapper
        return original_forward(x) + salora_layer(x)

    in_proj.forward = patched_forward


def add_salora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Inject SALoRA adapters into every Transformer block of a timm ViT model.

    The adapters are added to both:
      • the QKV projection (``block.attn.qkv``)
      • the MLP up-projection (``block.mlp.fc1``)

    The function returns the patched model for convenience.
    """
    for idx, block in enumerate(model.blocks):
        # ---- QKV adapter ----
        qkv = block.attn.qkv
        salora_qkv = SALoRALayer(qkv.in_features, qkv.out_features, rank=rank, alpha=alpha)
        _register_salora(block, qkv, salora_qkv, f"salora_qkv_{idx}")

        # ---- MLP adapter ----
        fc1 = block.mlp.fc1
        salora_fc1 = SALoRALayer(fc1.in_features, fc1.out_features, rank=rank, alpha=alpha)
        _register_salora(block, fc1, salora_fc1, f"salora_fc1_{idx}")

    print(f"Added SALoRA layers with rank={rank} to the model.")
    return model

# --- RL-TOP Actor-Critic Agent ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RLTOPAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def select_action(self, state, valid_action_count=None):
        """Samples an action.

        Args:
            state (np.ndarray | list | torch.Tensor): Current environment state.
            valid_action_count (int, optional): Number of admissible actions in
                the *current* decision step. This can be smaller than the
                actor's maximum ``action_dim`` when the task buffer shrinks.
        Returns:
            (action_idx, log_prob) – the sampled *local* action index (0 ≤ idx <
            ``valid_action_count``) and its log-probability under the current
            policy.  ``log_prob`` is a torch scalar **already on the correct
            device** so it can be used directly in loss computations.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)  # (1, state_dim)

        probs = self.actor(state_t).squeeze(0)  # (action_dim,)

        if valid_action_count is not None and valid_action_count < probs.shape[0]:
            probs = probs[:valid_action_count]
            probs = probs / probs.sum()  # re-normalise so that Σ = 1

        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        if next_state_t.dim() == 1:
            next_state_t = next_state_t.unsqueeze(0)

        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)

        # Critic update -------------------------------------------------------
        value = self.critic(state_t)
        next_value = self.critic(next_state_t)
        td_target = reward_t + self.gamma * next_value * (1 - done_t)
        advantage = td_target - value

        loss_critic = F.mse_loss(value, td_target.detach())
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Actor update --------------------------------------------------------
        loss_actor = -(log_prob * advantage.detach()).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

# --- Metric Calculation -------------------------------------------------------

def _get_gradients(model, data_loader, device):
    """Helper to compute gradients for a small batch."""
    model.train()
    images, labels, _ = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    model.zero_grad()
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def compute_gradient_interference(model, loader1, loader2, device):
    """Computes gradient interference G_ti = -cosSim(g_t, g_i)."""
    model_copy = copy.deepcopy(model).to(device)
    grad1 = _get_gradients(model_copy, loader1, device)
    grad2 = _get_gradients(model_copy, loader2, device)

    interference = -F.cosine_similarity(grad1, grad2, dim=0)
    return interference.item()


def compute_fisher_similarity(model, loader1, loader2, device):
    """
    Computes Task Similarity S_ti.
    Simplified proxy: cosine similarity of gradients on the backbone.
    A true Fisher implementation is more involved.
    """
    model_copy = copy.deepcopy(model).to(device)
    # Freeze the head for backbone gradients
    if hasattr(model_copy, 'head') and isinstance(model_copy.head, nn.Module):
        for param in model_copy.head.parameters():
            param.requires_grad = False

    grad1 = _get_gradients(model_copy, loader1, device)
    grad2 = _get_gradients(model_copy, loader2, device)

    # Unfreeze head
    if hasattr(model_copy, 'head') and isinstance(model_copy.head, nn.Module):
        for param in model_copy.head.parameters():
            param.requires_grad = True

    similarity = F.cosine_similarity(grad1, grad2, dim=0)
    return similarity.item()

# --- Main Training Function ---------------------------------------------------

def get_optimizer(model, config):
    if config['training']['optimizer'].lower() == 'adam8bit':
        return bnb.optim.Adam8bit(model.parameters(), lr=config['training']['learning_rate'])
    else:
        return optim.Adam(model.parameters(), lr=config['training']['learning_rate'])


def train_task(model, train_loader, optimizer, device, epochs=1):
    """Trains the model on a single task."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, _ in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
    return model
