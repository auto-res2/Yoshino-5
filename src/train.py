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
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank

    def forward(self, x):
        return self.scale * (x @ self.lora_a @ self.lora_b)

def add_salora_to_model(model, rank=8):
    """
    Injects SALoRA layers into the QKV and MLP up-projections of a ViT model.
    """
    for block in model.blocks:
        # Target QKV linear layer
        qkv = block.attn.qkv
        salora_qkv = SALoRALayer(qkv.in_features, qkv.out_features, rank=rank)
        # This is a simplified monkey-patch. A real implementation would use hooks.
        original_qkv_forward = qkv.forward
        qkv.forward = lambda x: original_qkv_forward(x) + salora_qkv(x)

        # Target MLP up-projection
        fc1 = block.mlp.fc1
        salora_mlp = SALoRALayer(fc1.in_features, fc1.out_features, rank=rank)
        original_fc1_forward = fc1.forward
        fc1.forward = lambda x: original_fc1_forward(x) + salora_mlp(x)
    
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

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        
        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value
        
        loss_critic = F.mse_loss(value, td_target.detach())
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Actor update
        loss_actor = -log_prob * advantage.detach()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

# --- Metric Calculation ---
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
    model_copy = copy.deepcopy(model)
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
    model_copy = copy.deepcopy(model)
    # Freeze the head for backbone gradients
    if hasattr(model_copy, 'head'):
        for param in model_copy.head.parameters():
            param.requires_grad = False
    
    grad1 = _get_gradients(model_copy, loader1, device)
    grad2 = _get_gradients(model_copy, loader2, device)

    # Unfreeze head
    if hasattr(model_copy, 'head'):
        for param in model_copy.head.parameters():
            param.requires_grad = True
            
    similarity = F.cosine_similarity(grad1, grad2, dim=0)
    return similarity.item()

# --- Main Training Function ---
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
