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

try:
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.storage_policy import ReservoirSamplingBuffer
    _has_avalanche = True
except ImportError:
    _has_avalanche = False
    ReplayPlugin = ReservoirSamplingBuffer = object

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
        self.embed_dim = self.backbone.embed_dim

        for param in self.backbone.parameters():
            param.requires_grad = False

        if _has_peft:
            peft_config = LoraConfig(
                r=config.salora_rank,
                lora_alpha=getattr(config, 'salora_alpha', config.salora_rank * 2),
                lora_dropout=getattr(config, 'salora_dropout', 0.1),
                bias="none",
                target_modules=["qkv", "proj", "fc1", "fc2"],
                task_type=TaskType.IMAGE_CLASSIFICATION
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            print("Applied SALoRA (LoRA) adapters.")
            self.backbone.print_trainable_parameters()

        self.head_bank = nn.ModuleDict()
        self.active_head = None

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
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
            print("Falling back to torch.optim.AdamW")
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

def train_one_epoch(model, loader, optimizer, scaler, device, precision, grad_clip, grad_accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (x, y, _) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        unique_labels = torch.unique(y)
        label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
        y = torch.tensor([label_map[int(l)] for l in y], device=device, dtype=torch.long)

        amp_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if precision == 'bf16' and device.type != 'cpu' else nullcontext()
        with amp_context:
            outputs = model(x)
            loss = F.cross_entropy(outputs, y) / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}"})
    return total_loss / len(loader)

def get_gradient_sketch(model, loader, device, precision):
    model.train()
    try:
        x, y, _ = next(iter(loader))
        x, y = x.to(device), y.to(device)
    except StopIteration:
        return torch.tensor([]).to(device)
    
    unique_labels = torch.unique(y)
    label_map = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
    y = torch.tensor([label_map[int(l)] for l in y], device=device, dtype=torch.long)

    amp_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if precision == 'bf16' and 'cuda' in str(device) else nullcontext()
    with amp_context:
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            grads.append(p.grad.view(-1).detach().clone())
    model.zero_grad()
    return torch.cat(grads)

def run_strategy(config, benchmark):
    device = torch.device(config.device)
    model = ContinualVisionModel(config).to(device)
    optimizer = get_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler() if config.precision == 'bf16' and device.type == 'cuda' else None
    metrics = ContinualMetrics(config.num_tasks)
    agent = A2CAgent(config, device) if 'rl_top' in config.policy else None

    task_order = list(range(config.num_tasks))
    if config.policy == 'random':
        random.shuffle(task_order)
    
    replay_plugin = None
    if config.policy == 'er_500':
        if not _has_avalanche: raise ImportError("Avalanche-lib needed for ER baseline")
        replay_plugin = ReplayPlugin(mem_size=config.er_buffer_size, storage_policy=ReservoirSamplingBuffer(max_size=config.er_buffer_size))

    all_tasks = benchmark.train_stream
    test_tasks = benchmark.test_stream
    task_buffer = {i: all_tasks[i] for i in range(config.num_tasks)}
    trained_task_indices = []

    start_time = time.time()

    for t in range(config.num_tasks):
        print(f"\n--- Starting stage {t+1}/{config.num_tasks} ---")
        
        candidate_indices = sorted(list(task_buffer.keys()))
        
        if 'rl_top' in config.policy or 'greedy' in config.policy:
            m = min(config.search_window_m, len(candidate_indices))
            search_indices = candidate_indices[:m]
            
            s_matrix = np.zeros((m, m))
            g_matrix = np.zeros((m, m))

            if m > 1:
                grads = []
                for idx in search_indices:
                    temp_model = copy.deepcopy(model)
                    temp_model.add_or_set_task_head(idx)
                    temp_model = temp_model.to(device)
                    loader = DataLoader(task_buffer[idx].dataset, batch_size=config.train_batch_size, shuffle=True)
                    grads.append(get_gradient_sketch(temp_model, loader, device, config.precision))
                    del temp_model
                    torch.cuda.empty_cache()
                
                for r in range(m):
                    for c in range(r + 1, m):
                        g1, g2 = grads[r], grads[c]
                        if config.policy != 'rl_top_g_only':
                            s_matrix[r, c] = s_matrix[c, r] = F.cosine_similarity(g1, g2, dim=0).item()
                        if config.policy != 'rl_top_s_only':
                            g_matrix[r, c] = g_matrix[c, r] = -F.cosine_similarity(g1, g2, dim=0).item()
            
            if config.policy == 'similarity_greedy':
                action = np.argmax(s_matrix[0, 1:]) + 1 if m > 1 else 0
            elif config.policy == 'greedy_g':
                action = np.argmin(np.mean(g_matrix, axis=1))
            elif 'rl_top' in config.policy:
                state = np.concatenate([s_matrix.flatten(), g_matrix.flatten(), np.zeros(config.search_window_m)])
                mask = torch.zeros(config.search_window_m, device=device)
                mask[:m] = 1.0
                action = agent.select_action(state, mask)
            else:
                action = 0
            
            current_task_idx = search_indices[action]
        else:
            current_task_idx = task_order[t]

        current_task = task_buffer.pop(current_task_idx)
        trained_task_indices.append(current_task_idx)
        print(f"Selected Task: {current_task_idx}")

        model.add_or_set_task_head(current_task_idx)
        model = model.to(device)
        
        if replay_plugin:
            train_dataset = ConcatDataset([current_task.dataset, replay_plugin.storage_policy.buffer])
        else:
            train_dataset = current_task.dataset

        train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        if hasattr(config, 'epochs_per_task'):
            for epoch in range(config.epochs_per_task):
                train_one_epoch(model, train_loader, optimizer, scaler, device, config.precision, config.clip_grad_norm, config.grad_accum_steps)
        elif hasattr(config, 'steps_per_task'):
            step = 0
            while step < config.steps_per_task:
                train_one_epoch(model, train_loader, optimizer, scaler, device, config.precision, config.clip_grad_norm, config.grad_accum_steps)
                step += len(train_loader)
        
        if replay_plugin: replay_plugin.after_training_exp(current_task)

        task_accuracies = []
        for i in trained_task_indices:
            model.add_or_set_task_head(i)
            test_loader = DataLoader(test_tasks[i].dataset, batch_size=config.eval_batch_size)
            acc = evaluate(model, test_loader, device, i, config.classes_per_task)
            task_accuracies.append(acc)
        
        full_accuracies = np.zeros(config.num_tasks)
        for i, idx in enumerate(trained_task_indices):
            full_accuracies[idx] = task_accuracies[i]
        metrics.update(t, full_accuracies)
        print(f"Accuracies on seen tasks: {[f'{a:.2f}' for a in task_accuracies]}")

        if agent: agent.update(rewards=[np.mean(task_accuracies)/100])
    
    final_aacc = metrics.average_accuracy()
    final_af = metrics.average_forgetting()
    total_time = time.time() - start_time
    
    if _has_fvcore and device.type == 'cuda':
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        flops = FlopCountAnalysis(model, dummy_input).total()
        vram = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        flops, vram = -1, -1

    print(f"\nFinished policy '{config.policy}' for seed {config.seed}.")
    print(f"  AACC: {final_aacc:.2f}% | AF: {final_af:.2f}% | Time: {total_time:.2f}s")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOPs | Peak VRAM: {vram:.2f} GB")

    return {'AACC': final_aacc, 'AF': final_af, 'time': total_time, 'vram': vram, 'flops': flops}
