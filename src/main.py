import sys
import yaml
import torch
import numpy as np
import random
import gc
from collections import defaultdict, deque
from types import SimpleNamespace
from pathlib import Path
from torch.utils.data import DataLoader

from .preprocess import get_benchmark
from .train import get_model, get_optimizer, train_one_task, A2CAgent
from .evaluate import evaluate_on_all_tasks, ContinualMetrics, aggregate_results, plot_results, save_results_csv

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> SimpleNamespace:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def run_single_experiment(config):
    """Runs a single experiment for a given configuration."""
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    model = get_model(config).to(device)
    model.apply_salora_adapters()
    optimizer = get_optimizer(model, config)
    metrics = ContinualMetrics(config.num_tasks)
    agent = A2CAgent(config, device) if config.policy == 'rl_top' else None
    benchmark, _ = get_benchmark(config)

    task_order = list(range(config.num_tasks))
    if config.policy == 'random':
        np.random.shuffle(task_order)
    
    pending_tasks = deque(task_order)
    task_buffer = []

    for t_idx in range(config.num_tasks):
        print(f'\n--- Starting CL Stage {t_idx+1}/{config.num_tasks} ---')
        current_task_id = -1

        if config.policy in ['rl_top', 'similarity_greedy']:
            while len(task_buffer) < config.search_window_m and pending_tasks:
                task_buffer.append(pending_tasks.popleft())
            
            if config.policy == 'rl_top' and agent and task_buffer:
                # Placeholder state for RL agent
                state = np.random.rand(agent.state_dim)
                action_idx = agent.select_action(state, [True]*len(task_buffer))
                current_task_id = task_buffer.pop(action_idx)
            else: # Fallback for greedy or if buffer is empty
                current_task_id = task_buffer.pop(0) if task_buffer else pending_tasks.popleft()
        else: # Random or chronological
            current_task_id = task_order[t_idx]

        print(f'Training on Task ID: {current_task_id}')
        train_experience = benchmark.train_stream[current_task_id]
        train_loader = DataLoader(train_experience.dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        model.set_active_task(current_task_id)
        if optimizer is not None:
            train_one_task(model, train_loader, optimizer, device, config)

        accuracies = evaluate_on_all_tasks(model, benchmark.test_stream[:t_idx+1], device)
        metrics.update(t_idx, accuracies)
        print(f'After Task {current_task_id}, Accuracies: {[f"{acc:.2f}" for acc in accuracies]}')

        if agent:
            prev_accs = metrics.accuracy_matrix[t_idx-1, :t_idx] if t_idx > 0 else [0]
            curr_accs = metrics.accuracy_matrix[t_idx, :t_idx] if t_idx > 0 else [0]
            reward = np.mean(curr_accs) - np.mean(prev_accs)
            agent.store_reward(reward)
            if (t_idx+1) % 2 == 0: agent.update()

    final_metrics = metrics.final_metrics()
    print(f'\nFinal Metrics for Policy "{config.policy}" (Seed {config.seed}):')
    for key, value in final_metrics.items():
        print(f'  {key}: {value:.2f}%')

    del model, optimizer, agent, benchmark
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return final_metrics

def main():
    config = load_config('../config/config.yaml')
    print(f'# Running Experiment: {config.experiment_name}')
    
    # Define experiment parameters
    policies_to_run = ['random', 'rl_top']
    seeds_to_run = [12, 35, 71]
    
    all_results = defaultdict(list)
    raw_results_list = []

    for policy in policies_to_run:
        config.policy = policy
        for seed in seeds_to_run:
            print(f'\n>>> Running Policy: {policy}, Seed: {seed} <<<')
            config.seed = seed
            set_seed(seed)
            
            try:
                results = run_single_experiment(config)
                all_results[policy].append(results)
                raw_results_list.append({'policy': policy, 'seed': seed, **results})
            except Exception as e:
                print(f"ERROR during run with policy {policy}, seed {seed}: {e}")
                # Add error logging to results
                error_results = {'AACC': 0, 'AF': 100} 
                all_results[policy].append(error_results)
                raw_results_list.append({'policy': policy, 'seed': seed, **error_results, 'error': str(e)})


    summary_df = aggregate_results(all_results, policies_to_run)
    plot_results(summary_df, config.experiment_code)
    save_results_csv(raw_results_list, summary_df, config.experiment_code)

if __name__ == '__main__':
    main()