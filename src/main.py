import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import timm
import random

# NOTE:
# The original code used relative imports ("from . import module") which only
# work when the package is executed with the "-m" flag (e.g. ``python -m src.main``).
# In the execution environment for these tests the file is run directly as a
# script (``python src/main.py``), therefore there is **no parent package** and
# the relative import fails with:
#   ImportError: attempted relative import with no known parent package
#
# Switching to standard absolute-style imports (which fall back to looking in
# the current directory that already contains the modules) fixes the problem
# without requiring any change to how the script is launched.
import preprocess
import train
import evaluate


def main():
    # --- 1. Setup ---
    config_path = 'config/config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config['paths']['image_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)

    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --- 2. Data Loading ---
    train_task_datasets, test_task_datasets = preprocess.get_datasets(
        name=config['dataset']['name'],
        num_tasks=config['dataset']['num_tasks'],
        data_path=config['dataset']['path']
    )
    test_loaders = [DataLoader(d, batch_size=config['training']['batch_size']) for d in test_task_datasets]

    # --- 3. Model & Agent Initialization ---
    model = timm.create_model(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    ).to(device)

    model = train.add_salora_to_model(model, rank=config['peft']['rank'])
    optimizer = train.get_optimizer(model, config)

    m = config['rl_agent']['search_window_m']
    state_dim = m * m * 2 + m  # (S_ti + G_ti + accuracies)
    agent = train.RLTOPAgent(
        state_dim=state_dim,
        action_dim=m,
        actor_lr=config['rl_agent']['actor_lr'],
        critic_lr=config['rl_agent']['critic_lr'],
        gamma=config['rl_agent']['gamma'],
        device=device
    )

    # --- 4. Continual Learning Loop ---
    num_tasks = config['dataset']['num_tasks']
    task_buffer = list(range(num_tasks))
    seen_tasks = []

    results = {'aacc': [], 'af': []}
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    current_state = np.zeros(state_dim)

    for i in range(num_tasks):
        print(f"\n--- Starting Iteration {i+1}/{num_tasks} ---")

        # Define candidate tasks from the buffer
        candidate_tasks = task_buffer[:min(m, len(task_buffer))]

        if len(candidate_tasks) > 1:
            # --- a. State Construction ---
            s_matrix = np.zeros((m, m))
            g_matrix = np.zeros((m, m))
            accs = np.zeros(m)

            # Create small loaders for metric calculation
            metric_loaders = [DataLoader(train_task_datasets[t_idx], batch_size=32) for t_idx in candidate_tasks]

            for r in range(len(candidate_tasks)):
                for c in range(r, len(candidate_tasks)):
                    if r == c:
                        continue
                    sim = train.compute_fisher_similarity(model, metric_loaders[r], metric_loaders[c], device)
                    inter = train.compute_gradient_interference(model, metric_loaders[r], metric_loaders[c], device)
                    s_matrix[r, c] = s_matrix[c, r] = sim
                    g_matrix[r, c] = g_matrix[c, r] = inter

            # Use last known accuracy on candidate tasks if seen
            for k, task_idx in enumerate(candidate_tasks):
                if task_idx in seen_tasks:
                    accs[k] = accuracy_matrix[i-1, task_idx]

            # Flatten matrices and combine to form state
            current_state = np.concatenate([s_matrix.flatten(), g_matrix.flatten(), accs])

            # --- b. Action Selection ---
            action_idx, log_prob = agent.select_action(current_state)
            chosen_task_local_idx = action_idx
        else:
            chosen_task_local_idx = 0
            log_prob = None  # No RL decision needed when only one task remains

        # Get global task index and move from buffer to seen
        chosen_task_id = candidate_tasks[chosen_task_local_idx]
        task_buffer.remove(chosen_task_id)
        seen_tasks.append(chosen_task_id)

        print(f"Selected Task: {chosen_task_id}")

        # --- c. Train on Chosen Task ---
        train_loader = DataLoader(
            train_task_datasets[chosen_task_id],
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        model = train.train_task(model, train_loader, optimizer, device, epochs=config['training']['epochs_per_task'])

        # --- d. Evaluate ---
        current_accuracies = evaluate.evaluate_model(model, test_loaders, list(range(num_tasks)), device)
        accuracy_matrix[i, :] = current_accuracies

        # --- e. Reward and Agent Update ---
        if i > 0 and log_prob is not None:
            prev_aacc, prev_af = results['aacc'][-1], results['af'][-1]
            current_acc_submatrix = accuracy_matrix[:i+1, :i+1]
            aacc, af = evaluate.calculate_metrics(current_acc_submatrix)

            delta_acc = aacc - prev_aacc
            delta_forget = -(af - prev_af)  # Negative forgetting is good
            reward = float(delta_acc + delta_forget)

            # Simplified next_state placeholder
            next_state = np.zeros(state_dim)
            agent.update(current_state, log_prob, reward, next_state, done=(i == num_tasks - 1))
            print(f"RL Agent updated with reward: {reward:.4f}")

        # --- f. Log Results ---
        final_aacc, final_af = evaluate.calculate_metrics(accuracy_matrix[:i+1, :i+1])
        results['aacc'].append(final_aacc)
        results['af'].append(final_af)
        print(f"After task {chosen_task_id}: AACC = {final_aacc:.2f}%, AF = {final_af:.2f}%")

    # --- 5. Finalization ---
    print("\n--- Continual Learning Finished ---")
    print(f"Final AACC: {results['aacc'][-1]:.2f}%")
    print(f"Final AF: {results['af'][-1]:.2f}%")

    evaluate.plot_results(results, config['paths']['image_dir'])

    final_model_path = os.path.join(config['paths']['model_dir'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
