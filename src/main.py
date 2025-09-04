import os
import json
import time
import random
from typing import Dict, Any, List

import yaml
import torch

from .preprocess import set_seed, build_e1_tasks_vision, build_e3_tasks_text, prepare_output_dirs
from .train import (
    TinyCNN, TinyTextTransformer, get_trainable_params, ContinualTrainer, DyCLOSScheduler,
    compute_signals_for_new_task_vision, compute_signals_for_new_task_text, run_baseline_fixed_order
)
from .evaluate import (
    plot_training_loss, plot_accuracy_curves, plot_schedule_costs, plot_confusion_matrix, summarize_results
)


def main(config: Dict[str, Any]):
    set_seed(config.get('seed', 0))
    out_dir = prepare_output_dirs()
    print('Output directory for images:', out_dir)

    experiment = config.get('experiment', 'vision_e1')

    if experiment == 'vision_e1':
        # Data
        task_ids, task_dls = build_e1_tasks_vision(
            num_tasks=config.get('num_tasks', 6),
            probe_size=config.get('probe_size', 256),
            batch_size=config.get('batch_size', 64),
            use_fake_data=config.get('use_fake_data', True),
            fast_test=config.get('fast_test', True),
            seed=config.get('seed', 0),
        )
        arrival_order = task_ids.copy()
        random.shuffle(arrival_order)
        print('Arrival order:', arrival_order)
        # Model + trainer + scheduler
        model = TinyCNN(in_ch=3, num_classes=10, lora_r=config.get('lora_r', 8))
        optimizer = torch.optim.Adam(get_trainable_params(model), lr=config.get('lr', 1e-3), weight_decay=0.01)
        trainer = ContinualTrainer(model, optimizer, torch.nn.CrossEntropyLoss(), task_dls)
        scheduler = DyCLOSScheduler(alpha=config.get('alpha', 1.0), beta=config.get('beta', 1.0),
                                    window_k=config.get('window_k', 3), beam_size=config.get('beam_size', 4))
        # DyCLOS training
        probe_t0 = time.perf_counter()
        committed_count = 0
        for new_task in arrival_order:
            peers = scheduler.trained_prefix + scheduler.untrained_queue
            compute_signals_for_new_task_vision(model, new_task, peers, task_dls, scheduler,
                                                max_probe=config.get('probe_size', 256),
                                                max_batches_grad=config.get('max_batches_grad', 4))
            scheduler.insert_task(new_task)
            while len(scheduler.trained_prefix) > committed_count:
                t_to_train = scheduler.trained_prefix[committed_count]
                trainer.train_task(t_to_train, epochs=config.get('epochs_per_task', 2), base_lr=config.get('lr', 1e-3))
                committed_count += 1
        probe_t1 = time.perf_counter()
        omega_acc, omega_forg = trainer.compute_metrics()
        print(f"[DyCLOS] Final order: {scheduler.trained_prefix}")
        print(f"[DyCLOS] Omega-acc={omega_acc:.4f}, Omega-forg={omega_forg:.4f}")
        # Baseline
        base_model = TinyCNN(in_ch=3, num_classes=10, lora_r=config.get('lora_r', 8))
        base_trainer, base_oa, base_of, base_wall = run_baseline_fixed_order(
            base_model, arrival_order, task_dls, epochs_per_task=config.get('epochs_per_task', 2), lr=config.get('lr', 1e-3)
        )
        # Plots
        plot_training_loss(trainer.per_epoch_train_loss, out_dir, title='Training Loss per Task (DyCLOS)', filename='training_loss_dyclos.pdf')
        plot_accuracy_curves(trainer.acc_hist, out_dir, title='Per-task Accuracy Over Time (DyCLOS)', filename='accuracy_dyclos.pdf')
        plot_accuracy_curves(base_trainer.acc_hist, out_dir, title='Per-task Accuracy Over Time (Baseline)', filename='accuracy_baseline.pdf')
        plot_schedule_costs(scheduler, out_dir, title='DyCLOS Schedule Edge Costs', filename='schedule_costs_dyclos.pdf')
        # Confusion on a subset of test data (up to 512 samples for speed)
        # Build a small combined test loader
        combined_samples = []
        max_collect = 512
        for t in trainer.seen_tasks:
            ds = task_dls[t]['test'].dataset
            # Safely draw up to max_collect // len(seen_tasks)
            per_t = max(1, max_collect // max(1, len(trainer.seen_tasks)))
            idxs = list(range(min(per_t, len(ds))))
            for i in idxs:
                combined_samples.append(ds[i])
        if combined_samples:
            from torch.utils.data import DataLoader
            test_dl = DataLoader(combined_samples, batch_size=config.get('batch_size', 64), shuffle=False)
            plot_confusion_matrix(trainer.model, test_dl, n_classes=10, device=None, out_dir=out_dir,
                                  title='Confusion Matrix (DyCLOS)', filename='confusion_matrix_dyclos.pdf')
        results = {
            'arrival_order': arrival_order,
            'dyclos_order': scheduler.trained_prefix,
            'dyclos': {'Omega-acc': omega_acc, 'Omega-forg': omega_forg, 'probe+schedule_time_s': probe_t1 - probe_t0},
            'baseline': {'Omega-acc': base_oa, 'Omega-forg': base_of},
        }
        print('Experiment results:')
        print(json.dumps(results, indent=2))
        print(summarize_results(results))

    elif experiment == 'text_e3':
        # Data
        task_ids, task_dls = build_e3_tasks_text(
            num_tasks=config.get('num_tasks', 4), probe_size=config.get('probe_size', 256), batch_size=config.get('batch_size', 64),
            vocab_size=config.get('vocab_size', 5000), seq_len=config.get('seq_len', 64), fast_test=config.get('fast_test', True),
            seed=config.get('seed', 0)
        )
        arrival_order = task_ids.copy()
        random.shuffle(arrival_order)
        print('Arrival order:', arrival_order)
        # Model
        model = TinyTextTransformer(vocab_size=config.get('vocab_size', 5000), d_model=128, nhead=4, num_layers=1,
                                    num_classes=2, max_len=config.get('seq_len', 64), lora_r=config.get('lora_r', 8))
        optimizer = torch.optim.Adam(get_trainable_params(model), lr=config.get('lr', 2e-3), weight_decay=0.01)
        trainer = ContinualTrainer(model, optimizer, torch.nn.CrossEntropyLoss(), task_dls)
        scheduler = DyCLOSScheduler(alpha=config.get('alpha', 1.0), beta=config.get('beta', 1.0),
                                    window_k=config.get('window_k', 3), beam_size=config.get('beam_size', 4))
        # DyCLOS training
        probe_t0 = time.perf_counter()
        committed = 0
        for new_task in arrival_order:
            peers = scheduler.trained_prefix + scheduler.untrained_queue
            compute_signals_for_new_task_text(model, new_task, peers, task_dls, scheduler,
                                              max_probe=config.get('probe_size', 256), max_batches_grad=config.get('max_batches_grad', 4))
            scheduler.insert_task(new_task)
            while len(scheduler.trained_prefix) > committed:
                t_to_train = scheduler.trained_prefix[committed]
                trainer.train_task(t_to_train, epochs=config.get('epochs_per_task', 2), base_lr=config.get('lr', 2e-3))
                committed += 1
        probe_t1 = time.perf_counter()
        omega_acc, omega_forg = trainer.compute_metrics()
        print(f"[DyCLOS NLP] Order: {scheduler.trained_prefix}")
        print(f"[DyCLOS NLP] Omega-acc={omega_acc:.4f}, Omega-forg={omega_forg:.4f}")
        # Baseline
        base_model = TinyTextTransformer(vocab_size=config.get('vocab_size', 5000), d_model=128, nhead=4, num_layers=1,
                                         num_classes=2, max_len=config.get('seq_len', 64), lora_r=config.get('lora_r', 8))
        base_trainer, base_oa, base_of, base_wall = run_baseline_fixed_order(
            base_model, arrival_order, task_dls, epochs_per_task=config.get('epochs_per_task', 2), lr=config.get('lr', 2e-3)
        )
        # Plots
        plot_training_loss(trainer.per_epoch_train_loss, out_dir, title='Training Loss per Task (DyCLOS NLP)', filename='training_loss_dyclos_nlp.pdf')
        plot_accuracy_curves(trainer.acc_hist, out_dir, title='Per-task Accuracy Over Time (DyCLOS NLP)', filename='accuracy_dyclos_nlp.pdf')
        plot_accuracy_curves(base_trainer.acc_hist, out_dir, title='Per-task Accuracy Over Time (Baseline NLP)', filename='accuracy_baseline_nlp.pdf')
        plot_schedule_costs(scheduler, out_dir, title='DyCLOS Schedule Edge Costs (NLP)', filename='schedule_costs_dyclos_nlp.pdf')
        results = {
            'arrival_order': arrival_order,
            'dyclos_order': scheduler.trained_prefix,
            'dyclos': {'Omega-acc': omega_acc, 'Omega-forg': omega_forg, 'probe+schedule_time_s': probe_t1 - probe_t0},
            'baseline': {'Omega-acc': base_oa, 'Omega-forg': base_of},
        }
        print('Experiment results:')
        print(json.dumps(results, indent=2))
        print(summarize_results(results))

    else:
        raise ValueError(f"Unknown experiment type: {experiment}")


if __name__ == '__main__':
    # Load config from config/config.yaml
    cfg_path = os.path.join('config', 'config.yaml')
    if not os.path.exists(cfg_path):
        # Minimal default config for a quick test
        cfg = {
            'seed': 123,
            'experiment': 'vision_e1',
            'num_tasks': 4,
            'probe_size': 128,
            'batch_size': 32,
            'use_fake_data': True,
            'fast_test': True,
            'lora_r': 4,
            'lr': 1e-3,
            'epochs_per_task': 1,
            'alpha': 1.0,
            'beta': 1.0,
            'window_k': 3,
            'beam_size': 4,
            'max_batches_grad': 2,
        }
    else:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    main(cfg)
