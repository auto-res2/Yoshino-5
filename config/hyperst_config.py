"""
Configuration parameters for HyperST-LoRA experiments.
"""

MODEL_CONFIG = {
    "max_rank": 16,
    "keep_ratio": 0.1,
    "hidden_dim": 128,
    "use_spectral_gate": True,
    "use_hyper_rank": True,
    "use_token_prune": True,
    "use_dora_sign": True
}

TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "warmup_steps": 1000,
    "gradient_accumulation": 8
}

EXPERIMENT_CONFIG = {
    "dataset_sizes": {"exp1": 240, "exp2": 120, "exp3": 160},
    "budgets": [20e6, 40e6, 60e6, 80e6, 100e6],
    "seeds": [42, 43, 44],
    "output_dir": ".research/iteration1/images"
}
