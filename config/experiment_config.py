import torch

SEED = 20250727
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "vocab_size": 1000,
    "hidden_dim": 128,
    "num_layers": 4,
    "lora_rank": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}

TRAINING_CONFIG = {
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_steps": 500
}

LOSS_CONFIG = {
    "lambda_kl": 1.0,
    "beta_sup": 2.0,
    "gamma_nce": 0.1,
    "temperature": 0.07
}

SYNTHETIC_CONFIG = {
    "num_samples": 64,
    "min_seq_len": 8,
    "max_seq_len": 16,
    "trace_dim": 64,
    "mask_dim": 128,
    "vocab_size": 1000
}
