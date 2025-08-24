import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import inject_noise

class SurrogateModel(nn.Module):
    """Lightweight surrogate model for adversarial benchmarking."""
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SelfVerificationModule(nn.Module):
    """Self-verification network for meta reinforcement learning."""
    def __init__(self, input_dim, output_dim):
        super(SelfVerificationModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_baseline_model(X_train, y_train):
    """Train baseline logistic regression model."""
    print("Training baseline model...")
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train, y_train)
    print("Baseline model training completed")
    return baseline_model

def train_adversarial_model(X_train, y_train, X_test, y_test, num_epochs=20, initial_noise=0.1):
    """Train model with dynamic adversarial benchmarking."""
    print("Starting dynamic adversarial training...")
    
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    
    input_dim = X_train.shape[1]
    surrogate_model = SurrogateModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=0.01)
    
    accuracies = []
    noise_level = initial_noise
    
    for epoch in range(num_epochs):
        y_noisy = inject_noise(y_train, noise_level)
        y_noisy_torch = torch.tensor(y_noisy, dtype=torch.long)
        
        surrogate_model.train()
        optimizer.zero_grad()
        outputs = surrogate_model(X_train_torch)
        loss = criterion(outputs, y_noisy_torch)
        loss.backward()
        optimizer.step()
        
        surrogate_model.eval()
        with torch.no_grad():
            test_outputs = surrogate_model(X_test_torch)
            _, y_pred = torch.max(test_outputs, 1)
            acc = (y_pred.numpy() == y_test).mean()
            accuracies.append(acc)
        
        if loss.item() > 0.8:
            noise_level = max(0.01, noise_level * 0.9)
        else:
            noise_level = min(0.3, noise_level * 1.1)
        
        print(f'Epoch {epoch+1:02d} - Loss: {loss.item():.3f} - Test Acc: {acc:.3f} - Noise Level: {noise_level:.3f}')
    
    print("Dynamic adversarial training completed")
    return surrogate_model, accuracies

def train_self_verification_model(input_dim=10, output_dim=2):
    """Train self-verification module."""
    print("Training self-verification module...")
    sv_network = SelfVerificationModule(input_dim, output_dim)
    optimizer_sv = optim.Adam(sv_network.parameters(), lr=0.01)
    print("Self-verification module initialized")
    return sv_network, optimizer_sv
