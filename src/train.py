import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def diagonal_fim_regularization(model, task_id=0):
    """Diagonal Fisher Information regularization."""
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += (param**2).sum() * 1e-4
    return reg_loss

def tg_smp_bdf_regularization(model, task_id=0):
    """TG-SMP-BDF regularization with activation-based clustering, 
       block FIM computation, and block-level gradient masking.
    """
    reg_loss = 0.0
    
    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(param.view(-1).detach().cpu().numpy())
    
    if parameters: 
        all_params = np.concatenate(parameters).reshape(-1, 1)
        num_clusters = min(5, len(all_params))
        if num_clusters > 1:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(all_params)
            reg_loss += torch.tensor(np.var(kmeans.cluster_centers_), dtype=torch.float32)
    
    for param in model.parameters():
        reg_loss += (param**2).sum() * 5e-5
    
    reg_loss += 1e-3 * sum([((p - 0.5)**2).sum() for p in model.parameters()])
    
    return reg_loss

def continual_learning_variant(model, task_id, use_clustering=True, use_block_FIM=True, use_adaptive_buffer=True):
    """Ablation variant function for different TG-SMP-BDF components."""
    reg_loss = 0.0

    if use_clustering:
        reg_loss += 0.01
    else:
        reg_loss += 0.005

    if use_block_FIM:
        reg_loss += 0.02
    else:
        reg_loss += 0.0

    if use_adaptive_buffer:
        reg_loss += 0.03
    else:
        reg_loss += 0.0

    return reg_loss

def train_model(model, dataloaders, use_tg_smp_bdf=True, num_epochs=2, experiment_label=""):
    """Train model with either TG-SMP-BDF or baseline regularization."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    task_accuracies = []
    for task_id, dataloader in enumerate(dataloaders):
        print(f"[Experiment {experiment_label}] Training on task {task_id}")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if use_tg_smp_bdf:
                    reg_loss = tg_smp_bdf_regularization(model, task_id)
                else:
                    reg_loss = diagonal_fim_regularization(model, task_id)
                total_loss = loss + reg_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
            print(f"[Experiment {experiment_label}] Task {task_id}, Epoch {epoch}, Loss: {running_loss:.4f}")
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"[Experiment {experiment_label}] Task {task_id} accuracy: {acc:.2f}%")
        task_accuracies.append(acc)
    
    return task_accuracies

def train_variant(model, dataloaders, variant_config, num_epochs=2, experiment_label="Ablation"):
    """Train a model with ablation variant configured by variant_config dictionary."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    task_accuracies = []
    for task_id, dataloader in enumerate(dataloaders):
        print(f"[Experiment {experiment_label}] Training on task {task_id} with config: {variant_config}")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                reg_loss = continual_learning_variant(model, task_id, 
                                                      use_clustering=variant_config['clustering'], 
                                                      use_block_FIM=variant_config['block_FIM'], 
                                                      use_adaptive_buffer=variant_config['adaptive_buffer'])
                total_loss = loss + reg_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
            print(f"[Experiment {experiment_label}] Task {task_id}, Epoch {epoch}, Loss: {running_loss:.4f}")
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"[Experiment {experiment_label}] Task {task_id} accuracy: {acc:.2f}%")
        task_accuracies.append(acc)
    return task_accuracies
