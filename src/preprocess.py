#!/usr/bin/env python3
"""
Data preprocessing module for Adaptive MambaFormer experiments.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def create_synthetic_dataset(num_samples=1000, seq_len=32, d_model=256, 
                           complexity_ratio=0.2, noise_level=0.1):
    """
    Create synthetic dataset with varying complexity levels.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        d_model: Model dimension
        complexity_ratio: Ratio of high-complexity samples
        noise_level: Amount of noise to add
    
    Returns:
        TensorDataset with synthetic data
    """
    data = []
    labels = []  # 0 for simple, 1 for complex
    
    for i in range(num_samples):
        sequence = torch.randn(seq_len, d_model)
        
        is_complex = i < int(num_samples * complexity_ratio)
        
        if is_complex:
            pattern = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(1)
            pattern = pattern.expand(-1, d_model)
            sequence += 2.0 * pattern
            
            sequence[:5] += torch.randn(5, d_model) * 3.0
            
            for j in range(0, d_model-1, 2):
                sequence[:, j+1] = sequence[:, j] * 0.7 + torch.randn(seq_len) * 0.3
            
            labels.append(1)
        else:
            sequence += torch.randn(seq_len, d_model) * noise_level
            labels.append(0)
        
        data.append(sequence)
    
    data_tensor = torch.stack(data)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"Created synthetic dataset:")
    print(f"  Total samples: {num_samples}")
    print(f"  Complex samples: {sum(labels)} ({complexity_ratio*100:.1f}%)")
    print(f"  Simple samples: {num_samples - sum(labels)} ({(1-complexity_ratio)*100:.1f}%)")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    
    return TensorDataset(data_tensor, labels_tensor)


def create_nlp_like_dataset(num_samples=1000, seq_len=64, vocab_size=1000, 
                           d_model=256, embed_noise=0.1):
    """
    Create NLP-like synthetic dataset with token embeddings.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Embedding dimension
        embed_noise: Noise level in embeddings
    
    Returns:
        TensorDataset with NLP-like data
    """
    embedding_matrix = torch.randn(vocab_size, d_model)
    
    data = []
    complexity_labels = []
    
    for i in range(num_samples):
        tokens = torch.randint(0, vocab_size, (seq_len,))
        
        embeddings = embedding_matrix[tokens]
        
        embeddings += torch.randn_like(embeddings) * embed_noise
        
        is_complex = False
        
        for pattern_len in [3, 4, 5]:
            for start in range(seq_len - 2*pattern_len):
                pattern1 = tokens[start:start+pattern_len]
                pattern2 = tokens[start+pattern_len:start+2*pattern_len]
                if torch.equal(pattern1, pattern2):
                    is_complex = True
                    break
            if is_complex:
                break
        
        if i % 10 == 0:  # Every 10th sample
            is_complex = True
            embeddings[0] = embeddings[-1]  # First token = last token
            embeddings[seq_len//2] = embeddings[0]  # Middle token = first token
        
        complexity_labels.append(1 if is_complex else 0)
        data.append(embeddings)
    
    data_tensor = torch.stack(data)
    labels_tensor = torch.tensor(complexity_labels, dtype=torch.long)
    
    print(f"Created NLP-like dataset:")
    print(f"  Total samples: {num_samples}")
    print(f"  Complex samples: {sum(complexity_labels)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {d_model}")
    
    return TensorDataset(data_tensor, labels_tensor)


def create_time_series_dataset(num_samples=1000, seq_len=128, num_features=64,
                              trend_strength=0.5, seasonality_period=12):
    """
    Create time series-like dataset with trends and seasonality.
    
    Args:
        num_samples: Number of time series
        seq_len: Length of each time series
        num_features: Number of features per time step
        trend_strength: Strength of trend component
        seasonality_period: Period of seasonal component
    
    Returns:
        TensorDataset with time series data
    """
    data = []
    complexity_labels = []
    
    for i in range(num_samples):
        t = torch.linspace(0, 10, seq_len)
        
        base_series = torch.cumsum(torch.randn(seq_len, num_features) * 0.1, dim=0)
        
        trend = trend_strength * t.unsqueeze(1) * torch.randn(1, num_features)
        
        seasonal = 0.3 * torch.sin(2 * np.pi * t.unsqueeze(1) / seasonality_period)
        seasonal = seasonal.expand(-1, num_features)
        
        series = base_series + trend + seasonal
        
        variance = torch.var(series).item()
        trend_magnitude = torch.abs(trend[-1] - trend[0]).mean().item()
        
        is_complex = variance > 1.0 or trend_magnitude > 2.0
        
        if is_complex:
            series += torch.randn_like(series) * 0.2
        
        complexity_labels.append(1 if is_complex else 0)
        data.append(series)
    
    data_tensor = torch.stack(data)
    labels_tensor = torch.tensor(complexity_labels, dtype=torch.long)
    
    print(f"Created time series dataset:")
    print(f"  Total samples: {num_samples}")
    print(f"  Complex samples: {sum(complexity_labels)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of features: {num_features}")
    
    return TensorDataset(data_tensor, labels_tensor)


def normalize_dataset(dataset, method='standard'):
    """
    Normalize dataset using specified method.
    
    Args:
        dataset: TensorDataset to normalize
        method: Normalization method ('standard', 'minmax', 'none')
    
    Returns:
        Normalized TensorDataset and scaler object
    """
    data_tensor = dataset.tensors[0]
    labels_tensor = dataset.tensors[1] if len(dataset.tensors) > 1 else None
    
    if method == 'none':
        return dataset, None
    
    original_shape = data_tensor.shape
    data_reshaped = data_tensor.view(-1, original_shape[-1])
    
    if method == 'standard':
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_reshaped.numpy())
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_reshaped.numpy())
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    data_normalized = torch.tensor(data_normalized, dtype=torch.float32)
    data_normalized = data_normalized.view(original_shape)
    
    if labels_tensor is not None:
        normalized_dataset = TensorDataset(data_normalized, labels_tensor)
    else:
        normalized_dataset = TensorDataset(data_normalized)
    
    print(f"Applied {method} normalization to dataset")
    
    return normalized_dataset, scaler


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: TensorDataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_dataset)} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} samples ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                      batch_size=32, num_workers=0, shuffle_train=True):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Created DataLoaders:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def visualize_dataset_samples(dataset, num_samples=4, save_path=None):
    """
    Visualize samples from the dataset.
    
    Args:
        dataset: TensorDataset to visualize
        num_samples: Number of samples to show
        save_path: Path to save the visualization
    """
    data_tensor = dataset.tensors[0]
    labels_tensor = dataset.tensors[1] if len(dataset.tensors) > 1 else None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(data_tensor))):
        sample = data_tensor[i]  # Shape: (seq_len, d_model)
        label = labels_tensor[i].item() if labels_tensor is not None else "N/A"
        
        axes[i].plot(sample[:, :5].numpy())
        axes[i].set_title(f'Sample {i+1} (Label: {label})')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Dataset visualization saved to {save_path}")
    
    plt.close()


def analyze_dataset_statistics(dataset):
    """
    Analyze and print dataset statistics.
    
    Args:
        dataset: TensorDataset to analyze
    
    Returns:
        Dictionary with dataset statistics
    """
    data_tensor = dataset.tensors[0]
    labels_tensor = dataset.tensors[1] if len(dataset.tensors) > 1 else None
    
    stats = {
        'num_samples': len(data_tensor),
        'sequence_length': data_tensor.shape[1],
        'feature_dimension': data_tensor.shape[2],
        'data_mean': torch.mean(data_tensor).item(),
        'data_std': torch.std(data_tensor).item(),
        'data_min': torch.min(data_tensor).item(),
        'data_max': torch.max(data_tensor).item(),
    }
    
    if labels_tensor is not None:
        unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
        stats['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
    
    print(f"\nDataset Statistics:")
    print(f"  Number of samples: {stats['num_samples']}")
    print(f"  Sequence length: {stats['sequence_length']}")
    print(f"  Feature dimension: {stats['feature_dimension']}")
    print(f"  Data range: [{stats['data_min']:.3f}, {stats['data_max']:.3f}]")
    print(f"  Data mean: {stats['data_mean']:.3f}")
    print(f"  Data std: {stats['data_std']:.3f}")
    
    if 'label_distribution' in stats:
        print(f"  Label distribution: {stats['label_distribution']}")
    
    return stats
