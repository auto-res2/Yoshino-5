import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DummyFewShotDataset(Dataset):
    """
    Dummy dataset for few-shot learning experiments with vision and text modalities.
    Simulates features from EfficientNet (vision) and DistilBERT (text).
    """
    def __init__(self, num_samples=100, vision_dim=2048, text_dim=768, num_classes=5):
        self.num_samples = num_samples
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        
        self.vision_data = torch.randn(num_samples, vision_dim)
        self.text_data = torch.randn(num_samples, text_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.vision_data[idx], self.text_data[idx], self.labels[idx]

def create_few_shot_dataloader(batch_size=16, num_samples=100):
    """Create a DataLoader for few-shot learning experiments."""
    dataset = DummyFewShotDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def add_modality_conflict(vision, text, conflict_level=0.8, conflict_prob=0.5):
    """
    Add conflicting signals between modalities to test robustness.
    
    Args:
        vision: Vision features tensor
        text: Text features tensor
        conflict_level: Strength of noise to add
        conflict_prob: Probability of adding conflict to each sample
    
    Returns:
        Tuple of (vision_noisy, text_noisy)
    """
    batch_size = text.size(0)
    
    conflict_mask = (torch.rand(batch_size, 1) < conflict_prob).float()
    
    noise = conflict_level * torch.randn_like(text)
    text_noisy = text + conflict_mask * noise
    
    vision_noisy = vision
    
    return vision_noisy, text_noisy
