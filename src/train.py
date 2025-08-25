import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DummyVisionEncoder(nn.Module):
    """Dummy vision encoder simulating EfficientNet feature extraction."""
    def __init__(self, input_dim=2048, output_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DummyTextEncoder(nn.Module):
    """Dummy text encoder simulating DistilBERT feature extraction."""
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class TaskAwareUncertaintyEstimation(nn.Module):
    """
    Bayesian uncertainty estimation module for task-aware modality weighting.
    Uses dropout-based uncertainty estimation.
    """
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        self.dropout_rate = dropout_rate
    
    def forward(self, features):
        self.train()
        uncertainty = self.uncertainty_net(features)
        return uncertainty

class DynamicCrossModalGating(nn.Module):
    """
    Dynamic cross-modal gating mechanism with contrastive alignment.
    Adaptively weights modality contributions based on uncertainty estimates.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.Sigmoid()
        )
        
        self.contrastive_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, vision_feat, text_feat, uncert_vision, uncert_text):
        batch_size = vision_feat.size(0)
        
        vision_modulated = vision_feat * uncert_vision
        text_modulated = text_feat * uncert_text
        
        vision_attended, _ = self.cross_attention(
            vision_modulated.unsqueeze(1), 
            text_modulated.unsqueeze(1), 
            text_modulated.unsqueeze(1)
        )
        text_attended, _ = self.cross_attention(
            text_modulated.unsqueeze(1), 
            vision_modulated.unsqueeze(1), 
            vision_modulated.unsqueeze(1)
        )
        
        vision_attended = vision_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)
        
        combined_features = torch.cat([vision_attended, text_attended], dim=-1)
        gate_weights = self.gate_network(combined_features)
        
        vision_gate, text_gate = torch.chunk(gate_weights, 2, dim=-1)
        
        vision_gated = vision_attended * vision_gate
        text_gated = text_attended * text_gate
        
        fused_features = vision_gated + text_gated
        
        return fused_features, vision_gated, text_gated

class DCAMFLPP(nn.Module):
    """
    Dynamic Contrasting Adaptation for Multimodal Few-Shot Learning++ (DCAMFL++)
    
    Main components:
    1. Task-Aware Uncertainty Estimation
    2. Dynamic Cross-Modal Gating with Contrastive Alignment
    3. Hierarchical Meta-Learning Strategy
    4. Cross-Modal Feature Disentanglement
    """
    def __init__(self, vision_dim=2048, text_dim=768, feature_dim=128, num_classes=5):
        super().__init__()
        
        self.vision_encoder = DummyVisionEncoder(vision_dim, feature_dim)
        self.text_encoder = DummyTextEncoder(text_dim, feature_dim)
        
        self.uncertainty_vision = TaskAwareUncertaintyEstimation(feature_dim)
        self.uncertainty_text = TaskAwareUncertaintyEstimation(feature_dim)
        
        self.gating = DynamicCrossModalGating(feature_dim)
        
        self.shared_proj = nn.Linear(feature_dim, feature_dim // 2)
        self.vision_specific_proj = nn.Linear(feature_dim, feature_dim // 2)
        self.text_specific_proj = nn.Linear(feature_dim, feature_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, vision_input, text_input):
        vision_feat = self.vision_encoder(vision_input)
        text_feat = self.text_encoder(text_input)
        
        uncert_vision = self.uncertainty_vision(vision_feat)
        uncert_text = self.uncertainty_text(text_feat)
        
        fused_features, vision_gated, text_gated = self.gating(
            vision_feat, text_feat, uncert_vision, uncert_text
        )
        
        shared_vision = self.shared_proj(vision_feat)
        shared_text = self.shared_proj(text_feat)
        specific_vision = self.vision_specific_proj(vision_feat)
        specific_text = self.text_specific_proj(text_feat)
        
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'fused_features': fused_features,
            'vision_gated': vision_gated,
            'text_gated': text_gated,
            'uncertainty_vision': uncert_vision,
            'uncertainty_text': uncert_text,
            'shared_vision': shared_vision,
            'shared_text': shared_text,
            'specific_vision': specific_vision,
            'specific_text': specific_text
        }

def compute_contrastive_alignment_loss(vision_feat, text_feat, temperature=0.1):
    """Compute contrastive alignment loss between modalities."""
    batch_size = vision_feat.size(0)
    
    vision_norm = nn.functional.normalize(vision_feat, dim=1)
    text_norm = nn.functional.normalize(text_feat, dim=1)
    
    similarity = torch.matmul(vision_norm, text_norm.T) / temperature
    
    labels = torch.arange(batch_size, device=vision_feat.device)
    
    loss_v2t = nn.functional.cross_entropy(similarity, labels)
    loss_t2v = nn.functional.cross_entropy(similarity.T, labels)
    
    return (loss_v2t + loss_t2v) / 2

def compute_disentanglement_loss(shared_vision, shared_text, specific_vision, specific_text):
    """Compute feature disentanglement loss."""
    shared_loss = nn.functional.mse_loss(shared_vision, shared_text)
    
    vision_orthogonal = torch.sum(shared_vision * specific_vision, dim=1).mean()
    text_orthogonal = torch.sum(shared_text * specific_text, dim=1).mean()
    
    orthogonal_loss = (vision_orthogonal.abs() + text_orthogonal.abs()) / 2
    
    return shared_loss + orthogonal_loss

def train_model(model, dataloader, epochs=2, lr=1e-3, device='cpu'):
    """Train the DCAMFL++ model."""
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        
        for vision, text, labels in dataloader:
            vision, text, labels = vision.to(device), text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(vision, text)
            
            cls_loss = criterion(outputs['logits'], labels)
            
            contrastive_loss = compute_contrastive_alignment_loss(
                outputs['vision_gated'], outputs['text_gated']
            )
            
            disentanglement_loss = compute_disentanglement_loss(
                outputs['shared_vision'], outputs['shared_text'],
                outputs['specific_vision'], outputs['specific_text']
            )
            
            total_loss_batch = cls_loss + 0.1 * contrastive_loss + 0.05 * disentanglement_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item() * vision.size(0)
            total_samples += vision.size(0)
        
        avg_loss = total_loss / total_samples
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}')
    
    return epoch_losses
