import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np

class BaseDiffusionModel(nn.Module):
    """Base diffusion model class."""
    def __init__(self, name='Base'):
        super(BaseDiffusionModel, self).__init__()
        self.name = name
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return F.relu(self.layer(x))

    def generate(self, prompt, scene_info=None):
        """Generate image from text prompt."""
        random.seed(hash(prompt) % (2**32))
        torch.manual_seed(hash(prompt) % (2**32))
        img = torch.rand(3, 64, 64)
        
        if scene_info is not None and 'noise' in scene_info:
            noise_level = scene_info['noise']
            img = img + noise_level * torch.rand(3, 64, 64)
            img = torch.clamp(img, 0.0, 1.0)
        return img

class CDMAD20Model(BaseDiffusionModel):
    """CDMAD 2.0 Model with dynamic Consistency-guided Global Modulation (CGM) module."""
    def __init__(self, dynamic=True):
        super(CDMAD20Model, self).__init__(name='CDMAD20')
        self.dynamic = dynamic
        
        self.local_encoder = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        self.global_transformer = nn.Linear(64*64*3, 256)
        
        self.cgm_module = nn.Linear(256, 1)

    def generate(self, prompt, scene_info=None):
        """Generate image using CDMAD 2.0 with dynamic CGM."""
        base_img = super().generate(prompt, scene_info)

        local_feat = self.local_encoder(base_img.unsqueeze(0))
        local_summary = local_feat.mean(dim=[1,2,3])

        global_input = base_img.view(-1)
        global_feat = self.global_transformer(global_input)

        if self.dynamic:
            gate = torch.sigmoid(self.cgm_module(global_feat.unsqueeze(0)))
            uncertainty = random.uniform(0.8, 1.2)
            fused_value = gate.item() * local_summary.item() * uncertainty + (1 - gate.item()) * local_summary.item()
        else:
            fused_value = 0.5 * local_summary.item() + 0.5 * local_summary.item()

        print(f"[Model: {self.name}] Prompt: '{prompt}' | Dynamic: {self.dynamic} | Fused value: {fused_value:.4f}")

        factor = (fused_value % 1) + 0.5
        generated_img = torch.clamp(base_img * factor, 0.0, 1.0)
        return generated_img

class OneStepDiffusionModel(BaseDiffusionModel):
    """Baseline one-step diffusion model without teacher guidance."""
    def __init__(self):
        super(OneStepDiffusionModel, self).__init__(name='OneStepBaseline')

class PretrainedMultiStepModel(BaseDiffusionModel):
    """Multi-step teacher model (pre-trained)."""
    def __init__(self):
        super(PretrainedMultiStepModel, self).__init__(name='MultiStepTeacher')

    def generate(self, prompt, scene_info=None):
        """Generate with simulated multi-step delay."""
        time.sleep(0.05)  # Simulate slower generation
        return super().generate(prompt, scene_info)

def train_models():
    """Initialize and prepare models for experiments."""
    print("Initializing models for CDMAD 2.0 experiments...")
    
    model_cdmad = CDMAD20Model(dynamic=True)
    model_baseline = OneStepDiffusionModel()
    model_teacher = PretrainedMultiStepModel()
    model_static = CDMAD20Model(dynamic=False)
    
    print("Models initialized:")
    print(f"- CDMAD 2.0 (dynamic): {model_cdmad.name}")
    print(f"- Baseline one-step: {model_baseline.name}")
    print(f"- Multi-step teacher: {model_teacher.name}")
    print(f"- CDMAD 2.0 (static): {model_static.name}")
    
    return model_cdmad, model_baseline, model_teacher, model_static
