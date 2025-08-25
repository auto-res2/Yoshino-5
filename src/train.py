import torch
import torch.nn as nn
import numpy as np

class AIBModule(nn.Module):
    """Adaptive Information Bottleneck Module"""
    def __init__(self):
        super(AIBModule, self).__init__()
    
    def forward(self, activation, final_output=None):
        return torch.sigmoid(torch.mean(activation))

class PrecisionAllocationLayer(nn.Module):
    """Learnable Precision Allocation Layer"""
    def __init__(self):
        super(PrecisionAllocationLayer, self).__init__()
        self.precision_param = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, info_measure, timestep):
        bits = 2 + (8 - 2) * (1 - info_measure)  # more information means higher precision
        bits = torch.clamp(bits, min=2, max=8)
        return int(torch.round(bits).item())

class DummyDiffusionModel(nn.Module):
    """AIB-2Q-Diff Model with Adaptive Information Bottleneck"""
    def __init__(self, use_aib=True):
        super(DummyDiffusionModel, self).__init__()
        self.use_aib = use_aib
        self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(10)])
        if self.use_aib:
            self.aib = AIBModule()
            self.prec_alloc = PrecisionAllocationLayer()

    def forward(self, x, timesteps):
        for t, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.tanh(x)  # simulate a non-linearity
            if self.use_aib:
                info = self.aib(x, final_output=None)  # In real use, final_output needed
                quant_bits = self.prec_alloc(info, timestep=t)
                x = torch.round(x * (2 ** quant_bits)) / (2 ** quant_bits)
            else:
                x = torch.round(x * (2 ** 2)) / (2 ** 2)
        return x

class NoAIBDiffusionModel(DummyDiffusionModel):
    """Modified diffusion model for ablation study that bypasses AIB module"""
    def __init__(self, use_aib=False):
        super(NoAIBDiffusionModel, self).__init__(use_aib=False)
    
    def forward(self, x, timesteps):
        for t, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.tanh(x)
            x = torch.round(x * (2 ** 2)) / (2 ** 2)
        return x
