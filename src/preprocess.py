import torch
import numpy as np
import random

def get_dummy_text_prompts(num_prompts=5):
    """Generate dummy text prompts for testing."""
    prompts = [f"A sample image of item {i}" for i in range(1, num_prompts+1)]
    return prompts

def get_dummy_ground_truth_images(prompts):
    """Generate dummy ground truth images for simulation."""
    gt_images = [torch.rand(3, 64, 64) for _ in prompts]
    return gt_images

class Scene:
    """Scene class for Experiment 3 with varying complexity."""
    def __init__(self, prompt, ground_truth, complexity=1):
        self.prompt = prompt
        self.ground_truth = ground_truth
        self.info = {'complexity': complexity, 'noise': 0.0}
        self.complexity = complexity

    def copy(self):
        new_scene = Scene(self.prompt, self.ground_truth.clone(), self.complexity)
        new_scene.info = self.info.copy()
        return new_scene

    def add_noise(self, noise_level):
        """Add noise to the scene for robustness testing."""
        self.info['noise'] = noise_level
        self.complexity = self.complexity + noise_level
        self.ground_truth = torch.clamp(self.ground_truth + noise_level * torch.rand(3, 64, 64), 0.0, 1.0)

def prepare_experiment_data():
    """Prepare data for all experiments."""
    print("Preparing experiment data...")
    
    prompts = get_dummy_text_prompts(5)
    gt_images = get_dummy_ground_truth_images(prompts)
    
    scenes = []
    for i in range(len(prompts)):
        scenes.append(Scene(prompts[i], gt_images[i], complexity=1 + i))
    
    print(f"Generated {len(prompts)} text prompts and corresponding ground truth images")
    print(f"Created {len(scenes)} scenes with varying complexity")
    
    return prompts, gt_images, scenes
