import torch
import numpy as np
import random
import time

def compute_fid(generated_images, ground_truth_images):
    """Compute Fréchet Inception Distance (dummy implementation)."""
    fid = np.mean([torch.norm(gen - gt).item() for gen, gt in zip(generated_images, ground_truth_images)])
    return fid

def compute_clip_score(generated_images, text_prompts):
    """Compute CLIP score (dummy implementation)."""
    return np.mean([0.3 + random.uniform(-0.05, 0.05) for _ in text_prompts])

def evaluate_model(model, prompts, gt_images):
    """Evaluate a model on given prompts and ground truth images."""
    generated_images = []
    start = time.time()
    
    for prompt in prompts:
        img = model.generate(prompt)
        generated_images.append(img)
    
    gen_time = (time.time() - start) / len(prompts)
    fid = compute_fid(generated_images, gt_images)
    clip_score = compute_clip_score(generated_images, prompts)
    
    return fid, clip_score, gen_time

def run_ablation_study(model, prompts, gt_images):
    """Run ablation study for CGM module."""
    results = []
    for prompt, gt in zip(prompts, gt_images):
        img = model.generate(prompt)
        fid = compute_fid([img], [gt])
        clip = compute_clip_score([img], [prompt])
        results.append((fid, clip))
    
    avg_fid = np.mean([r[0] for r in results])
    avg_clip = np.mean([r[1] for r in results])
    return avg_fid, avg_clip

def evaluate_scene_complexity(model, scenes, noise_levels):
    """Evaluate model performance under varying scene complexity."""
    performance_records = []
    
    for noise in noise_levels:
        for scene in scenes:
            scene_copy = scene.copy()
            scene_copy.add_noise(noise)
            
            start = time.time()
            img = model.generate(scene_copy.prompt, scene_info=scene_copy.info)
            exec_time = time.time() - start
            
            fid = compute_fid([img], [scene_copy.ground_truth])
            clip = compute_clip_score([img], [scene_copy.prompt])
            
            performance_records.append({
                'noise': noise,
                'scene_complexity': scene_copy.complexity,
                'exec_time': exec_time,
                'fid': fid,
                'clip_score': clip
            })
    
    return performance_records
