import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from train import DummyDiffusionModel

def test_experiments():
    print('--- Running Test Function for AIB-2Q-Diff Experiments ---')
    test_data = torch.randn(20, 256)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=False)
    model = DummyDiffusionModel(use_aib=True)
    start = time.time()
    for batch in test_loader:
        _ = model(batch, timesteps=5)
    elapsed = time.time() - start
    print(f'Test completed in {elapsed:.2f}s')
    
    plt.figure()
    plt.plot([0, 1, 2], [0, 1, 0])
    plt.title('Test Plot')
    
    output_path = os.path.join('.research', 'iteration1', 'images', 'test_plot.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Test plot saved as {output_path}')
    
    print('Test Function completed successfully.')
