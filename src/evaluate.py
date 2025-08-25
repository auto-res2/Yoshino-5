import torch
import time
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.num_graphs
    accuracy = correct / total
    return accuracy


def time_forward_pass(model, x, edge_index, batch, iterations=10):
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(iterations):
            _ = model(x, edge_index, batch)
        elapsed = (time.time() - start) / iterations
    return elapsed


def plot_training_loss(epoch_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
    plt.title('Training Loss Curve for HGDP-AT Graph Classification')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_attention_scores(token_attention, save_path):
    plt.figure(figsize=(10, 4))
    markerline, stemlines, baseline = plt.stem(token_attention.cpu().numpy())
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.title('Dynamic Tokenization Attention Scores')
    plt.xlabel('Node Index')
    plt.ylabel('Attention Score')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_runtime_comparison(models, runtimes, save_path):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, runtimes, color=['skyblue', 'salmon'])
    plt.title('Average Forward Pass Runtime Comparison')
    plt.ylabel('Time (seconds)')
    for bar, rt in zip(bars, runtimes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{rt:.6f}',
                 ha='center', va='bottom')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
