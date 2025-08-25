import torch
import os
import json
from train import HGDP_AT, HGDP_AT_Interpret, BaselineGNN, train_model
from preprocess import load_proteins_dataset, load_imdb_binary_dataset, generate_synthetic_graph
from evaluate import evaluate_model, time_forward_pass, plot_training_loss, plot_attention_scores, plot_runtime_comparison


def experiment_benchmark_classification():
    print('Starting Experiment 1: Benchmark Graph Classification')
    
    dataset, train_loader, test_loader = load_proteins_dataset()
    
    in_channels = dataset.num_features if dataset.num_features > 0 else 1
    hidden_channels = 64
    out_channels = dataset.num_classes
    
    model = HGDP_AT(in_channels, hidden_channels, out_channels)
    
    epoch_losses = train_model(model, train_loader, num_epochs=20, lr=0.005)
    
    plot_training_loss(epoch_losses, '.research/iteration1/images/training_loss.pdf')
    print('Training loss curve saved as .research/iteration1/images/training_loss.pdf')
    
    accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return accuracy


def experiment_interpretability_analysis():
    print('Starting Experiment 2: Interpretability and Dynamic Tokenization Analysis')
    
    dataset = load_imdb_binary_dataset()
    data = dataset[0]
    
    in_channels = data.num_features if data.num_features > 0 else 1
    model = HGDP_AT_Interpret(in_channels, 64, dataset.num_classes)
    model.eval()
    
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        out, token_attention = model(data.x, data.edge_index, batch)
    
    plot_attention_scores(token_attention, '.research/iteration1/images/dynamic_tokenization_attention.pdf')
    print('Attention scores plot saved as .research/iteration1/images/dynamic_tokenization_attention.pdf')


def experiment_scalability_efficiency():
    print('Starting Experiment 3: Scalability and Efficiency Evaluation')
    
    x, edge_index, batch = generate_synthetic_graph(num_nodes=5000, edge_prob=0.005, feature_dim=16)
    
    model_hgdpat = HGDP_AT(16, 32, 2)
    model_baseline = BaselineGNN(16, 32, 2)
    
    runtime_hgdpat = time_forward_pass(model_hgdpat, x, edge_index, batch, iterations=10)
    runtime_baseline = time_forward_pass(model_baseline, x, edge_index, batch, iterations=10)
    
    print(f'Average runtime per forward pass (HGDP-AT): {runtime_hgdpat:.6f} seconds')
    print(f'Average runtime per forward pass (Baseline): {runtime_baseline:.6f} seconds')
    
    models = ['HGDP-AT', 'Baseline']
    runtimes = [runtime_hgdpat, runtime_baseline]
    
    plot_runtime_comparison(models, runtimes, '.research/iteration1/images/inference_latency.pdf')
    print('Runtime comparison plot saved as .research/iteration1/images/inference_latency.pdf')


def test_experiments():
    print('Running quick tests for experiments...')
    
    try:
        print('\n[TEST] Experiment 1: Benchmark Graph Classification')
        experiment_benchmark_classification()
    except Exception as e:
        print(f'Experiment 1 failed: {e}')

    try:
        print('\n[TEST] Experiment 2: Interpretability Analysis')
        experiment_interpretability_analysis()
    except Exception as e:
        print(f'Experiment 2 failed: {e}')
    
    try:
        print('\n[TEST] Experiment 3: Scalability and Efficiency')
        experiment_scalability_efficiency()
    except Exception as e:
        print(f'Experiment 3 failed: {e}')

    print('\nQuick tests completed.')


def set_status_stopped():
    status_file = '.research/status.json'
    status_data = {"status_enum": "stopped"}
    
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f'Status set to "stopped" in {status_file}')


if __name__ == '__main__':
    os.makedirs('.research/iteration1/images', exist_ok=True)
    
    print('Running Experiment 1: Benchmark Graph Classification')
    experiment_benchmark_classification()

    print('\nRunning Experiment 2: Interpretability and Dynamic Tokenization Analysis')
    experiment_interpretability_analysis()

    print('\nRunning Experiment 3: Scalability and Efficiency Evaluation')
    experiment_scalability_efficiency()

    print('\nRunning quick test function')
    test_experiments()
    
    set_status_stopped()
    print('\nAll experiments completed successfully!')
