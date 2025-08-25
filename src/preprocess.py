import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import erdos_renyi_graph


def load_proteins_dataset():
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    if dataset.num_features == 0:
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
    
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return dataset, train_loader, test_loader


def load_imdb_binary_dataset():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-BINARY')
    if dataset.num_features == 0:
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
    
    return dataset


def generate_synthetic_graph(num_nodes=5000, edge_prob=0.005, feature_dim=16):
    edge_index = erdos_renyi_graph(num_nodes, edge_prob)
    x = torch.randn(num_nodes, feature_dim)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    return x, edge_index, batch


def prepare_dataset_features(dataset):
    if dataset.num_features == 0:
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
    return dataset
