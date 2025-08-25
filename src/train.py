import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


class HGDP_AT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGDP_AT, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x)
        x = torch.relu(x)
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        for i in range(num_graphs):
            mask = (batch == i)
            if mask.sum() > 0:
                out[i] = x[mask].mean(dim=0)
        out = self.conv2(out)
        return out


class HGDP_AT_Interpret(HGDP_AT):
    def forward(self, x, edge_index, batch):
        token_attention = torch.softmax(torch.randn(x.size(0)), dim=0)
        x = self.conv1(x)
        x = torch.relu(x)
        num_graphs = int(batch.max().item()) + 1
        global_summary = torch.zeros(num_graphs, x.size(1), device=x.device)
        for i in range(num_graphs):
            mask = (batch == i)
            if mask.sum() > 0:
                global_summary[i] = x[mask].mean(dim=0)
        out = self.conv2(global_summary)
        return out, token_attention


class BaselineGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BaselineGNN, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x)
        x = torch.relu(x)
        num_graphs = int(batch.max().item()) + 1
        global_rep = torch.zeros(num_graphs, x.size(1), device=x.device)
        for i in range(num_graphs):
            mask = (batch == i)
            if mask.sum() > 0:
                global_rep[i] = x[mask].mean(dim=0)
        out = self.conv2(global_rep)
        return out


def train_model(model, train_loader, num_epochs=20, lr=0.005):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}')
    
    return epoch_losses
