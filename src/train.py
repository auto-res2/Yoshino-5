#!/usr/bin/env python3
"""
Training module for Adaptive MambaFormer experiments.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, optimizer, criterion, num_epochs=3, device='cpu'):
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        List of loss values per epoch
    """
    model.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            x = batch[0].to(device)
            x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
            optimizer.zero_grad()
            
            output = model(x)
            if isinstance(output, tuple):
                output, _ = output  # Adaptive model returns (output, branch_usage)
            
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return loss_history


def train_with_validation(model, train_loader, val_loader, optimizer, criterion, 
                         num_epochs=10, device='cpu', early_stopping_patience=3):
    """
    Train a model with validation and early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Maximum number of training epochs
        device: Device to train on
        early_stopping_patience: Number of epochs to wait before early stopping
    
    Returns:
        Dictionary with training and validation loss histories
    """
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            x = batch[0].to(device)
            x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
            optimizer.zero_grad()
            
            output = model(x)
            if isinstance(output, tuple):
                output, _ = output
            
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x = x.transpose(0, 1)  # Convert from (batch, seq, features) to (seq, batch, features)
                output = model(x)
                if isinstance(output, tuple):
                    output, _ = output
                
                loss = criterion(output, x)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(model, optimizer, filepath, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load on
    
    Returns:
        Dictionary with loaded checkpoint info
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }
