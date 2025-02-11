import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List, Tuple
from .evaluation import evaluate_model
from .plotter import plot_losses

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10
) -> Tuple[nn.Module, List[float], List[float]]:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch_times = []
    
    model.to(device)
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss, _ = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model.pth'))
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Plot progress
        plot_losses(train_losses, val_losses, save_dir)
    
    return model, train_losses, val_losses
