from typing import Dict, Tuple, Callable, Any
import torch
from torch import nn
from torch.utils.data import DataLoader

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    val_loss = 0.0
    val_component_losses = {}
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    return val_loss, val_component_losses

