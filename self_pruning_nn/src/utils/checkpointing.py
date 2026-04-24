import os
import torch
import torch.nn as nn
from typing import Optional

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    """Saves model and optimizer state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)
    assert os.path.exists(path), "Checkpoint file was not created successfully."

def load_checkpoint(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """Loads model and optimizer state, returns the epoch."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('epoch', -1)
