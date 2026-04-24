import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .model.prunable_mlp import PrunableMLP
from .utils.metrics import AverageMeter, calculate_accuracy
from .utils.logging_utils import get_logger

logger = get_logger("Train")

def train_one_epoch(
    model: PrunableMLP,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_reg: float,
    clip_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch and return metrics."""
    model.train()
    
    total_losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    top1 = AverageMeter()
    
    # Optional debug wrap with tqdm if iter mode
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Defensive check
        assert not torch.isnan(outputs).any(), "NaN found in network outputs!"
        assert outputs.shape[1] == model.network[-1].out_features, "Mismatch in class prediction count!"
        
        cls_loss = criterion(outputs, targets)
        reg_loss = model.get_total_regularization_loss()
        
        loss = cls_loss + lambda_reg * reg_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient norm logging (useful for debugging exploding gradients)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Update metrics
        acc1 = calculate_accuracy(outputs, targets)
        batch_size = inputs.size(0)
        
        total_losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        reg_losses.update(reg_loss.item(), batch_size)
        top1.update(acc1, batch_size)
        
    metrics = {
        "loss": total_losses.avg,
        "cls_loss": cls_losses.avg,
        "reg_loss": reg_losses.avg,
        "accuracy": top1.avg
    }
    return metrics

def run_smoke_test(
    model: PrunableMLP, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
):
    """
    Runs a single-batch overfit test to ensure the model can minimize loss and backpropagation works.
    """
    logger.info("Running smoke test (1-batch overfit)...")
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    model.train()
    initial_loss = None
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
            
    final_loss = loss.item()
    logger.info(f"Smoke test loss: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Model failed to overfit a single batch. Learning is broken."
    logger.info("Smoke test passed.")
