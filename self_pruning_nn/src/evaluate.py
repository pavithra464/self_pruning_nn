import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils.metrics import AverageMeter, calculate_accuracy
from .utils.logging_utils import get_logger

logger = get_logger("Evaluate")

def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    lambda_reg: float,
    threshold: float
) -> dict:
    """
    Evaluates the model on the provided data loader, computing classification loss,
    accuracy, and current network-level sparsity.
    """
    model.eval()
    
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            assert inputs.shape[0] > 0, "Empty batch received."
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            cls_loss = criterion(outputs, targets)
            
            acc1 = calculate_accuracy(outputs, targets)
            
            cls_losses.update(cls_loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            
    # Sparsity
    sparsity_level = model.get_network_sparsity(threshold=threshold) * 100.0
    
    # Regularization
    reg_loss = model.get_total_regularization_loss().item()
    total_loss = cls_losses.avg + lambda_reg * reg_loss
    
    metrics = {
        "loss": total_loss,
        "cls_loss": cls_losses.avg,
        "reg_loss": reg_loss,
        "accuracy": top1.avg,
        "sparsity": sparsity_level
    }
    
    return metrics
