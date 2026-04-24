import torch
import torch.nn as nn
from typing import List
from .prunable_linear import PrunableLinear

class PrunableMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for classification, implementing self-pruning
    capabilities by using PrunableLinear layers instead of standard linear layers.
    
    Structure: Flatten -> [PrunableLinear -> BN -> ReLU -> Dropout] * N -> PrunableLinear -> Logits
    """
    
    def __init__(
        self, 
        in_features: int, 
        hidden_sizes: List[int], 
        num_classes: int, 
        dropout_p: float = 0.1,
        debug: bool = False
    ):
        """
        Args:
            in_features: Flattened input dimension (e.g. 3 * 32 * 32 for CIFAR-10)
            hidden_sizes: List of hidden feature sizes
            num_classes: Number of output classes
            dropout_p: Dropout probability
            debug: If True, prints forward shapes
        """
        super().__init__()
        self.debug = debug
        self.in_features = in_features
        
        self.flatten = nn.Flatten()
        
        layers = []
        current_in = in_features
        
        for h_size in hidden_sizes:
            layers.append(PrunableLinear(current_in, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_in = h_size
            
        # Final classification layer
        layers.append(PrunableLinear(current_in, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        """
        x_flat = self.flatten(x)
        
        if self.debug:
            print(f"[DEBUG] Input shape: {x.shape}")
            print(f"[DEBUG] Flattened shape: {x_flat.shape}")
            
        out = self.network(x_flat)
        return out

    def get_prunable_layers(self) -> List[PrunableLinear]:
        """Returns all prunable layers in the model."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def get_total_regularization_loss(self) -> torch.Tensor:
        """
        Computes the total regularization loss by summing the gate values 
        across all prunable layers in the network.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.get_prunable_layers():
            total_loss += layer.regularization_loss()
        return total_loss

    def get_network_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Returns the overall proportion of pruned weights across all prunable layers.
        """
        total_gates = 0
        pruned_gates = 0
        
        for layer in self.get_prunable_layers():
            with torch.no_grad():
                gates = layer.get_gate_values()
                pruned_gates += (gates < threshold).sum().item()
                total_gates += gates.numel()
                
        if total_gates == 0:
            return 0.0
            
        return pruned_gates / total_gates
