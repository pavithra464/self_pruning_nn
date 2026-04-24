import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A custom linear layer that implements a self-pruning mechanism.
    
    Instead of using standard nn.Linear, this layer maintains a trainable 
    `gate_scores` tensor of the same shape as the `weight` tensor. During the 
    forward pass, the gate scores are transformed via a sigmoid function to obtain 
    gate values in [0, 1]. These gate values multiplicatively modulate the weights.
    By regularizing the gate values towards 0, the model learns to self-prune 
    individual weights.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        init_gate_val: float = 3.0
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use a bias term. Default: True.
            init_gate_val: Constant initialization value for the gate scores before sigmoid.
                           A value of 3.0 gives an initial gate value of ~0.95.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_gate_val = init_gate_val
        
        # Standard weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Gate scores
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """
        Initializes the parameters of the layer.
        Uses Xavier uniform initialization for weights to ensure balanced starting variances.
        """
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.weight)
        
        # Standard bias initialization dependent on fan-in
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores
        # We start with a positive value so that sigmoid(gate_score) is initially close to 1.
        # This allows gradients to flow normally and the network to learn before pruning kicks in.
        nn.init.constant_(self.gate_scores, self.init_gate_val)
        
    def get_gate_values(self) -> torch.Tensor:
        """
        Returns the gate values constrained to [0, 1] using the sigmoid function.
        
        Engineering Note (Gradient Flow Maintenance):
        Because the sigmoid function produces a continuous and smoothly differentiable 
        curve natively bounding strictly within (0, 1), standard backpropagation derivatives 
        cascade naturally back into the fundamental `gate_scores` tensor parameters directly 
        without disruption limitation blockages. This elegantly completely eliminates any need 
        for structurally unstable optimization approximations (e.g. Straight-Through Estimators 
        (STE) associated frequently with binary operational step functions).
        """
        return torch.sigmoid(self.gate_scores)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using modulated weights.
        """
        # Assertions for defensive programming and sanity checking
        assert x.shape[-1] == self.in_features, f"Input feature dim expected: {self.in_features}, got: {x.shape[-1]}"
        
        gates = self.get_gate_values()
        pruned_weights = self.weight * gates
        
        return F.linear(x, pruned_weights, self.bias)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """
        Calculates the sparsity of the layer as the fraction of gates below a specific threshold.
        
        Args:
            threshold: Values below this are considered completely pruned.
            
        Returns:
            The fraction of weights that are pruned (0 to 1).
        """
        with torch.no_grad():
            gates = self.get_gate_values()
            fraction_pruned = (gates < threshold).float().mean().item()
            return fraction_pruned
            
    def regularization_loss(self) -> torch.Tensor:
        """
        Computes the L1-style regularization term for the gates.
        Returns the sum of all gate values. Pushing this to zero encourages sparsity.
        """
        return self.get_gate_values().sum()

    def extra_repr(self) -> str:
        """String representation showing structural info."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
