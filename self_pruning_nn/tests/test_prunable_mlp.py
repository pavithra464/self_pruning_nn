import torch
import pytest
from src.model.prunable_mlp import PrunableMLP

def test_mlp_forward_shape():
    """Verifies that the forward pass transforms input dimensions correctly."""
    in_features = 3 * 32 * 32
    num_classes = 10
    batch_size = 4
    
    model = PrunableMLP(
        in_features=in_features, 
        hidden_sizes=[128, 64], 
        num_classes=num_classes
    )
    
    # Input matching CIFAR-10 dimensions
    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)
    
    assert out.shape == (batch_size, num_classes)

def test_mlp_total_regularization_loss():
    """Checks that regularization loss is collected from all constituent prunable layers."""
    model = PrunableMLP(
        in_features=10, 
        hidden_sizes=[5], 
        num_classes=2
    )
    
    # Force specific gate scores
    # Layer 1: 10 * 5 = 50 params. Sigmoid(0) = 0.5. Loss = 25
    # Layer 2: 5 * 2 = 10 params. Sigmoid(0) = 0.5. Loss = 5
    # Total expected: 30
    
    for layer in model.get_prunable_layers():
        layer.gate_scores.data.fill_(0.0)
        
    total_loss = model.get_total_regularization_loss()
    assert torch.allclose(total_loss, torch.tensor(30.0))

def test_mlp_overall_sparsity():
    """Tests the calculation of the network-wide sparsity metric."""
    model = PrunableMLP(
        in_features=10, 
        hidden_sizes=[5], 
        num_classes=2
    )
    
    layers = model.get_prunable_layers()
    
    # Layer 1 has 50 elements
    layers[0].gate_scores.data.fill_(10.0) # > threshold
    
    # Layer 2 has 10 elements
    layers[1].gate_scores.data.fill_(-10.0) # < threshold
    
    # 10 out of 60 weights are pruned -> 1/6 sparsity
    sparsity = model.get_network_sparsity(threshold=0.5)
    expected_sparsity = 10.0 / 60.0
    
    assert torch.allclose(torch.tensor(sparsity), torch.tensor(expected_sparsity))
