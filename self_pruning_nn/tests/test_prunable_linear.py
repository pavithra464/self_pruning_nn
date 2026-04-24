import torch
import pytest
from src.model.prunable_linear import PrunableLinear

def test_prunable_linear_shapes():
    """Test standard forward pass tensor shapes."""
    batch_size, in_features, out_features = 32, 128, 64
    layer = PrunableLinear(in_features, out_features)
    
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    
    assert out.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {out.shape}"
    

def test_gate_values_range():
    """Test that gates are always constrained to [0, 1]."""
    layer = PrunableLinear(10, 5)
    
    # Default initialization
    gates = layer.get_gate_values()
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)
    
    # Extreme values
    layer.gate_scores.data.fill_(100.0)
    assert torch.all(layer.get_gate_values() <= 1.0)
    
    layer.gate_scores.data.fill_(-100.0)
    assert torch.all(layer.get_gate_values() >= 0.0)


def test_gradient_flow_to_gates():
    """Test that gradients successfully propagate to the gate_scores parameter."""
    layer = PrunableLinear(10, 5)
    
    x = torch.randn(2, 10)
    y = torch.randn(2, 5)
    
    out = layer(x)
    loss = torch.nn.functional.mse_loss(out, y)
    
    # Explicitly calculate grads
    loss.backward()
    
    assert layer.weight.grad is not None, "Weight gradients should not be None."
    assert layer.gate_scores.grad is not None, "Gate score gradients should not be None."
    # With random inputs and initialization, gradient should be non-zero
    assert torch.sum(torch.abs(layer.gate_scores.grad)) > 0


def test_regularization_loss():
    """Test regularization loss sums up gates correctly."""
    layer = PrunableLinear(10, 5, init_gate_val=0.0)
    # At gate_scores = 0.0, sigmoid(0) = 0.5
    # Total gates = 50, so sum should be 25
    
    reg_loss = layer.regularization_loss()
    assert torch.allclose(reg_loss, torch.tensor(25.0)), f"Expected ~25, got {reg_loss.item()}"


def test_sparsity():
    """Test sparsity calculation works under controlled conditions."""
    layer = PrunableLinear(10, 5)
    
    # Force 30 elements to have small gates (sigmoid(-10) is practically 0)
    # Total 50 elements. 30 pruned means 60% sparsity.
    layer.gate_scores.data[:3, :] = -10.0
    
    sparsity = layer.sparsity(threshold=0.5)
    assert torch.allclose(torch.tensor(sparsity), torch.tensor(0.6))

def test_save_load_state_dict():
    """Check that saving and loading state dict respects gate scores."""
    layer1 = PrunableLinear(10, 5)
    layer1.gate_scores.data.fill_(1.23)
    
    state_dict = layer1.state_dict()
    
    layer2 = PrunableLinear(10, 5)
    layer2.load_state_dict(state_dict)
    
    assert torch.allclose(layer1.gate_scores, layer2.gate_scores)
    assert torch.allclose(layer1.weight, layer2.weight)
