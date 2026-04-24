import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from ..model.prunable_mlp import PrunableMLP

def get_flattened_gates(model: PrunableMLP) -> np.ndarray:
    """Aggregates all gate values across prunable layers into a single 1D numpy array."""
    gates = []
    for layer in model.get_prunable_layers():
        # Get the detached gate values
        gate_vals = layer.get_gate_values().detach().cpu().flatten()
        gates.append(gate_vals)
    
    if len(gates) > 0:
        return torch.cat(gates).numpy()
    return np.array([])


def plot_gate_distribution(model: PrunableMLP, filepath: str, title: str = "Gate Value Distribution") -> None:
    """Plots and saves a histogram of the final gate values."""
    gates_np = get_flattened_gates(model)
    
    assert len(gates_np) > 0, "No gate values found to plot!"
    
    plt.figure(figsize=(10, 6))
    
    # We use a custom bin allocation to properly view extreme polar values (0 and 1)
    # Since they pile up at exactly 0.0 or 1.0 due to saturation.
    bins = np.linspace(0, 1, 50)
    plt.hist(gates_np, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    
    plt.title(title)
    plt.xlabel('Gate Value (after Sigmoid)')
    plt.ylabel('Frequency (Weight Count)')
    plt.grid(axis='y', alpha=0.75)
    
    # Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
