from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    """Hyperparameters for the self-pruning neural network."""
    
    # Architecture
    in_features: int = 3 * 32 * 32  # CIFAR-10 shape
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    dropout_p: float = 0.1
    
    # Pruning & Regularization
    init_gate_val: float = 3.0
    gate_l1_penalty: float = 1e-4  # Multiplier for regularization loss (lambda)
    pruning_threshold: float = 1e-2
    
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 20
    
    # System
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    device: str = "auto"
    lambdas: List[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3])

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Loads configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Only unpack valid attributes
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Saves configuration to a YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f)
