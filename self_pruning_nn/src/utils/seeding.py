import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets the random seed across all libraries to ensure reproducible experiments.
    
    Args:
        seed: The integer seed to establish.
        deterministic: Whether to configure CuDNN for deterministic behavior (may impact performance).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
