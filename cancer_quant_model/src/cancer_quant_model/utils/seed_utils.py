"""Random seed utilities for reproducibility."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but fully reproducible)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_worker_seed(worker_id: int, base_seed: int = 42) -> int:
    """Get seed for DataLoader worker.

    Args:
        worker_id: Worker ID
        base_seed: Base random seed

    Returns:
        Seed for the worker
    """
    return base_seed + worker_id


def worker_init_fn(worker_id: int):
    """Worker initialization function for DataLoader.

    Args:
        worker_id: Worker ID
    """
    worker_seed = get_worker_seed(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
