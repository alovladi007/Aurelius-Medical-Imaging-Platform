"""Seed utilities for reproducibility."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms (slower but reproducible)
        benchmark: Use cudnn benchmarking (faster but less reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark

    # Environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Initialize worker seed for DataLoader.

    Args:
        worker_id: Worker ID
        seed: Base random seed
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch generator with a specific seed.

    Args:
        seed: Random seed

    Returns:
        PyTorch generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
