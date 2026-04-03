"""Random seed utilities for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
