"""
Reproducibility — Seeding Utility
==================================
Seeds Python's ``random``, ``numpy``, and ``torch`` (+ CUDA) for
fully deterministic runs.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Args:
        seed: Integer seed value (e.g. ``cfg.experiment.seed``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # for multi-GPU
    # Enforce deterministic algorithms where available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
