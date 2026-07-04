"""
seed.py — Reproducibility utilities.

Sets random seeds for Python, NumPy, and PyTorch to ensure deterministic
behaviour across training runs. Also configures CuDNN determinism when
requested.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool
        If True, configure CuDNN for deterministic behaviour.
        This may reduce performance by 10-15%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.0+ deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch without warn_only
            pass
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info("Random seed set to %d (deterministic=%s)", seed, deterministic)
