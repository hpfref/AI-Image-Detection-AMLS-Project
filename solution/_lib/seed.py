"""
Deterministic seeding + thread caps.

Mirrors the thread-cap settings from PDF Appendix C so local timing matches
the grader's --cpus 8 environment.
"""

from __future__ import annotations

import os
import random


def set_deterministic(seed: int = 0) -> None:
    """Seed Python, numpy, and torch; cap CPU threads to <= 8."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        # Match Appendix C — grader runs with --cpus 8.
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
