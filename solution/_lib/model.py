"""
Model architectures.

Includes the reference CNN from PDF Appendix B as a starting point. Per PDF
§1.2 you must compare at least two model families in the report — e.g. this
CNN vs. a classical engineered-feature baseline — but only the single best
pipeline is packaged in solution/.
"""

from __future__ import annotations

import torch.nn as nn


def build_appendix_b_cnn(k: int = 32) -> nn.Module:
    """Reference CNN from PDF Appendix B.

    Two nn.Sequential blocks (`features` + `classifier`) are wrapped here as a
    single nn.Sequential so it can be used directly with optimizer / loss.
    Treat as a starting point, not a final solution.
    """

    # --- BEGIN: copied from PDF Appendix B ---
    features = nn.Sequential(
        nn.Conv2d(3, k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * k, 2),
    )
    # --- END: copied from PDF Appendix B ---

    return nn.Sequential(features, classifier)


def build_classical_baseline():
    """Engineered-feature baseline (e.g. color/texture stats + logistic reg).

    Required for PDF §1.2 ("at least two different model families") so the
    report can compare a classical baseline against the CNN.
    """
    raise NotImplementedError("Task 1.2: implement classical baseline")
