"""
Evaluation metrics for the AI-image-detection task.

Two headline metrics (PDF §1.2):
    recall_ai  =  TP / (TP + FN)         on AI-labeled samples (label = 1)
    fpr_real   =  FP / (FP + TN)         on real-labeled samples (label = 0)

Constraint: fpr_real <= 0.20. Goal: maximize recall_ai under that constraint.
"""

from __future__ import annotations

from typing import Sequence


def recall_ai(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall on the AI class (label = 1)."""
    raise NotImplementedError("Task 1.2: implement recall_ai")


def fpr_real(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """False-positive rate on real images (label = 0 misclassified as 1)."""
    raise NotImplementedError("Task 1.2: implement fpr_real")


def confusion(y_true: Sequence[int], y_pred: Sequence[int]):
    """Return a 2x2 confusion matrix as a dict or array — used in the report."""
    raise NotImplementedError("Task 1.2: implement confusion matrix")
