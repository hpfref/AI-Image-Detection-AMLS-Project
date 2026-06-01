"""
Evaluation metrics for the AI-image-detection task.

Two headline metrics (PDF §1.2):
    recall_ai  =  TP / (TP + FN)         on AI-labeled samples (label = 1)
    fpr_real   =  FP / (FP + TN)         on real-labeled samples (label = 0)

Constraint: fpr_real <= 0.20. Goal: maximize recall_ai under that constraint.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def recall_ai(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall on the AI class (label = 1)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def fpr_real(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """False-positive rate on real images (label = 0 misclassified as 1)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Return tn, fp, fn, tp counts as a dict."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return {
        "tn": int(((yp == 0) & (yt == 0)).sum()),
        "fp": int(((yp == 1) & (yt == 0)).sum()),
        "fn": int(((yp == 0) & (yt == 1)).sum()),
        "tp": int(((yp == 1) & (yt == 1)).sum()),
    }
