"""
Threshold calibration to a target false-positive rate.

PDF §1.2 / §1.3:
    - Goal: maximize recall_ai under FPR_real <= 0.20.
    - Calibrate on data/calibration/ (or data/calibration_augmented/ for Task 3).
    - **Independently validate** the achieved FPR on data/validation/
      (or data/validation_augmented/) — we must NOT use validation to set the
      threshold.
    - Calibration must be automatic, not manual.
"""

from __future__ import annotations

from typing import Sequence


def pick_threshold_for_fpr(
    scores_real: Sequence[float],
    target_fpr: float = 0.20,
) -> float:
    """Pick the smallest threshold such that the FPR on real-class scores is
    no greater than `target_fpr`.

    `scores_real` is the model's AI-class probability (or score) on calibration
    samples whose true label is 0 (real). The returned threshold is then used
    at inference: `predicted = 1 if score >= threshold else 0`.
    """
    raise NotImplementedError("Task 1.2: implement FPR-target threshold picker")


def write_threshold_json(path, threshold: float, target_fpr: float) -> None:
    """Persist the calibrated threshold so predict.py can load it back."""
    raise NotImplementedError("Task 1.2: write threshold + metadata to artifacts/<task>/threshold.json")


def read_threshold_json(path) -> float:
    raise NotImplementedError("Task 1.2: read calibrated threshold")
