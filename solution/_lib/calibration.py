"""
Threshold calibration to a target false-positive rate.

PDF §1.2 / §1.3:
    - Goal: maximize recall_ai under FPR_real <= 0.20.
    - Calibrate on data/calibration/ (or data/calibration_augmented/ for Task 3).
    - Independently validate the achieved FPR on data/validation/
      (or data/validation_augmented/) -- we must NOT use validation to set the
      threshold.
    - Calibration must be automatic, not manual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def pick_threshold_for_fpr(
    scores_real: Sequence[float],
    target_fpr: float = 0.20,
) -> float:
    """Smallest threshold t s.t. empirical FPR on real scores <= target_fpr.

    Iterates unique score values ascending; the first t where the count of
    real scores >= t falls within the FP budget is returned. This avoids the
    quantile-index approach which overshoots when many reals cluster at the
    boundary.
    """
    s = np.asarray(scores_real, dtype=float)
    n = len(s)
    if n == 0:
        return 0.5
    max_fp = int(np.floor(target_fpr * n))
    if max_fp >= n:
        return float(s.min())
    candidates = np.sort(np.unique(s))
    for t in candidates:
        if int((s >= t).sum()) <= max_fp:
            return float(t)
    return float(candidates[-1]) + 1e-9


def write_threshold_json(path, data: dict) -> None:
    """Persist threshold + ensemble config to JSON.

    data must contain 'thr' (float) and 'alpha' (float) at minimum.
    """
    Path(path).write_text(json.dumps(data, indent=2))


def read_threshold_json(path) -> dict:
    """Read threshold JSON, return dict with at least 'thr' and 'alpha'."""
    return json.loads(Path(path).read_text())
