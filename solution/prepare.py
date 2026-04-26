"""
Task 1.2 — Data Preparation  (part of the 35-point Modeling task)

Reads:    artifacts/clean/                 (cleaned training data from clean.py)
          data/calibration/, data/validation/, data/validation_augmented/
Writes:   artifacts/prepared/              (tensor caches / feature matrices)

Scope:
    Materialize whatever train.py needs to start as fast as possible —
    decoded tensors, engineered features, train/val splits, etc. — so the
    1800 s training budget isn't burned re-decoding parquet files.

WARNING (PDF §Submission Guidelines):
    Do NOT process data/predict/ here. The contents of data/predict/ may be
    swapped between training and evaluation, so anything derived from it
    must be computed inside predict.py / predict_augmented.py at runtime.

CLI:
    python prepare.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import sys

from _lib import io as _io
from _lib import seed as _seed


def prepare_train_features() -> None:
    """Decode + transform cleaned training data into tensors / features."""
    raise NotImplementedError("Task 1.2: prepare train features from artifacts/clean/")


def prepare_calibration_features() -> None:
    """Same transform as training, applied to data/calibration/."""
    raise NotImplementedError("Task 1.2: prepare calibration features")


def prepare_validation_features() -> None:
    """Same transform as training, for both validation/ and validation_augmented/."""
    raise NotImplementedError("Task 1.2: prepare validation features")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: prepare features for training")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget for this script.")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    _io.ensure_artifact_dir("prepared")

    print(f"prepare.py: not yet implemented (timeout_seconds={args.timeout_seconds}).")
    # TODO: prepare_train_features()
    # TODO: prepare_calibration_features()
    # TODO: prepare_validation_features()
    # NOTE: do NOT touch data/predict/ here.
    return 0


if __name__ == "__main__":
    sys.exit(main())
