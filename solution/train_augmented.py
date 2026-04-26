"""
Task 1.3 — Data Augmentation & Feature Engineering  (30/100 points)

Reads:    artifacts/prepared/                  (or artifacts/clean/ — your call)
          artifacts/task02/best.pt             (optional: fine-tune from here)
          data/calibration_augmented/          (calibration on shifted data)
          data/validation/                     (still validated here)
          data/validation_augmented/           (primary robustness target)
Writes:   artifacts/task03/best.pt
          artifacts/task03/threshold.json

Goals (PDF §1.3):
    - Improve robustness to realistic distortions: scaling, JPEG compression,
      blur, etc. Justify the chosen augmentations in the report.
    - May train from scratch OR continue from artifacts/task02/best.pt.
    - Same overall constraints as Task 1.2: same CPU budget, same hard
      fpr_real <= 0.20.
    - Threshold calibrated automatically on data/calibration_augmented/
      (and report robustness on data/validation_augmented/).

Targets:
    - recall_ai >= 0.6 on data/validation_augmented/ with fpr_real <= 0.20.
    - Strong solutions push higher AND keep competitive performance on
      data/validation/.

Augmentation candidates to consider (decide and justify in report):
    - random resize / down-up sampling (mimics scaling artifacts)
    - random JPEG re-encode at lower quality (compression artifacts)
    - Gaussian blur, motion blur
    - color jitter, random crop / pad
    - additive noise

CLI:
    python train_augmented.py --timeout_seconds 1800
"""

from __future__ import annotations

import argparse
import sys
import time

from _lib import io as _io
from _lib import seed as _seed
# from _lib.model import build_appendix_b_cnn
# from _lib.calibration import pick_threshold_for_fpr, write_threshold_json


def build_augmentation_pipeline():
    """Return a torchvision-style transform composing the chosen augmentations."""
    raise NotImplementedError("Task 1.3: define augmentation pipeline")


def build_model_or_load_task02():
    """Either build_model() from scratch or load artifacts/task02/best.pt."""
    raise NotImplementedError("Task 1.3: build or fine-tune from Task 2 checkpoint")


def train_one_epoch(model, dataloader, optimizer, criterion):
    raise NotImplementedError("Task 1.3: training step (with augmentation)")


def calibrate_threshold(model, calibration_aug_loader, target_fpr: float = 0.20) -> float:
    """Calibrate against data/calibration_augmented/."""
    raise NotImplementedError("Task 1.3: calibrate threshold on calibration_augmented")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.3: robust training")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget — training MUST stop in time.")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("task03")
    print(f"train_augmented.py: not yet implemented (timeout_seconds={args.timeout_seconds}, out={out_dir}).")

    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        # TODO: train_one_epoch(...) with augmentations
        # TODO: write best.pt regularly to survive timeout-kill
        break  # remove once implemented

    # TODO: threshold = calibrate_threshold(model, calibration_aug_loader)
    # TODO: write_threshold_json(out_dir / "threshold.json", threshold, target_fpr=0.20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
