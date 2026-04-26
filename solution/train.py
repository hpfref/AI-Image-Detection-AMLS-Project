"""
Task 1.2 — Modeling and Tuning under Time Constraints  (35/100 points)

Reads:    artifacts/prepared/              (features from prepare.py)
          data/calibration/                (for threshold calibration)
          data/validation/                 (for independent FPR validation)
          data/validation_augmented/       (also reported per PDF)
Writes:   artifacts/task02/best.pt         (best checkpoint, written REGULARLY)
          artifacts/task02/threshold.json  (calibrated FPR-target threshold)

Goals (PDF §1.2):
    - Maximize recall_ai under fpr_real <= 0.20.
    - Calibrate the operating threshold AUTOMATICALLY on data/calibration/ —
      no manually picked thresholds.
    - Independently validate FPR on data/validation/ (and report on
      data/validation_augmented/ too).
    - Report at least TWO model families in the report (e.g. classical
      engineered-feature baseline + neural model trained from scratch). Only
      the single best pipeline is packaged in this folder.
    - CPU only, no internet, no pretrained downloads at runtime.
    - Local training time <= 5x the elapsed time of train_time_reference.py.

Targets:
    - recall_ai >= 0.8 on data/validation/ (strong solutions push higher).
    - fpr_real <= 0.20 on data/validation/ (hard constraint).

This file uses the Appendix B reference CNN as a starting point (see
_lib/model.py). Justify the final architecture and hyperparameters in the
report.

CLI:
    python train.py --timeout_seconds 1800
"""

from __future__ import annotations

import argparse
import sys
import time

from _lib import io as _io
from _lib import seed as _seed
# from _lib.model import build_appendix_b_cnn
# from _lib.calibration import pick_threshold_for_fpr, write_threshold_json
# from _lib.metrics import recall_ai, fpr_real


def build_model():
    """Construct the model used for Task 1.2.

    Start from build_appendix_b_cnn() (Appendix B) and tune from there. The
    report compares this against at least one other family (classical baseline).
    """
    raise NotImplementedError("Task 1.2: build the chosen model")


def train_one_epoch(model, dataloader, optimizer, criterion):
    raise NotImplementedError("Task 1.2: training step")


def evaluate(model, dataloader):
    """Compute recall_ai, fpr_real, and AI-class scores for calibration."""
    raise NotImplementedError("Task 1.2: evaluation")


def calibrate_threshold(model, calibration_loader, target_fpr: float = 0.20) -> float:
    """Pick the threshold that yields fpr_real <= target_fpr on calibration data."""
    raise NotImplementedError("Task 1.2: calibrate threshold against data/calibration/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: train + calibrate")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget — training MUST stop in time.")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("task02")
    print(f"train.py: not yet implemented (timeout_seconds={args.timeout_seconds}, out={out_dir}).")

    # Skeleton for budget-aware training (write a checkpoint regularly so a
    # timeout-kill still leaves a usable best.pt behind):
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        # TODO: train_one_epoch(...)
        # TODO: if val metric improved: torch.save(model.state_dict(), out_dir / "best.pt")
        break  # remove once implemented

    # TODO: threshold = calibrate_threshold(model, calibration_loader)
    # TODO: write_threshold_json(out_dir / "threshold.json", threshold, target_fpr=0.20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
