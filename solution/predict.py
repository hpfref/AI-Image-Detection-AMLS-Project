"""
Task 1.2 — Inference  (part of the 35-point Modeling task)

Reads:    artifacts/task02/best.pt
          artifacts/task02/threshold.json
          data/predict/                       (parquet, columns row_id, image)
Writes:   artifacts/task02/predictions.csv    (row_id, predicted_label)

Output schema (PDF §1.2):
    row_id,predicted_label
    0,1
    1,0
    2,1

CLI:
    python predict.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import sys

from _lib import io as _io
from _lib import seed as _seed
# from _lib.calibration import read_threshold_json
# from _lib.data import PredictDataset


def load_checkpoint(path):
    raise NotImplementedError("Task 1.2: load model from artifacts/task02/best.pt")


def predict(model, predict_loader, threshold: float):
    """Score images, apply threshold, yield (row_id, predicted_label)."""
    raise NotImplementedError("Task 1.2: inference loop with calibrated threshold")


def write_predictions_csv(rows, csv_path) -> None:
    """Write rows to artifacts/task02/predictions.csv with the required header."""
    raise NotImplementedError("Task 1.2: write predictions.csv (header: row_id,predicted_label)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: predict on data/predict/")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget for this script.")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("task02")
    print(f"predict.py: not yet implemented (timeout_seconds={args.timeout_seconds}, out={out_dir}).")

    # TODO: model = load_checkpoint(out_dir / "best.pt")
    # TODO: threshold = read_threshold_json(out_dir / "threshold.json")
    # TODO: rows = predict(model, PredictDataset(_io.DATA_ROOT / "predict"), threshold)
    # TODO: write_predictions_csv(rows, out_dir / "predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
