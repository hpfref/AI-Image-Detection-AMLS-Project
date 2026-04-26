"""
Task 1.3 — Robust inference  (part of the 30-point Augmentation task)

Reads:    artifacts/task03/best.pt
          artifacts/task03/threshold.json
          data/predict/
Writes:   artifacts/task03/predictions.csv    (row_id, predicted_label)

Output format identical to Task 1.2 (PDF §1.3 cross-references §1.2):
    row_id,predicted_label
    0,1
    1,0
    2,1

CLI:
    python predict_augmented.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import sys

from _lib import io as _io
from _lib import seed as _seed
# from _lib.calibration import read_threshold_json
# from _lib.data import PredictDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.3: predict on data/predict/")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget for this script.")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("task03")
    print(f"predict_augmented.py: not yet implemented (timeout_seconds={args.timeout_seconds}, out={out_dir}).")

    # TODO: model = load_checkpoint(out_dir / "best.pt")          # see predict.py for shape
    # TODO: threshold = read_threshold_json(out_dir / "threshold.json")
    # TODO: rows = predict(model, PredictDataset(_io.DATA_ROOT / "predict"), threshold)
    # TODO: write_predictions_csv(rows, out_dir / "predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
