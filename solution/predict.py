"""
Task 1.2 -- Inference  (part of the 35-point Modeling task)

Reads:    artifacts/task02/best.pt           -- CNN state dict + config
          artifacts/task02/rf_model.pkl      -- classical model
          artifacts/task02/threshold.json    -- ensemble threshold + alpha
          data/predict/                      -- parquet, columns row_id, image
Writes:   artifacts/task02/predictions.csv  -- row_id,predicted_label

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
import csv
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

from _lib import io as _io
from _lib import seed as _seed
from _lib.calibration import read_threshold_json
from _lib.features import features_single
from _lib.model import build_cnn_bn


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: predict on data/predict/")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()
    _start = time.monotonic()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("task02")

    # ------------------------------------------------------------------
    # Load model and threshold
    # ------------------------------------------------------------------
    ckpt_path = out_dir / "best.pt"
    rf_path   = out_dir / "rf_model.pkl"
    thr_path  = out_dir / "threshold.json"

    for p in (ckpt_path, rf_path, thr_path):
        if not p.exists():
            print(f"ERROR: {p} not found -- run train.py first", file=sys.stderr)
            return 1

    import joblib
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cnn  = build_cnn_bn(k=ckpt["k"])
    cnn.load_state_dict(ckpt["state"])
    cnn.eval()
    mean = ckpt["mean"]  # (3,) float32
    std  = ckpt["std"]   # (3,) float32
    img_size = ckpt["img_size"]

    # Convert mean/std to tensors shaped (1, 3, 1, 1) for CHW normalization
    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_t  = torch.from_numpy(std).view(1, 3, 1, 1)

    rf_pipe   = joblib.load(str(rf_path))
    thr_data  = read_threshold_json(thr_path)
    thr       = float(thr_data["thr"])
    alpha     = float(thr_data["alpha"])

    # ------------------------------------------------------------------
    # Inference loop over data/predict/
    # ------------------------------------------------------------------
    predict_dir = _io.DATA_ROOT / "predict"
    if not predict_dir.exists():
        print(f"ERROR: {predict_dir} not found", file=sys.stderr)
        return 1

    rows: list[tuple[int, int]] = []
    n_processed = 0

    for row_id, img_bytes in _io.read_predict_split(predict_dir):
        arr = _io.clean_image(img_bytes)  # (224, 224, 3) float32 [0, 1]
        if arr is None:
            # Fallback: predict real (0) for unreadable images
            rows.append((row_id, 0))
            continue

        # CNN score: normalize (HWC) -> permute to CHW -> resize -> forward
        arr_norm = (arr - mean) / std            # (224, 224, 3)
        t = torch.from_numpy(arr_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        if t.shape[-1] != img_size:
            t = F.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            p_cnn = float(torch.softmax(cnn(t), dim=1)[0, 1])

        # RF score: (H, W, 3) uint8 -> 101-dim features
        arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
        feat   = features_single(arr_u8).reshape(1, -1)
        p_rf   = float(rf_pipe.predict_proba(feat)[0, 1])

        p = alpha * p_cnn + (1 - alpha) * p_rf
        rows.append((row_id, int(p >= thr)))
        n_processed += 1

    # ------------------------------------------------------------------
    # Write predictions.csv
    # ------------------------------------------------------------------
    csv_path = out_dir / "predictions.csv"
    with open(str(csv_path), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "predicted_label"])
        for row_id, label in rows:
            writer.writerow([row_id, label])

    elapsed = time.monotonic() - _start
    n_ai = sum(1 for _, lbl in rows if lbl == 1)
    print(f"predict.py done: {n_processed} images  "
          f"AI={n_ai}  real={len(rows)-n_ai}  "
          f"in {elapsed:.1f}s  -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
