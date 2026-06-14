"""
Task 1.2 -- Data Preparation  (part of the 35-point Modeling task)

Reads:    artifacts/clean/train_manifest.csv  (from clean.py)
          data/train/, data/calibration/, data/validation/, data/validation_augmented/
Writes:   artifacts/prepared/
            X_fit.mmap, X_hold.mmap, X_cal.mmap, X_val.mmap, X_va.mmap
              -- uint8 memmaps, shape (N, 224, 224, 3)
            y_{fit,hold,cal,val,va}.npy, src_{fit,hold,cal,val,va}.npy
            n_{fit,hold,cal,val,va}.npy   -- actual row counts
            F_{fit,hold,cal,val,va}.npy   -- 101-dim engineered features
            mean.npy, std.npy             -- per-channel stats from fit fold

Scope:
    Materialize everything train.py needs so the 1800s training budget is
    not burned on decoding + feature extraction.

WARNING (PDF §Submission Guidelines):
    Do NOT process data/predict/ here. The contents of data/predict/ may be
    swapped between training and evaluation, so anything derived from it
    must be computed inside predict.py at runtime.

CLI:
    python prepare.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _lib import io as _io
from _lib import seed as _seed
from _lib.data import binarize_label
from _lib.features import features_from_uint8

HOLDOUT_FRAC = 0.10
SPLIT_SEED   = 0
IMG_SIZE     = _io.IMG_SIZE  # 224
N_JOBS       = min(8, os.cpu_count() or 1)   # parallel decode workers (grader = --cpus 8)


# ---------------------------------------------------------------------------
# Fold assignment -- matches notebook A.1 exactly (same seed, same strategy)
# ---------------------------------------------------------------------------

def _make_fold_assignment(manifest: pd.DataFrame) -> pd.Series:
    """Stratified 90/10 holdout split by source_class, seed=0.

    Reproduces the notebook's fold_assignment.csv logic so the script-pipeline
    split is identical to the one used during notebook development.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    fold_col = ["fit"] * len(manifest)
    for cls in sorted(manifest["source_class"].unique()):
        idx_cls = manifest.index[manifest["source_class"] == cls].to_numpy()
        perm = rng.permutation(len(idx_cls))
        n_hold = int(round(len(idx_cls) * HOLDOUT_FRAC))
        hold_positions = set(perm[:n_hold].tolist())
        for pos, gi in enumerate(idx_cls):
            if pos in hold_positions:
                fold_col[gi] = "holdout"
    return pd.Series(fold_col, index=manifest.index)


# ---------------------------------------------------------------------------
# Image cache builder
# ---------------------------------------------------------------------------

def _decode_to_uint8(buf: bytes):
    """Decode image bytes -> (224, 224, 3) uint8, or None if unreadable."""
    arr = _io.clean_image(buf)
    if arr is None:
        return None
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _build_labeled_cache(
    name: str,
    split_dir,
    out_dir,
    row_filter: dict | None = None,
):
    """Decode and cache one labeled split as a uint8 memmap.

    row_filter: {parquet_filename: {row_idx: (binary_label, source_class)}}
    If None, reads all rows from split_dir and binarizes labels automatically.
    """
    mmap_path = out_dir / f"X_{name}.mmap"
    y_path    = out_dir / f"y_{name}.npy"
    src_path  = out_dir / f"src_{name}.npy"
    n_path    = out_dir / f"n_{name}.npy"

    if mmap_path.exists() and y_path.exists() and n_path.exists():
        n = int(np.load(n_path)[0])
        print(f"  {name}: cached ({n} rows)")
        return

    # Count total rows for pre-allocation
    files = sorted(split_dir.glob("*.parquet"))
    if row_filter is not None:
        n_total = sum(len(v) for v in row_filter.values())
    else:
        n_total = sum(pq.read_metadata(p).num_rows for p in files)

    X = np.lib.format.open_memmap(
        str(mmap_path), mode="w+", dtype=np.uint8,
        shape=(n_total, IMG_SIZE, IMG_SIZE, 3),
    )
    y   = np.zeros(n_total, dtype=np.int64)
    src = np.zeros(n_total, dtype=np.int64)

    from joblib import Parallel, delayed

    write_idx = 0
    t0 = time.monotonic()
    DCHUNK = 1024   # bounded-memory parallel-decode batch (~150 MB of arrays in flight)
    for path in files:
        pf = path.name
        if row_filter is not None and pf not in row_filter:
            continue
        table = pq.read_table(path, columns=["image", "source_class"])
        imgs   = table.column("image")
        labels = table.column("source_class")

        # Ordered work list for this file: (image_bytes, binary_label, source_class).
        work: list[tuple] = []
        for i in range(len(table)):
            if row_filter is not None and i not in row_filter.get(pf, {}):
                continue
            if row_filter is not None:
                bl, sc = row_filter[pf][i]
            else:
                lab = int(labels[i].as_py())
                bl, sc = binarize_label(lab), lab
            work.append((imgs[i].as_py(), int(bl), int(sc)))

        # Decode in parallel (PIL releases the GIL); write valid rows back in order so the
        # X/y/src caches stay aligned and identical regardless of worker count.
        for cs in range(0, len(work), DCHUNK):
            batch = work[cs:cs + DCHUNK]
            arrs = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(_decode_to_uint8)(buf) for buf, _, _ in batch
            )
            for (buf, bl, sc), arr in zip(batch, arrs):
                if arr is None:
                    continue
                X[write_idx] = arr
                y[write_idx]   = bl
                src[write_idx] = sc
                write_idx += 1

    X.flush()
    np.save(str(y_path),   y[:write_idx])
    np.save(str(src_path), src[:write_idx])
    np.save(str(n_path),   np.array([write_idx]))
    elapsed = time.monotonic() - t0
    print(f"  {name}: {write_idx} rows in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def _compute_norm_stats(out_dir) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std from the fit fold (float [0,1] space)."""
    mean_path = out_dir / "mean.npy"
    std_path  = out_dir / "std.npy"
    if mean_path.exists() and std_path.exists():
        mean = np.load(str(mean_path))
        std  = np.load(str(std_path))
        print(f"  norm stats: cached  mean={mean.round(4).tolist()}")
        return mean, std

    n_fit = int(np.load(str(out_dir / "n_fit.npy"))[0])
    X_fit = np.lib.format.open_memmap(
        str(out_dir / "X_fit.mmap"), mode="r", dtype=np.uint8,
        shape=(n_fit, IMG_SIZE, IMG_SIZE, 3),
    )
    sums   = np.zeros(3, dtype=np.float64)
    sumsqs = np.zeros(3, dtype=np.float64)
    n_pix  = 0
    CHUNK  = 256
    for i in range(0, n_fit, CHUNK):
        block = X_fit[i:i+CHUNK].astype(np.float32) / 255.0
        sums   += block.sum(axis=(0, 1, 2))
        sumsqs += (block * block).sum(axis=(0, 1, 2))
        n_pix  += block.shape[0] * IMG_SIZE * IMG_SIZE

    mean = (sums / n_pix).astype(np.float32)
    std  = np.sqrt(np.maximum(sumsqs / n_pix - mean.astype(np.float64)**2, 1e-12)).astype(np.float32)
    np.save(str(mean_path), mean)
    np.save(str(std_path),  std)
    print(f"  norm stats: mean={mean.round(4).tolist()}  std={std.round(4).tolist()}")
    return mean, std


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(name: str, out_dir) -> None:
    feat_path = out_dir / f"F_{name}.npy"
    if feat_path.exists():
        print(f"  features {name}: cached")
        return

    n = int(np.load(str(out_dir / f"n_{name}.npy"))[0])
    X = np.lib.format.open_memmap(
        str(out_dir / f"X_{name}.mmap"), mode="r", dtype=np.uint8,
        shape=(n, IMG_SIZE, IMG_SIZE, 3),
    )
    t0 = time.monotonic()
    F = features_from_uint8(X)
    np.save(str(feat_path), F)
    print(f"  features {name}: {F.shape[1]}-dim in {time.monotonic()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: prepare features for training")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()
    _start = time.monotonic()

    _seed.set_deterministic(0)
    out_dir = _io.ensure_artifact_dir("prepared")

    # ------------------------------------------------------------------
    # Step 1: read manifest, build fold split
    # ------------------------------------------------------------------
    manifest_path = _io.ARTIFACTS_ROOT / "clean" / "train_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found -- run clean.py first", file=sys.stderr)
        return 1

    manifest = pd.read_csv(str(manifest_path))
    manifest = manifest[manifest["is_valid"] == 1].reset_index(drop=True)
    manifest["fold"] = _make_fold_assignment(manifest)
    print(f"train manifest: {len(manifest)} valid rows  "
          f"fit={int((manifest['fold']=='fit').sum())}  "
          f"holdout={int((manifest['fold']=='holdout').sum())}")

    # Build row-filter dicts: {parquet_file: {row_idx: (binary_label, source_class)}}
    fit_rows: dict[str, dict[int, tuple]] = {}
    hold_rows: dict[str, dict[int, tuple]] = {}
    for _, row in manifest.iterrows():
        pf = str(row["parquet_file"])
        ri = int(row["row_idx"])
        bl = int(row["binary_label"])
        sc = int(row["source_class"])
        target = fit_rows if row["fold"] == "fit" else hold_rows
        target.setdefault(pf, {})[ri] = (bl, sc)

    # ------------------------------------------------------------------
    # Step 2: decode and cache image arrays
    # ------------------------------------------------------------------
    print("decoding images (cached after first run)...")
    train_dir = _io.DATA_ROOT / "train"
    _build_labeled_cache("fit",  train_dir, out_dir, fit_rows)
    _build_labeled_cache("hold", train_dir, out_dir, hold_rows)
    _build_labeled_cache("cal",  _io.DATA_ROOT / "calibration",         out_dir)
    _build_labeled_cache("val",  _io.DATA_ROOT / "validation",          out_dir)
    _build_labeled_cache("va",   _io.DATA_ROOT / "validation_augmented", out_dir)

    # ------------------------------------------------------------------
    # Step 3: normalization stats from fit fold
    # ------------------------------------------------------------------
    print("computing normalization stats...")
    _compute_norm_stats(out_dir)

    # ------------------------------------------------------------------
    # Step 4: feature extraction
    # ------------------------------------------------------------------
    print("extracting engineered features...")
    for name in ("fit", "hold", "cal", "val", "va"):
        _extract_features(name, out_dir)

    elapsed = time.monotonic() - _start
    print(f"\nprepare.py done in {elapsed:.1f}s  (budget={args.timeout_seconds}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
