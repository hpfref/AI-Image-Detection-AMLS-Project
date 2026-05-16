"""
Task 1.1 — Dataset Exploration & Cleaning  (15/100 points)

Reads:    data/train/                 (read-only mount)
          data/calibration/, data/validation/  (also useful for stats)
Writes:   artifacts/clean/            (cleaned training data or a regen script)

Scope (PDF §1.1):
    1. Analyze the training data and report:
         - class distribution (six original classes 0..5; also collapsed 0 vs. 1..5)
         - image-size distribution (width, height, channels, aspect ratios)
         - basic descriptive statistics
         - any characteristics that could be used to deduce the class
           (this is the most important finding to call out)
    2. Construct a DETERMINISTIC cleaning pipeline. Justify each choice in
       report/report.md §1.1 (don't just apply a fixed recipe).
    3. Output a cleaned training dataset (or a regen script) suitable for
       downstream CPU-friendly modeling.

Out of scope:
    - augmentation (PDF: "This task is about exploration and cleaning, not
      augmentation"). Augmentation belongs in train_augmented.py.

Implementation (Task 1.1 findings):
    - Dataset is already clean: no corrupt rows, no format variation, no missing labels.
    - Key finding: AI images are always square (SD/MJ=320x320, DALL-E3=270x270),
      real images are non-square and variable. Aspect ratio is a near-perfect classifier
      and must be neutralised by the preprocessing transform.
    - Cleaning step: shorter-edge resize + center-crop to 224x224 (see _lib/io.py::clean_image).
    - Output: artifacts/clean/train_manifest.csv listing every train row with validity flag.
      prepare.py reads this to skip corrupt rows without re-validating.

CLI:
    python clean.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Must precede numpy/PIL imports to avoid the dual-OpenMP crash on systems
# that load both Intel MKL and LLVM OpenMP runtimes simultaneously.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pyarrow.parquet as pq

from _lib import io as _io
from _lib import seed as _seed
from _lib.data import binarize_label

CLASS_NAMES = {0: "real", 1: "SD2.1", 2: "SDXL", 3: "SD3", 4: "DALL-E3", 5: "Midjourney"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_classes(split_dir: Path) -> dict[int, int]:
    """Read only source_class column — never touches image bytes."""
    counts: dict[int, int] = {}
    for path in sorted(split_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=["source_class"]):
            col = batch.column("source_class").to_numpy(zero_copy_only=False)
            for val, cnt in zip(*np.unique(col, return_counts=True)):
                counts[int(val)] = counts.get(int(val), 0) + int(cnt)
    return counts


def _sample_labeled(split_dir: Path, per_file: int, seed: int) -> list[tuple[int, bytes]]:
    """Return up to per_file rows per parquet file as (source_class, image_bytes).

    Reads only the first batch from each file — deterministic and fast.
    """
    rng = np.random.default_rng(seed)
    records: list[tuple[int, bytes]] = []
    for path in sorted(split_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=per_file * 2, columns=["image", "source_class"]))
        n = len(batch)
        idxs = rng.choice(n, size=min(per_file, n), replace=False)
        imgs = batch.column("image")
        labels = batch.column("source_class")
        for i in idxs:
            records.append((int(labels[i].as_py()), imgs[i].as_py()))
    return records


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def explore_class_distribution() -> None:
    """Count original classes 0..5 and the collapsed binary distribution."""
    print("\n=== Class Distribution ===")
    for split in ("train", "calibration", "validation"):
        counts = _count_classes(_io.DATA_ROOT / split)
        total = sum(counts.values())
        real = counts.get(0, 0)
        ai = total - real
        print(f"\n{split} (n={total})")
        for cls_id in sorted(counts):
            n = counts[cls_id]
            print(f"  {cls_id} {CLASS_NAMES.get(cls_id, '?'):12s}: {n:6d}  ({100*n/total:.1f}%)")
        print(f"  -> binary: real={real}  ai={ai}  ai_share={100*ai/total:.1f}%")


def explore_image_sizes() -> None:
    """Width / height / aspect-ratio distribution per class."""
    print("\n=== Image Size & Format ===")
    import io as _std_io
    from PIL import Image

    records = _sample_labeled(_io.DATA_ROOT / "train", per_file=150, seed=1)
    rows = []
    for label, buf in records:
        try:
            with Image.open(_std_io.BytesIO(buf)) as im:
                rows.append({"cls": label, "w": im.width, "h": im.height, "bytes": len(buf)})
        except Exception:
            pass

    print(f"Sampled {len(records)} images, {len(records) - len(rows)} unreadable")
    print("\nPer-class median (w x h), file size:")
    for cls_id in sorted(set(r["cls"] for r in rows)):
        cr = [r for r in rows if r["cls"] == cls_id]
        print(f"  {cls_id} {CLASS_NAMES.get(cls_id, '?'):12s}: "
              f"{np.median([r['w'] for r in cr]):.0f}x{np.median([r['h'] for r in cr]):.0f}  "
              f"{np.median([r['bytes'] for r in cr]) / 1024:.1f} KB  (n={len(cr)})")


def explore_descriptive_stats() -> None:
    """Mean/std RGB per source class on cleaned 224x224 images."""
    print("\n=== Descriptive Statistics (RGB mean/std per class) ===")
    records = _sample_labeled(_io.DATA_ROOT / "train", per_file=60, seed=2)

    rows = []
    for label, buf in records:
        arr = _io.clean_image(buf)
        if arr is None:
            continue
        rows.append({
            "cls": label,
            "r": float(arr[..., 0].mean()),
            "g": float(arr[..., 1].mean()),
            "b": float(arr[..., 2].mean()),
            "std": float(arr.std()),
        })

    by_class: dict[int, list] = {}
    for r in rows:
        by_class.setdefault(r["cls"], []).append(r)

    print(f"{'class':20s} {'mean_r':>8} {'mean_g':>8} {'mean_b':>8} {'std':>8}")
    for cls_id in sorted(by_class):
        rs = by_class[cls_id]
        name = f"{cls_id}:{CLASS_NAMES.get(cls_id, '?')}"
        print(f"{name:20s} "
              f"{np.mean([r['r'] for r in rs]):8.3f} "
              f"{np.mean([r['g'] for r in rs]):8.3f} "
              f"{np.mean([r['b'] for r in rs]):8.3f} "
              f"{np.mean([r['std'] for r in rs]):8.3f}")


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

def clean_pipeline(artifact_dir: Path, deadline: float) -> None:
    """Validate every train image within the time budget. Write manifest CSV.

    Manifest columns: parquet_file, row_idx, source_class, binary_label, is_valid
    prepare.py reads this to skip corrupt rows without re-validating.
    """
    print("\n=== Cleaning Pipeline: validating train images ===")
    train_dir = _io.DATA_ROOT / "train"
    manifest_path = artifact_dir / "train_manifest.csv"

    n_total = n_valid = n_invalid = 0
    timed_out = False

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["parquet_file", "row_idx", "source_class", "binary_label", "is_valid"])

        for path in sorted(train_dir.glob("*.parquet")):
            if timed_out:
                break
            pf = pq.ParquetFile(path)
            row_offset = 0
            for batch in pf.iter_batches(batch_size=64, columns=["image", "source_class"]):
                if time.perf_counter() > deadline:
                    timed_out = True
                    break
                imgs = batch.column("image")
                labels = batch.column("source_class")
                for i in range(len(imgs)):
                    label = int(labels[i].as_py())
                    valid = _io.clean_image(imgs[i].as_py()) is not None
                    writer.writerow([path.name, row_offset + i,
                                     label, binarize_label(label), int(valid)])
                    n_total += 1
                    n_valid += int(valid)
                    n_invalid += int(not valid)
                row_offset += len(imgs)

    if timed_out:
        print(f"Time limit reached after {n_total} rows.")
    drop_pct = 100 * n_invalid / max(1, n_total)
    print(f"Validated {n_total} rows: {n_valid} valid, {n_invalid} invalid ({drop_pct:.2f}% drop)")
    print(f"Manifest -> {manifest_path}")


def save_cleaned_dataset(artifact_dir: Path, deadline: float) -> None:
    """Write to artifacts/clean/ — input for prepare.py."""
    clean_pipeline(artifact_dir, deadline)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.1: explore and clean training data")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget for this script (PDF §Submission Guidelines)")
    args = parser.parse_args()

    start = time.perf_counter()
    deadline = start + args.timeout_seconds - 30  # 30 s safety margin before hard kill

    _seed.set_deterministic(0)
    artifact_dir = _io.ensure_artifact_dir("clean")

    explore_class_distribution()
    explore_image_sizes()
    explore_descriptive_stats()
    save_cleaned_dataset(artifact_dir, deadline)

    print(f"\nclean.py done in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
