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

CLI:
    python clean.py --timeout_seconds 600
"""

from __future__ import annotations

import argparse
import sys

from _lib import io as _io
from _lib import seed as _seed


def explore_class_distribution() -> None:
    """Count original classes 0..5 and the collapsed binary distribution."""
    raise NotImplementedError("Task 1.1: class distribution")


def explore_image_sizes() -> None:
    """Width / height / channel / aspect-ratio distribution per class."""
    raise NotImplementedError("Task 1.1: image-size distribution")


def explore_descriptive_stats() -> None:
    """E.g. mean intensity, file-size in bytes, JPEG vs. PNG ratio, EXIF presence."""
    raise NotImplementedError("Task 1.1: descriptive statistics")


def clean_pipeline() -> None:
    """Apply the deterministic cleaning steps decided based on exploration.

    Examples of decisions to justify in the report:
        - resize / center-crop to a fixed resolution suitable for CPU training
        - drop duplicates (hash-based)
        - drop unusable samples (corrupt bytes, wrong channel count, tiny images)
        - normalize color space / strip alpha
    """
    raise NotImplementedError("Task 1.1: deterministic cleaning")


def save_cleaned_dataset() -> None:
    """Write to artifacts/clean/ — input for prepare.py."""
    raise NotImplementedError("Task 1.1: write cleaned dataset under artifacts/clean/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.1: explore and clean training data")
    parser.add_argument("--timeout_seconds", type=int, required=True,
                        help="Hard runtime budget for this script (PDF §Submission Guidelines)")
    args = parser.parse_args()

    _seed.set_deterministic(0)
    _io.ensure_artifact_dir("clean")

    print(f"clean.py: not yet implemented (timeout_seconds={args.timeout_seconds}).")
    # TODO: explore_class_distribution()
    # TODO: explore_image_sizes()
    # TODO: explore_descriptive_stats()
    # TODO: clean_pipeline()
    # TODO: save_cleaned_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
