"""
Parquet I/O and artifact-path helpers.

Dataset structure (PDF §Dataset structure):
    data/train/                        parquet, columns image: binary, source_class: int8
    data/calibration/                  parquet, same schema
    data/calibration_augmented/        parquet, same schema
    data/validation/                   parquet, same schema
    data/validation_augmented/         parquet, same schema
    data/predict/                      parquet, columns row_id: int32, image: binary

Artifact layout we produce under solution/artifacts/ at runtime:
    artifacts/clean/                   cleaned training data (Task 1.1)
    artifacts/prepared/                tensor caches / feature matrices (Task 1.2)
    artifacts/task02/best.pt           best Task 2 checkpoint
    artifacts/task02/threshold.json    calibrated threshold (FPR <= 0.20)
    artifacts/task02/predictions.csv   row_id,predicted_label
    artifacts/task03/best.pt           best Task 3 (robust) checkpoint
    artifacts/task03/threshold.json
    artifacts/task03/predictions.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple


# Conventional roots — pipeline scripts use these instead of hardcoding strings.
DATA_ROOT = Path("data")
ARTIFACTS_ROOT = Path("artifacts")


def read_parquet_split(split_dir: Path) -> Iterator[Tuple[bytes, int | None]]:
    """Yield (image_bytes, label_or_None) over all parquet files in a split.

    For labeled splits (train / calibration / validation, with or without
    _augmented suffix), label is the int8 source_class. For data/predict/,
    label is None.
    """
    raise NotImplementedError("Task 1.1: implement parquet streaming reader")


def decode_image(image_bytes: bytes):
    """Decode the binary image column to a numpy / PIL image."""
    raise NotImplementedError("Task 1.1: implement image decoding")


def ensure_artifact_dir(*parts: str) -> Path:
    """Create artifacts/<parts...> if missing, return its Path."""
    path = ARTIFACTS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
