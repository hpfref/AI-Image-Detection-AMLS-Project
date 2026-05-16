"""
Parquet I/O, image decoding, and artifact-path helpers.

Dataset structure (PDF §Dataset structure):
    data/train/                        parquet, columns image: binary, source_class: int8
    data/calibration/                  parquet, same schema
    data/calibration_augmented/        parquet, same schema
    data/validation/                   parquet, same schema
    data/validation_augmented/         parquet, same schema
    data/predict/                      parquet, columns row_id: int32, image: binary

Artifact layout produced under solution/artifacts/:
    artifacts/clean/                   train manifest CSV (Task 1.1)
    artifacts/prepared/                tensor caches / feature matrices (Task 1.2)
    artifacts/task02/best.pt           best Task 2 checkpoint
    artifacts/task02/threshold.json    calibrated threshold (FPR <= 0.20)
    artifacts/task02/predictions.csv   row_id,predicted_label
    artifacts/task03/best.pt           best Task 3 (robust) checkpoint
    artifacts/task03/threshold.json
    artifacts/task03/predictions.csv
"""

from __future__ import annotations

import io as _stdlib_io
from pathlib import Path
from typing import Iterator, Optional, Tuple

DATA_ROOT = Path("data")
ARTIFACTS_ROOT = Path("artifacts")
IMG_SIZE = 224


def read_parquet_split(split_dir: Path) -> Iterator[Tuple[bytes, Optional[int]]]:
    """Yield (image_bytes, source_class) for labeled splits, or (image_bytes, None) for predict."""
    import pyarrow.parquet as pq

    for path in sorted(split_dir.glob("*.parquet")):
        schema = pq.read_schema(path)
        has_label = "source_class" in schema.names
        cols = ["image", "source_class"] if has_label else ["image"]
        table = pq.read_table(path, columns=cols)
        imgs = table.column("image")
        labels = table.column("source_class") if has_label else None
        for i in range(len(table)):
            label = int(labels[i].as_py()) if labels is not None else None
            yield imgs[i].as_py(), label


def read_predict_split(predict_dir: Path) -> Iterator[Tuple[int, bytes]]:
    """Yield (row_id, image_bytes) for the unlabeled predict split."""
    import pyarrow.parquet as pq

    for path in sorted(predict_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["row_id", "image"])
        row_ids = table.column("row_id")
        imgs = table.column("image")
        for i in range(len(table)):
            yield int(row_ids[i].as_py()), imgs[i].as_py()


def decode_image(image_bytes: bytes):
    """Decode binary bytes to a PIL Image in RGB mode."""
    from PIL import Image
    return Image.open(_stdlib_io.BytesIO(image_bytes)).convert("RGB")


def clean_image(image_bytes: bytes):
    """Decode, resize shorter-edge to IMG_SIZE, center-crop to IMG_SIZE x IMG_SIZE.

    Aspect-preserving resize avoids encoding original dimensions as a class signal
    (AI images are always square, real images are not).
    Returns (IMG_SIZE, IMG_SIZE, 3) float32 in [0, 1], or None if unreadable.
    """
    import numpy as np
    from PIL import Image

    try:
        im = Image.open(_stdlib_io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    scale = IMG_SIZE / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    left = (nw - IMG_SIZE) // 2
    top = (nh - IMG_SIZE) // 2
    im = im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    return np.asarray(im, dtype=np.float32) / 255.0


def ensure_artifact_dir(*parts: str) -> Path:
    """Create artifacts/<parts...> if missing, return its Path."""
    path = ARTIFACTS_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
