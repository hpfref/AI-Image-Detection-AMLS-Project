"""
Dataset / DataLoader wrappers and label binarization.

The dataset has six original classes:
    0  real
    1  SD 2.1
    2  SDXL
    3  SD 3
    4  DALL-E 3
    5  Midjourney

Per PDF §1, labels 1..5 are collapsed into a single class `1: ai_generated`.
The binary label is the one used everywhere in the pipeline.
"""

from __future__ import annotations

from _lib import io as _io


def binarize_label(source_class: int) -> int:
    """0 stays 0 (real); 1..5 collapse to 1 (ai_generated)."""
    return 0 if source_class == 0 else 1


class LabeledImageDataset:
    """Iterable over (image_array, binary_label) pairs.

    Streams from parquet; memory footprint is one batch at a time.
    Used by prepare.py, train.py, and train_augmented.py.
    """

    def __init__(self, split_dir, transform=None):
        self.split_dir = split_dir
        self.transform = transform

    def __iter__(self):
        for img_bytes, label in _io.read_parquet_split(self.split_dir):
            if label is None:
                continue
            arr = _io.clean_image(img_bytes)
            if arr is None:
                continue
            if self.transform is not None:
                arr = self.transform(arr)
            yield arr, binarize_label(label)


class PredictDataset:
    """Iterable over (row_id, image_array) pairs for data/predict/.

    Used by predict.py and predict_augmented.py only.
    """

    def __init__(self, predict_dir, transform=None):
        self.predict_dir = predict_dir
        self.transform = transform

    def __iter__(self):
        for row_id, img_bytes in _io.read_predict_split(self.predict_dir):
            arr = _io.clean_image(img_bytes)
            if arr is None:
                continue
            if self.transform is not None:
                arr = self.transform(arr)
            yield row_id, arr
