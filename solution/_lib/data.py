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


def binarize_label(source_class: int) -> int:
    """0 stays 0 (real); 1..5 collapse to 1 (ai_generated)."""
    return 0 if source_class == 0 else 1


class LabeledImageDataset:
    """Iterable over (image_tensor, binary_label) pairs.

    Used by clean.py / prepare.py / train.py / train_augmented.py for the
    labeled splits.
    """

    def __init__(self, split_dir, transform=None):
        # TODO: stream parquet via _lib.io.read_parquet_split, decode images,
        # apply `transform`, yield (image_tensor, binarize_label(source_class)).
        self.split_dir = split_dir
        self.transform = transform

    def __iter__(self):
        raise NotImplementedError("Task 1.1/1.2: implement labeled dataset iterator")


class PredictDataset:
    """Iterable over (row_id, image_tensor) pairs for data/predict/.

    Used by predict.py and predict_augmented.py only.
    """

    def __init__(self, predict_dir, transform=None):
        self.predict_dir = predict_dir
        self.transform = transform

    def __iter__(self):
        raise NotImplementedError("Task 1.2: implement predict dataset iterator")
