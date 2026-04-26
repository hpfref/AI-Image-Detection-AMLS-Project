"""
Task 1.4 — Explainability entry point.

Goal (PDF §1.4): reason about why the final model makes its decisions and
where it fails. Pick at least one of the four directions below and justify
the choice in report/report.md §1.4.

Reasonable directions:
    1. saliency / gradient-based explanations
    2. occlusion / perturbation analysis
    3. analysis of false positives and false negatives
    4. comparison of what the model attends to for real vs. AI images

Inputs (read-only):
    ../solution/artifacts/task03/best.pt        (preferred — robust final model)
    ../solution/artifacts/task03/threshold.json
    ../solution/data/validation/                (or validation_augmented/)

Outputs:
    figures and any quantitative tables — referenced from report/report.md.

This script is NOT invoked by the grader's pipeline. It runs locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# TODO: import torch and the shared helpers when implementing.
# from solution._lib.model import build_appendix_b_cnn
# from solution._lib.io import read_parquet_split, decode_image


def load_model_from_artifacts(checkpoint_path: Path):
    """Reconstruct the architecture used in train(_augmented).py and load weights."""
    raise NotImplementedError("Task 1.4: load checkpoint from solution/artifacts/")


def saliency_map(model, image):
    """Direction 1: gradient of the AI-class score wrt input pixels."""
    raise NotImplementedError("Task 1.4: saliency / gradient explanation")


def occlusion_analysis(model, image, patch_size: int = 32, stride: int = 16):
    """Direction 2: slide a mask, measure score drop, build a heatmap."""
    raise NotImplementedError("Task 1.4: occlusion / perturbation analysis")


def analyze_failures(model, dataset, threshold: float):
    """Direction 3: collect FP and FN samples for qualitative inspection."""
    raise NotImplementedError("Task 1.4: failure-mode analysis")


def compare_real_vs_ai(model, dataset):
    """Direction 4: attention statistics differing between real and AI images."""
    raise NotImplementedError("Task 1.4: real-vs-AI attention comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1.4 explainability")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../solution/artifacts/task03/best.pt"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--data_split",
        type=Path,
        default=Path("../solution/data/validation"),
        help="Validation split (parquet files) to draw examples from.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("../report/figures"),
        help="Where to write figures referenced from the report.",
    )
    args = parser.parse_args()

    print(
        "task04/explain.py: stub — pick at least one explainability direction "
        "and implement it. See module docstring."
    )
    # TODO: model = load_model_from_artifacts(args.checkpoint)
    # TODO: produce figures into args.out_dir


if __name__ == "__main__":
    main()
