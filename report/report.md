# AMLS 2026 Exercise — AI Image Detection — Report

Up to 8 pages, 10pt. Compile this to `report.pdf` for submission.

## Team

- Hannes Pohnke (469945)

---

## §1.1 Dataset Exploration & Cleaning (15 pts)

### Dataset overview

Train set: 29,688 images (7 parquet files); calibration: 1,924; validation: 1,124; predict: 100.
All images are JPEG-encoded RGB. No corrupt rows, no format variation, no missing labels were found.
The dataset is clean; the preprocessing steps below are motivated by label leakage, not data quality issues.

### Class distribution

Six source classes: 0=real, 1=SD 2.1, 2=SDXL, 3=SD 3, 4=DALL-E 3, 5=Midjourney. Classes 1-5 collapse to `ai_generated` for the binary task.
The binary split is consistent across all splits at ~83% AI / 17% real, with the six AI classes roughly balanced within the AI group.

![Class distribution](figures/fig1_class_distribution.png)

### Image-size distribution and label leakage

Inspecting image dimensions across a 2,000-image train sample reveals a near-perfect class signal in image size alone:

| Class | Observed dimensions | Square? |
|---|---|---|
| real (0) | variable, median 640x480, range 320x201 to 640x640 | No |
| SD 2.1 (1) | exactly 320x320 | Yes |
| SDXL (2) | exactly 320x320 | Yes |
| SD 3 (3) | exactly 320x320 | Yes |
| DALL-E 3 (4) | exactly 270x270 | Yes |
| Midjourney (5) | exactly 320x320 | Yes |

Every AI image is square; real images are not. Aspect ratio alone (`width == height`) is a near-perfect binary classifier without looking at content. DALL-E 3 is further separable by its distinct 270 px size.

### Descriptive statistics

SD 3 is notably warmer and brighter than all other classes (mean red channel 0.560 vs 0.44-0.50 for others). SDXL has the lowest pixel standard deviation (0.191), consistent with its over-smoothed appearance. Neither metric cleanly separates real from AI in general. Original JPEG file size also correlates with class (real ~50 KB vs AI ~18-32 KB), but this reflects different compression settings per generator, not image content, and would not generalise to a holdout with re-encoded images. It is not used as a feature.

| Class | mean R | mean G | mean B | pixel std | file size (KB) |
|---|---|---|---|---|---|
| real | 0.463 | 0.450 | 0.421 | 0.247 | 49.6 |
| SD 2.1 | 0.486 | 0.466 | 0.425 | 0.225 | 31.7 |
| SDXL | 0.503 | 0.477 | 0.435 | 0.191 | 26.4 |
| SD 3 | 0.560 | 0.547 | 0.490 | 0.270 | 29.2 |
| DALL-E 3 | 0.482 | 0.452 | 0.410 | 0.251 | 17.7 |
| Midjourney | 0.445 | 0.429 | 0.385 | 0.223 | 23.8 |

*Table 1: per-class mean RGB, pixel std, and original JPEG file size (600-image sample from train).*

### Visual characteristics

![Sample grid](figures/fig2_sample_grid.png)

Real photographs are casual snapshots with natural imperfections and no particular aesthetic intent.

**DALL-E 3**: most obviously AI at a glance. Images look plastic, animated, or illustrated with unnatural lighting and an uncanny rendered quality.

**SD 2.1**: more realistic lighting than DALL-E 3, but frequent structural and logical errors: wrong number of fingers, malformed faces, broken geometry in complex objects.

**SD 3**: similar character to SD 2.1, more realistic lighting, still frequently obvious due to logical errors in complex structures.

**SDXL**: fewer logical errors than earlier SD models, but images look heavily filtered: oversaturated, high contrast, artificial-feeling lighting.

**Midjourney**: hardest to detect. The best outputs look like high-quality camera photographs. The main tell is a cinematic "too perfect" quality that real casual photos rarely have. Does have outliers that are obviously stylised.

The most reliable detection cues, roughly in order: (1) logical errors in anatomy and geometry, (2) plastic/animated (most obvious in DALL-E 3), (3) overly polished, filmlike composition that no casual photographer produces.

### Deterministic cleaning pipeline

**Resize shorter-edge and center-crop to 224x224.** Image dimensions are a near-perfect class signal. Mapping all images to a fixed square canvas via aspect-preserving resize followed by center-crop removes both the square/non-square distinction and the 270 vs 320 px difference between generators. Stretching was rejected because it encodes the original aspect ratio in pixel coordinates. 224x224 matches the reference CNN in Appendix B.

**Convert to RGB.** Defensive: all images are already RGB, but this ensures consistent 3-channel input for any downstream model.

**Drop unreadable rows.** A 500-image smoke test found zero corrupt rows. The guard quantifies the drop rate on the full dataset and prevents training failures.

## §1.2 Modeling & Tuning under Time Constraints (35 pts)

PDF §1.2 deliverables:

- [ ] At least **two model families** compared (e.g. classical engineered-feature
      baseline + neural model trained from scratch). Only the best is packaged
      in [solution/](../solution/), but both are reported here.
- [ ] Hyperparameter search and ablations — impact of architecture, loss,
      optimizer, learning rate, threshold.
- [ ] **Calibration protocol**: how `data/calibration/` was used to pick the
      threshold for FPR <= 20%; the calibration is automated, not a hand-set
      threshold.
- [ ] Independent validation on `data/validation/` and a parallel report on
      `data/validation_augmented/`.
- [ ] Metrics table: recall_ai, FPR_real, accuracy, confusion matrix.
      Target: recall_ai >= 0.8 with FPR <= 0.20.
- [ ] CPU-budget evidence — local training elapsed vs.
      [train_time_reference.py](../train_time_reference.py) (must be <= 5x).

## §1.3 Data Augmentation & Feature Engineering (30 pts)

PDF §1.3 deliverables:

- [ ] Augmentation strategy and **why** these transforms (scale, JPEG compression,
      blur, etc.) reflect realistic distribution shift.
- [ ] Whether the model was trained from scratch or fine-tuned from
      `artifacts/task02/best.pt`.
- [ ] Comparison of Task 2 vs. Task 3 model on **both** `data/validation/` and
      `data/validation_augmented/` (same threshold-calibration protocol, same
      CPU budget).
- [ ] Target: recall_ai >= 0.6 on `data/validation_augmented/` with FPR <= 0.20.

## §1.4 Explainability (20 pts)

PDF §1.4 deliverables:

- [ ] Method choice and rationale — saliency maps, occlusion / perturbation,
      FP/FN inspection, real-vs-AI attention comparison.
- [ ] Visual or quantitative examples.
- [ ] **Critical discussion** — are the explanations plausible? Do they reveal
      shortcut features or dataset bias?
- [ ] Code lives in [task04_explainability/](../task04_explainability/).

---

## Appendix — Submission checklist

- [ ] `solution/Dockerfile` builds an image <= 4 GB
- [ ] No internet at runtime (`--network none` works)
- [ ] `python clean.py --timeout_seconds 600` runs to completion
- [ ] `python prepare.py --timeout_seconds 600` runs (does NOT touch `data/predict/`)
- [ ] `python train.py --timeout_seconds 1800` writes `artifacts/task02/best.pt`
- [ ] `python predict.py --timeout_seconds 600` writes `artifacts/task02/predictions.csv`
- [ ] `python train_augmented.py --timeout_seconds 1800` writes `artifacts/task03/best.pt`
- [ ] `python predict_augmented.py --timeout_seconds 600` writes `artifacts/task03/predictions.csv`
- [ ] Final `AMLS_Exercise_<student_ID>.zip` is <= 20 MB and **excludes** the
      built image, `solution/data/`, `solution/artifacts/`.
