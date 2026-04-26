# AMLS 2026 Exercise — AI Image Detection — Report

Up to 8 pages, 10pt. Compile this to `report.pdf` for submission.

## Team

- TODO: name 1
- TODO: name 2
- TODO: name 3

Quality expectations scale with team size (PDF §Grading). Pass = >= 50/100;
>= 90 awards 5 extra exam points.

---

## §1.1 Dataset Exploration & Cleaning (15 pts)

What this section must cover (PDF §1.1):

- [ ] Class distribution across the six original classes (0..5) and after
      collapse to the binary task (0 vs. 1..5).
- [ ] Image-size distribution (width, height, aspect ratio, channels).
- [ ] Basic descriptive statistics (e.g., mean intensity, file-size in bytes,
      JPEG vs. PNG ratio, EXIF presence) per class.
- [ ] **Class-revealing characteristics** — anything in the metadata or pixel
      stats that leaks the label (this is critical to call out).
- [ ] Description and **justification** of the deterministic cleaning pipeline
      implemented in [solution/clean.py](../solution/clean.py). No augmentation here.
- [ ] At least one figure (histograms / class examples).

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
