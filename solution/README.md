# `solution/` — Graded Pipeline

This is the folder the grader builds and executes. Everything required to
build and run the ML pipeline must live here.

## Pipeline contract (PDF §Submission Guidelines)

The grader runs the six scripts in this exact order, each with the given
timeout. Each script is killed at its timeout, so write best checkpoints to
`artifacts/` regularly.

```
1. python clean.py             --timeout_seconds 600
2. python prepare.py           --timeout_seconds 600
3. python train.py             --timeout_seconds 1800
4. python predict.py           --timeout_seconds 600
5. python train_augmented.py   --timeout_seconds 1800
6. python predict_augmented.py --timeout_seconds 600
```

## Path conventions

| Path | Permission | Notes |
|------|------------|-------|
| `data/train/`, `data/calibration/`, `data/calibration_augmented/`, `data/validation/`, `data/validation_augmented/` | **read-only** | parquet files, columns `image: binary`, `source_class: int8` |
| `data/predict/` | **read-only** | parquet files, columns `row_id: int32`, `image: binary` (no labels) |
| `artifacts/` | read-write | only place we are allowed to write |

**Never** write under `data/`. **Never** read `data/predict/` from `prepare.py`
(PDF: "prepare.py should not prepare data from solution/data/predict/ as it
may change after training").

## Required outputs

- `artifacts/task02/predictions.csv` — produced by `predict.py`
- `artifacts/task03/predictions.csv` — produced by `predict_augmented.py`

Both CSVs use the format below, with one row per image in `data/predict/`:

```csv
row_id,predicted_label
0,1
1,0
2,1
```

## Label convention

Original `source_class` column has six values: `0` real, `1`..`5` AI variants
(SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney). For this exercise, **labels 1..5
are collapsed to a single AI class `1`**. Implemented in
[_lib/data.py](_lib/data.py).

## Performance targets

| Task | Constraint | On split |
|------|------------|----------|
| 1.2  | recall_ai >= 0.8 | `data/validation/` |
| 1.2  | FPR_real <= 0.20 | `data/validation/` |
| 1.3  | recall_ai >= 0.6 | `data/validation_augmented/` |
| 1.3  | FPR_real <= 0.20 | (same constraint, robust model) |

The threshold targeting FPR <= 0.20 must be **calibrated automatically** on
`data/calibration/` (or `data/calibration_augmented/` for Task 3) and
**independently validated** on the corresponding validation split. No
manually picked thresholds.

## CPU & runtime constraints

- CPU-only — no `cuda`, no GPU wheels.
- No internet during runtime (the grader runs `--network none`).
- No pretrained models downloaded at runtime — bake everything into the image.
- Image size <= 4 GB.
- Local training time <= 5x the elapsed time printed by
  [../train_time_reference.py](../train_time_reference.py).

## Layout

```
solution/
├── Dockerfile                 Appendix A
├── requirements.txt
├── .dockerignore
├── clean.py                   Task 1.1
├── prepare.py                 Task 1.2
├── train.py                   Task 1.2 (uses Appendix B CNN)
├── predict.py                 Task 1.2
├── train_augmented.py         Task 1.3
├── predict_augmented.py       Task 1.3
└── _lib/
    ├── __init__.py
    ├── io.py                  parquet read, image decode, artifact paths
    ├── data.py                Dataset/Dataloader, label binarization (1..5 -> 1)
    ├── model.py               Appendix B CNN + custom families
    ├── calibration.py         FPR-target threshold calibration
    ├── metrics.py             recall_ai, fpr_real, confusion
    └── seed.py                deterministic seeding + thread caps
```
