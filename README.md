# AMLS 2026 Exercise — AI Image Detection

Binary classification pipeline that decides whether an image is **real (0)** or
**ai_generated (1)**. Original `source_class` labels 1..5 (SD 2.1, SDXL, SD 3,
DALL-E 3, Midjourney) are collapsed into the single class `1`.

See [AMLS_2026_Exercise.pdf](AMLS_2026_Exercise.pdf) for the full spec.

## Team

- TODO: list team members here (1-3 people, names go in `report/report.md` too).

## Repository layout

```
report/                       8-page PDF report source (10pt)
solution/                     graded submission folder — Dockerized ML pipeline
  Dockerfile                  Appendix A
  clean.py                    Task 1.1 — exploration + deterministic cleaning
  prepare.py                  Task 1.2 — feature/tensor caches (NOT data/predict/)
  train.py                    Task 1.2 — train + calibrate, target FPR <= 20%
  predict.py                  Task 1.2 — writes artifacts/task02/predictions.csv
  train_augmented.py          Task 1.3 — robustness via augmentation
  predict_augmented.py        Task 1.3 — writes artifacts/task03/predictions.csv
  _lib/                       shared helpers (io, data, model, calibration, metrics, seed)
task04_explainability/        Task 1.4 — saliency / occlusion / FP-FN analysis
train_time_reference.py       Appendix C — local 5x CPU-budget reference
```

`solution/data/` is mounted read-only by the grader and is **never committed**.
`solution/artifacts/` is created at runtime and is **never committed**.

## Task map (PDF -> file)

| Task | Points | Where |
|------|--------|-------|
| 1.1 Dataset exploration & cleaning | 15 | [solution/clean.py](solution/clean.py), report §1.1 |
| 1.2 Modeling & tuning under time constraints | 35 | [solution/prepare.py](solution/prepare.py), [solution/train.py](solution/train.py), [solution/predict.py](solution/predict.py), report §1.2 |
| 1.3 Data augmentation & feature engineering | 30 | [solution/train_augmented.py](solution/train_augmented.py), [solution/predict_augmented.py](solution/predict_augmented.py), report §1.3 |
| 1.4 Explainability | 20 | [task04_explainability/](task04_explainability/), report §1.4 |

Pass threshold is 50/100; >= 90 awards 5 extra exam points.

## Build & run (matches grader invocation)

```bash
# Build the image (CPU-only, no internet at runtime; final image must be <= 4 GB)
docker build -t amls-solution solution/

# Mount data/ read-only and artifacts/ writable, then run the six scripts in order
docker run --rm --cpus 8 --network none \
  -v "$PWD/solution/data:/workspace/solution/data:ro" \
  -v "$PWD/solution/artifacts:/workspace/solution/artifacts" \
  amls-solution python clean.py             --timeout_seconds 600
docker run --rm --cpus 8 --network none ... amls-solution python prepare.py           --timeout_seconds 600
docker run --rm --cpus 8 --network none ... amls-solution python train.py             --timeout_seconds 1800
docker run --rm --cpus 8 --network none ... amls-solution python predict.py           --timeout_seconds 600
docker run --rm --cpus 8 --network none ... amls-solution python train_augmented.py   --timeout_seconds 1800
docker run --rm --cpus 8 --network none ... amls-solution python predict_augmented.py --timeout_seconds 600
```

Each script is killed at its timeout, so write best checkpoints to
`solution/artifacts/` regularly.

## Local CPU budget

Run [train_time_reference.py](train_time_reference.py) once on your laptop —
that's your reference. Local training must finish within **5x** that
reference (PDF §1.2). The grader runs with `--cpus 8`.

## Submission

Final deliverable: `AMLS_Exercise_<student_ID>.zip` (max 20 MB) containing
`report.pdf` + the `solution/` folder + any Task 1.4 code. Do **not** include
the built Docker image, `solution/data/`, or `solution/artifacts/`.
