# AMLS 2026 - AI Image Detection

TU Berlin exercise. Binary classification: 0=real, 1=ai_generated. Labels 1-5 all collapse to 1.
Deadline: Jul 15 2026.

## Dataset

- 6 source classes: 0=real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=Midjourney
- Splits: train (~29,700 rows), calibration (~1,924), validation (~1,124), predict (100)
- All images: JPEG, RGB
- Location: `solution/data/` (read-only at Docker runtime)

## Key findings (Task 1.1)

- Dimension leak: AI images are always square (SD/MJ = 320x320, DALL-E3 = 270x270), real images are non-square and variable. Aspect ratio alone is a near-perfect classifier.
- File size leak: real ~50 KB median, AI much smaller (DALL-E3 ~18 KB, others ~24-32 KB).
- Clean transform: shorter-edge resize + center-crop to 224x224 neutralises the dimension leak.

## Submission requirements

- CPU only, no internet at Docker runtime, max 4 GB image, max 20 MB zip
- Script execution order and timeouts:
  1. `python clean.py --timeout_seconds 600`
  2. `python prepare.py --timeout_seconds 600`
  3. `python train.py --timeout_seconds 1800`
  4. `python predict.py --timeout_seconds 600`
  5. `python train_augmented.py --timeout_seconds 1800`
  6. `python predict_augmented.py --timeout_seconds 600`
- All outputs go to `solution/artifacts/` (never write to `solution/data/`)
- Required outputs: `artifacts/task02/predictions.csv`, `artifacts/task03/predictions.csv`
- Goal: maximise recall_ai with FPR on real <= 20%, calibrated via `data/calibration/`

## Workflow

Develop and validate in `notebooks/` first, then port to `solution/` scripts. Do not write to scripts until the notebook code is verified. Each notebook's final cell lists explicit port targets.

## Project structure

```
notebooks/              development notebooks (not submitted)
solution/
  _lib/                 shared helpers: io.py, data.py, model.py, seed.py
  clean.py              Task 1.1 -> artifacts/clean/
  prepare.py            Task 1.2 -> artifacts/prepared/
  train.py              Task 1.2 -> artifacts/task02/
  predict.py            Task 1.2 -> artifacts/task02/predictions.csv
  train_augmented.py    Task 1.3 -> artifacts/task03/
  predict_augmented.py  Task 1.3 -> artifacts/task03/predictions.csv
  data/                 read-only at runtime
  artifacts/            all runtime outputs
```

## Style notes

- No emojis, no em-dashes (use commas or " -> " instead)
- No GPU code, no pretrained model downloads at runtime
- Add `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before numpy import in every notebook (Windows dual-OpenMP fix)
- IMG_SIZE = 224 everywhere
