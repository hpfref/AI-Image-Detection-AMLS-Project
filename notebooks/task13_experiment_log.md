# Task 1.3 Data Augmentation and Robustness, Experiment Log

Running log for Task 1.3 development. Findings, run versions, and decisions land here as
work happens, so the report can cite the evidence later.

## Setup

**Task:** make the Task 1.2 detector robust to realistic distortions (scale, JPEG
compression, blur, noise). Same constraints as Task 1.2: CPU-only, same time budget,
hard `fpr_real <= 0.20`.
**Targets:** `recall_ai >= 0.60` on `data/validation_augmented/` with `fpr_real <= 0.20`;
keep `data/validation/` competitive.
**Init:** fine-tune from `artifacts/task02/best.pt` (PDF: "continuing from the Task 2
starting point").
**Time budget:** 5x Appendix-C reference = 5 * 155.616s, about 778s (same machine as Task 1.2).
**Calibration:** primary on `data/calibration_augmented/`; clean `data/calibration/`
reported for contrast.
**Augmentation is applied on the fly during training**, never stored as a new split.

**Starting point, Task 2 model on shifted data (run 23, from task12 log):**
| Model | val recall | val fpr | va recall | va fpr |
|-------|-----------|---------|-----------|--------|
| CNN | 0.724 | 0.197 | 0.558 | 0.225 |
| RF | 0.669 | 0.128 | 0.501 | 0.203 |
| ENS | 0.811 | 0.170 | 0.604 | 0.289 |

The ENS already reaches about 0.6 recall on `va` but `fpr=0.289` violates the constraint.
Core problem: augmented **real** images score like AI, too many false positives.

---

## Findings: what the provided `*_augmented` shift is (notebook `04_task13_aug_analysis.ipynb`)

Evidence is observational and from **unpaired** splits; we identify the transform family,
its per-image application, and rough magnitude, not the exact generator.

**Encoded form (raw bytes, before cleaning):** unchanged 320x320, square, RGB, JPEG in
both clean and augmented. Encoded byte size roughly halves: calibration 30.5 to 17.5 KB,
validation 30.8 to 18.2 KB. So re-compression, no geometric/format/grayscale change.

**Pairing:** not row-aligned (per-index label match 0.719, about chance for an 83% AI
split; thumbnail nearest-neighbor diagonal 0.000). Nearest-neighbor matching on contrast-
normalized thumbnails finds a confident clean original for only a minority of augmented
reals (about 60 to 95 of 321 at NN dist < 20), and those are the near-unchanged ones, so
the augmented split mostly contains **different photos**. Exact same-image clean vs
augmented pairs are not recoverable for the visibly-degraded cases.

**Pixel metric battery (cleaned 224, reals; calibration):**
| metric | clean | augmented | direction |
|--------|-------|-----------|-----------|
| lapvar (sharpness) | 0.0183 | 0.0106 | down (blur) |
| hf_ratio (HF power) | 0.0056 | 0.0037 | down (blur/compression) |
| noise residual | 0.0360 | 0.0236 | down (smoothing, NOT added noise) |
| saturation | 0.283 | 0.220 | down ~20% (chroma subsampling) |
| contrast | 0.235 | 0.212 | down ~10% |
| brightness | 0.450 | 0.442 | about unchanged (no color cast) |

Validation reals show the same pattern. Spectral power shifts from high to low bands (low-pass).

**Direct JPEG quality (quantization tables):** augmented JPEGs have a heavy low-quality
tail (estimated Q p10 about 14 to 22, min about 1) absent in clean; median stays high, so
only a *subset* is strongly compressed.

**Per-image randomness:** about 43% of augmented reals stay near clean (hf_ratio >= clean
p25), the rest degraded with varying strength, so augmentation is applied **per image with
a probability and random strength**, not uniformly.

**Forward-matching:** no single transform reproduces all metrics; the augmented point sits
between moderate JPEG (byte size matches about q70 at native 320px) and mild blur/downsample
(needed for the noise-residual drop). Operation order matters (JPEG at 320 then downsample
is not the same as JPEG at 224), so the 224-space match overstates JPEG strength.

**Ruled in:** per-image, randomly-applied low-pass / quality-reduction shift (JPEG
re-compression plus mild blur and/or downsample).
**Ruled out / no evidence:** geometric resize/crop/aspect change; format or grayscale
change; brightness/color-cast shift; additive noise (residual decreased).
**Cannot determine (unpaired):** horizontal flip / 90-deg rotation (metric-invariant);
exact JPEG quality / blur sigma / downsample factor and their order.

**Figures (in `report/figures/`):** `provided_aug_examples_real.png` and
`provided_aug_examples_ai.png` (clean vs most-degraded provided), `distributions_real.png`
(hf and noise histograms).

---

## Planned augmentation for notebook 05 (built and validated there)

From the analysis, the candidate training augmentation (per image, random strength):
random JPEG re-encode (quality mostly [45,90], occasional heavy [12,40]), Gaussian blur
(sigma [0.3,0.9]), downscale-upscale (factor [0.7,0.92]), horizontal flip, mild
brightness/contrast jitter. Omit additive noise, geometric crops, hue/heavy color jitter.

Prototype self-validation (from analysis prototyping, to be reproduced in 05): applying
this per-image to clean reals gives hf median about 0.0019 to 0.0022 vs provided 0.0019,
near-clean fraction about 0.43 to 0.45 vs provided 0.43, with noise and saturation in
range. Matched config region: p_down about 0.25 to 0.3, p_blur about 0.3 to 0.4, p_jpeg
about 0.55 to 0.6. The B.6 ablation tunes final strengths.

---

## Run Table

| Run | Init | Augmentations | LR | px | budget | CNN val r/fpr | CNN va r/fpr | model va r/fpr | Result |
|-----|------|---------------|----|----|--------|--------------|--------------|----------------|--------|
| _   | _    | _             | _  | _  | _      | _            | _            | _              | _      |

---

## Decisions

- **Init: fine-tune vs scratch:** fine-tune from `artifacts/task02/best.pt` (per plan / PDF). Scratch to be spot-checked in B.5.
- **Augmentation set:** see "Planned augmentation" above; confirmed by the B.6 ablation.
- **Mimic the exact provided shift vs general robustness:** do NOT precision-fit the exact
  JPEG quality / blur sigma / downsample factor of the provided split. Reasons: (a) the PDF
  frames this as general robustness ("scaled, compressed, blurred, or subjected to other
  transformations") and rewards improving "beyond this level"; (b) precision-fitting risks
  overfitting to the one known shift and generalizing worse to the hidden holdout. Instead
  we cover the observed family (JPEG / blur / downscale) with somewhat broader randomized
  ranges plus label-preserving extras (horizontal flip), so we are robust to the known shift
  with margin. The analysis self-validation (notebook 05) confirms our ranges still reproduce
  the provided shift, so we are not drifting far from it either.
- **No tuning on `validation_augmented`:** augmentation strengths and model selection use an
  augmented holdout derived from train (B.3) and the threshold uses `calibration_augmented`
  (B.8). `validation_augmented` is touched once, for final reporting only, to avoid
  overfitting the reported robustness number.
- **Composition: CNN-only vs ensemble:** to fill from B.7.
- **Calibration set:** to fill from B.8.
