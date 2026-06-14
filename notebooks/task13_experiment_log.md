# Task 1.3 - Full Experiment Log

Running log for Task 1.3 (augmentation + feature engineering) with the **new CapacityCNN**
(ResNet-SE) shipped in Task 1.2. Findings, run versions and decisions land here as work
happens, so the report can cite the evidence later. A first 1.3 attempt was made with the
old weak k=16 CNN and is archived in `notebooks/old/`; its provided-shift analysis is reused
(same data) but every model run here is redone with the new backbone.

## Setup

**Task:** make the Task 1.2 detector robust to realistic distortions (scale, JPEG
compression, blur). Same constraints as Task 1.2: CPU-only, same time budget, hard
`fpr_real <= 0.20`.
**Targets:** `recall_ai >= 0.60` on `data/validation_augmented/` (va) with `fpr_real <= 0.20`;
keep clean `data/validation/` competitive (1.2 shipped ~0.87 rec @ 0.186 fpr).
**Init:** fine-tune from `artifacts/task02/best.pt` (PDF: "continuing from the Task 2 starting
point"). The old attempt confirmed fine-tune > from-scratch within budget.
**Time budget:** 5x Appendix-C reference = ~778s for notebook development (deadline loop, same
as 1.2); the shipped `train_augmented.py` runs to the full ~1800s grader timeout.
**Calibration:** threshold picked on `data/calibration_augmented/` only; clean
`data/calibration/` reported for contrast. `validation_augmented` read once for the final
report. Selection signal is a train-derived augmented holdout - never va, never calibration.
**Augmentation is applied on the fly during training**, never stored as a new split.

**Backbone:** CapacityCNN (ResNet-SE, stem stride-4 + 4 SE-residual stages 32->256, 2.84M
params, 192px), FocalLoss(gamma=1.5), AdamW + warmup/cosine, step-based eval, holdout-AUC
selection. RF = 101-dim hand-crafted features. Ensemble `p = alpha*p_cnn + (1-alpha)*p_rf`,
1.2 ship alpha=0.50.

**Starting point - the Task 2 model on shifted data (1.2 script-mode ship, threshold from
clean `calibration` @ 0.18):**
| model | val rec | val fpr | va rec | va fpr |
|-------|---------|---------|--------|--------|
| CNN | 0.862 | 0.181 | 0.740 | 0.273 |
| RF | 0.650 | 0.117 | 0.482 | 0.193 |
| ENS | 0.871 | 0.186 | 0.736 | 0.262 |

**Does the Task 2 model already pass on va? No - but it is close.** Recall 0.736 already clears
0.60 comfortably; the blocker is `fpr 0.262 > 0.20`. Two reasons stack: (a) **calibration
mismatch** - the threshold was picked on clean `calibration`, so on the shifted va it sits too
low and too many augmented reals cross it; (b) the **CNN branch smears under distortion**
(va fpr 0.273) while the RF stays robust (0.193), so the ensemble inherits the CNN's high fpr.
This is a more favorable start than the old k=16 attempt (old ENS va 0.685/0.374): the new
model's recall is not the problem, only the false-positive rate is. Task 1.3 addresses it by
(a) recalibrating on `calibration_augmented` and (b) fine-tuning the CNN with augmentation so
augmented reals stop scoring like AI, ideally without giving back the clean-val operating point.

---

## Plan / hypotheses to try  [TEMPORARY - replaced by real Runs as numbers come in]

This section seeds the exploration from what we learned in the old k=16 attempt and from the
1.2 work. It is a scratchpad, not results; each item becomes a Run row below once measured.

**What we expect, going in:**
- The new model starts at va 0.736/0.262, so we likely need *less* aggressive augmentation than
  the old attempt - the job is mostly pulling CNN va-fpr from ~0.27 down under 0.20 while holding
  recall, not clawing recall up from below 0.60.
- Recalibration alone (Run 0) will cut fpr but also cut recall hard (old attempt: recalibrate-only
  dropped va recall to ~0.49). Fine-tuning with augmentation should recover recall at low fpr.
- The va-AUC "ceiling ~0.76" from the old log was a property of the **k=16/160px** model. The new
  2.84M-param backbone may rank augmented AI vs real better; an open question is whether it lifts
  va AUC meaningfully above 0.76 (the real prize - margin at the operating point).

**Runs we intend to do (smarter/condensed order vs the old it1-it12 grind):**
- **Run 0 - baseline "before".** 1.2 ensemble unchanged; recalibrate threshold on
  `calibration_augmented`. Isolates what recalibration alone buys (this is also the "reuse the
  1.2 RF, just recalibrate" idea). Report CNN-only, ENS (RF reused), with both clean-cal and
  cal_aug thresholds.
- **Run 1 - fine-tune + aug v1, recalibrated.** Fine-tune CapacityCNN from best.pt with broad
  augmentation (JPEG re-encode + blur + downscale, + flip), 5x budget, step-based eval + snapshot
  averaging. Ensemble with the frozen 1.2 RF. Expect CNN va-fpr to fall sharply.
- **Run 2 - augmentation ablation.** Drop one family at a time (no_jpeg / no_blur / no_down /
  jpeg_only) to see which transforms drive robustness at this capacity (may differ from k=16,
  where all three helped and sharpen/noise/desat hurt).
- **Run 3 - aug realism / strength match.** Use the grid-wash JPEG (compress at 1.15-1.45x canvas
  then resize back, no fake 8x8 grid at 224) and tune strength so our augmented metrics sit near
  va (block ratio ~1.0, saturation ~0.18, sharpness/hf near va). Figures: aug-strength strip,
  ours-vs-provided population match. Note from old attempt: a *correct* match did not raise k=16
  AUC - re-test whether it helps the stronger model.
- **Run F - feature engineering / composition.** On the augmented holdout compare CNN_aug only vs
  CNN_aug + frozen 1.2 RF (RF reused) vs CNN_aug + RF refit on augmented features + the +8-dim
  color block (cross-channel corr, saturation mean/std, RGB balance; +0.013 RF AUC in the old
  attempt). Pick composition + alpha by holdout AUC.
- **Run C - calibration sensitivity.** `calibration_augmented` has only ~321 reals (~+/-0.04
  binomial error at the fpr target), so the cal->va transfer is noisy. Compare fpr target 0.18 vs
  0.17 and pick a safe margin. va error-analysis battery figure (which va images we still miss).
- **Run S - final pick + clean-val regression check.** Confirm the chosen model holds clean
  `validation` (PDF wants both), Task2-vs-Task3 comparison on both splits, budget proof.

**Reused priors (do not rediscover):** the provided shift is per-image random JPEG re-compress
(heavy low-Q tail) + mild blur/downscale + ~20% desaturation, bimodal (~43% near-clean); families
that help = JPEG/blur/downscale; sharpen neutral; additive noise / crops / heavy color jitter hurt.

---

## Provided-shift findings (notebook `07_task13_aug_analysis.ipynb`)

Ported from the old `04_task13_aug_analysis` and re-confirmed on the identical `*_augmented`
splits. Evidence is observational and from **unpaired** splits; we identify the transform family,
its per-image application and rough magnitude, not the exact generator.

**Encoded form (raw bytes, before cleaning):** unchanged 320x320, square, RGB, JPEG in both clean
and augmented. Encoded byte size roughly halves (calibration 30.5 -> 17.5 KB, validation 30.8 ->
18.2 KB). So re-compression, no geometric/format/grayscale change.

**Pairing:** not row-aligned (per-index label match ~0.72, about chance; thumbnail NN diagonal
0.000). The augmented split mostly contains **different photos**, so same-image clean-vs-augmented
pairs are not recoverable for the degraded cases.

**Pixel metric battery (cleaned 224, reals; calibration):**
| metric | clean | augmented | direction |
|--------|-------|-----------|-----------|
| lapvar (sharpness) | 0.0183 | 0.0106 | down (blur) |
| hf_ratio (HF power) | 0.0056 | 0.0037 | down (blur/compression) |
| noise residual | 0.0360 | 0.0236 | down (smoothing, NOT added noise) |
| saturation | 0.283 | 0.220 | down ~20% (chroma subsampling) |
| contrast | 0.235 | 0.212 | down ~10% |
| brightness | 0.450 | 0.442 | about unchanged (no color cast) |

**Direct JPEG quality (quantization tables):** augmented JPEGs have a heavy low-quality tail
(estimated Q p10 ~14-22, min ~1) absent in clean; median stays high, so only a *subset* is
strongly compressed.
**Per-image randomness:** ~43% of augmented reals stay near clean, the rest degraded with varying
strength - augmentation is applied **per image with a probability and random strength**.
**Ruled in:** per-image, randomly-applied low-pass / quality-reduction shift (JPEG re-compression
plus mild blur and/or downsample).
**Ruled out / no evidence:** geometric resize/crop/aspect change; format or grayscale change;
brightness/color-cast shift; additive noise (residual decreased).
**Cannot determine (unpaired):** horizontal flip / 90-deg rotation; exact JPEG quality / blur sigma
/ downsample factor and order.

**Figures (regenerated, `report/figures/`):** `provided_aug_examples_real.png`,
`provided_aug_examples_ai.png` (clean vs most-degraded provided), `distributions_real.png`
(hf and noise histograms).

---

## Augmentation pipeline (notebook `08_task13_augmentation.ipynb`)

Our on-the-fly training augmentation (per image, random strength), built and validated in nb08.
Filled in as the versions stabilize.

- **v1 (broad):** [to fill] random JPEG re-encode, Gaussian blur, downscale-upscale, horizontal
  flip. Quality/sigma/factor ranges and per-family probabilities recorded here once run.
- **match (grid-wash):** [to fill] JPEG at 1.15-1.45x canvas then bicubic resize back, lighter
  blur/downscale, optional desaturation - tuned so our augmented battery matches va.
- **cost:** [to fill] per-batch augmentation overhead at 192px.

---

## Full Run Table (results history)

Notebook fine-tune budget 778s (5x ref). Threshold calibrated on `calibration_augmented`
(target fpr recorded per run). va = validation_augmented. "RF reused" = the frozen 1.2 RF;
"RF_aug" = RF refit on augmented features; "+color" = the 8-dim color block added.

| run | composition | aug | va rec | va fpr | val rec | val fpr | va AUC | notes |
|-----|-------------|-----|--------|--------|---------|---------|--------|-------|
| Task 1.2 baseline (clean-cal thr) | CNN+RF ENS | none | 0.736 | 0.262 | 0.871 | 0.186 | - | the "before" |
| _Run 0+ to be filled from nb08_ | | | | | | | | |

---

## Key Decisions (shipped)

[Filled at the end - init (fine-tune), final augmentation version, composition + alpha,
calibration target, and the honest trade-offs, in the style of the 1.2 log.]

---

## What remains / ports

After the notebook runs are confirmed (project workflow: validate in notebooks first):
- `solution/_lib/model.py` - ensure CapacityCNN is importable (may already be ported in 1.2).
- `solution/_lib/features.py` - add the on-the-fly augmentation helper and (if Run F adopts it)
  the +8-dim color feature block.
- `solution/train_augmented.py` - implement fine-tune-from-task02 + aug + calibrate at full
  ~1800s; writes `artifacts/task03/best.pt`.
- `solution/predict_augmented.py` - writes `artifacts/task03/predictions.csv`.
- `report/report.md` §1.3 - augmentation rationale, scratch-vs-fine-tune, Task2-vs-Task3 on both
  validation splits, target check.
