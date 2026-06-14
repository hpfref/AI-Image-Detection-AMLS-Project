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

All exploration and cleaning in this section is implemented in `solution/clean.py`, which validates every training row and writes `artifacts/clean/train_manifest.csv`, the per-row validity list that `prepare.py` then consumes.

**Resize shorter-edge and center-crop to 224x224.** Image dimensions are a near-perfect class signal. Mapping all images to a fixed square canvas via aspect-preserving resize followed by center-crop removes both the square/non-square distinction and the 270 vs 320 px difference between generators. Stretching was rejected because it encodes the original aspect ratio in pixel coordinates. 224x224 matches the reference CNN in Appendix B.

**Convert to RGB.** Defensive: all images are already RGB, but this ensures consistent 3-channel input for any downstream model.

**Drop unreadable rows.** A 500-image smoke test found zero corrupt rows. The guard quantifies the drop rate on the full dataset and prevents training failures.

## §1.2 Modeling and Tuning under Time Constraints (35 pts)

### Setup and protocol

The task is binary classification of real (0) against ai_generated (1). The graded objective is to maximise recall on the AI class while holding the false-positive rate on real images at or below 0.20, and that constraint is verified on a held-out validation split rather than on anything used to fit or tune the model.

Four splits are provided: `train` (around 29,700 images), `calibration` (1,924), `validation` (1,124) and an unlabeled `predict` set (100). Each has a fixed role, and a fifth split is created internally so each split is used for only one thing. A stratified 90/10 split of `train` produces a fit fold and a held-out fold (in `prepare.py`, fraction 0.10, seed 0, stratified by source class so all six generators are represented). This internal holdout is not one of the provided splits; it exists because unbiased checkpoint and model selection needs a labelled set that is neither the calibration set (reserved for setting the threshold) nor the validation set (reserved as the final generalisation check and proxy for the hidden test). It is also the largest labelled set available for this, about 2,970 images, so AUC measured on it is far less noisy than on the roughly 190 real images in calibration or validation. In short: `train` fits the models, the internal holdout drives early stopping and selection, `calibration` sets the threshold, and `validation` is read once at the end to confirm the operating point.

All experiments are developed and measured in development notebooks, and the submitted scripts mirror that validated pipeline. `prepare.py` decodes the cleaned images into a uint8 memmap cache, computes per-channel normalisation statistics on the fit fold, and extracts the 101-dimensional engineered feature set; `train.py` fits the classical model, trains the CNN, sweeps the ensemble weight, calibrates the threshold and saves the artifacts; `predict.py` loads the saved ensemble and writes `artifacts/task02/predictions.csv` in the required `row_id,predicted_label` format.

Local development is held to at most five times the Appendix C reference training time (`train_time_reference.py`), measured once on the development machine. The reference is used because it ties the budget to a fixed amount of computation rather than to wall-clock time on one machine, so the reported figures are not skewed by hardware. Every metric in this section is measured at that 5x-reference budget; how the submitted `train.py` behaves under the grader's own timeout is covered in the budget decision below.

### Two model families

Two model families are trained from scratch and compared, with no pretrained weights, consistent with the no-internet runtime constraint.

The classical baseline is a Random Forest over a 101-dimensional engineered feature vector built from the Task 1.1 cues and several more: per-channel colour mean, standard deviation and 8-bin histograms, Laplacian variance and Sobel edge energy, an FFT high-frequency ratio with four radial spectral-band ratios, a 3x3 grid of patch means and standard deviations, per-channel noise and Cb/Cr chroma-noise residuals, channel skewness, and a JPEG block-boundary discontinuity score. Both a regularised Logistic Regression and a Random Forest are trained every run and the stronger is kept by holdout recall; the forest wins consistently. On run 22, for instance, the best logistic model reached 0.749 holdout recall at 0.857 AUC against the forest's 0.781 recall at 0.885 AUC. The forest is preferred because the feature interactions are non-linear (for example the JPEG-block score interacts with the generator source, and the spectral bands interact with colour skewness), which trees capture through splits while a linear model cannot without manual crosses. The final forest is 400 trees with balanced class weights and square-root feature subsampling.

The neural family is a convolutional network trained from scratch on the cleaned 224x224 images, resampled to the network input size. Its architecture and training are the subject of the decisions below.

### Key design decisions

**Architecture and resolution.** The decisive change was replacing the small reference-style CNN with a ResNet-style network. Looking at the old network showed why earlier tuning had stopped improving: it stopped downsampling after two pooling layers, so its deepest convolutions ran at high spatial resolution and most of its compute sat in a single wide layer; it was heavy on FLOPs but poor in parameters (around 250k), spending the budget on resolution it could not make use of. The redesign downsamples to stride-4 immediately with a strided stem and pooling, then runs four residual stages of widths 32, 64, 128 and 256 with two residual blocks each and squeeze-and-excitation channel attention, ending in global average pooling, dropout and a linear head. This gives about 2.84M parameters, roughly eleven times the old network, at about 0.75 times the compute per image, so the budget goes to capacity rather than image resolution, and the residual connections now serve their real purpose in a network 17 convolutions deep. The input is 192px, chosen because the model is capacity-limited rather than resolution-limited here, so using a smaller image to fit more training steps is worth it.

![Capacity redesign: more parameters at less compute, and the resulting gradient steps that fit in budget](figures/fig7_ablation.png)

**Loss and optimiser.** The network is trained with focal loss (gamma 1.5) and class weights to handle the roughly 5:1 AI-to-real imbalance, optimised with AdamW (peak learning rate 1e-3, weight decay 1e-4) under a short warmup followed by a cosine decay to near zero. The schedule length is self-calibrated to the measured training throughput so the decay always lands at the time budget regardless of machine speed. Batches use channels-last memory format, which is faster for these convolutions on CPU. The training trace below shows the holdout AUC and the recall at the target false-positive rate climbing across steps as the learning rate decays, with the best checkpoint landing near the final step.

![Training trace: holdout AUC and recall climbing under the warmup plus cosine learning-rate schedule](figures/fig4_training_trace.png)

**Model selection on the internal holdout.** Checkpoints, and the run to submit, are selected by holdout AUC, not by recall at the calibrated threshold. Recall at a fixed threshold is a single noisy point on the ROC curve and was exactly what made identical configurations flip between pass and fail on seed and timing; AUC summarises the whole curve and is reproducible and threshold-independent. Because the model keeps improving to the last step, the final weights are always scored once after the deadline, so a slightly stale checkpoint is never submitted. Selection happens only on the internal holdout, never on calibration or validation, which keeps those splits clean for their own roles. The same metric decides which run is submitted, and the run tables make two of these choices look odd, so both are explained here. First, the final run is deliberately not the one with the highest validation recall: run 25 reached the highest recall of any run (0.879) but at an illegal false-positive rate (0.207), which is exactly why recall at a fixed threshold is treated as a noisy operating point rather than a selection target. Second, the final run is not even the highest in-budget holdout AUC: run 27 reached 0.933 against the final run's 0.929 and was still rejected, because a single run's holdout AUC carries its own noise, so a gain only counts when it is corroborated, showing in the CNN-alone holdout and ideally on validation and beating the stable baseline by a small margin. Run 27 did none of these: the CNN alone was flat (0.916 to 0.917), the validation AUC did not move (0.901 against 0.902), and the ensemble holdout had sat at 0.928 to 0.929 across the three preceding runs, so 0.933 looks like noise from a single run, at the cost of 42% more parameters and a failed operating point. The same reasoning answers why an earlier higher-recall run is not preferred: run 24 reached 0.849 validation recall, but it shares the final run's architecture and holdout AUC (0.929), so it is the same ROC; its higher recall is only the looser 0.19 calibration target it happened to use, and because the submitted pipeline calibrates at 0.18 regardless, those weights would read essentially the same 0.837. The rule throughout is to trust improvements that repeat and show up in several places, like the within-run forest lift described next, and to be sceptical of one-off jumps.

**Ensemble against CNN-only.** The CNN is combined with the Random Forest by averaging their probabilities, p_ens = 0.5 p_cnn + 0.5 p_rf, with the weight chosen by a holdout AUC sweep. The two families look at different evidence, the CNN at pixel and texture artifacts, the forest at colour, spectral and JPEG-block statistics, so their errors are only partly correlated and averaging raises the ROC above either alone. While the CNN was weak this lift was large (around 0.04 AUC); now that the CNN is strong it is smaller but still real, a steady within-run gain of about 0.01 holdout AUC, and the forest contributes a high-precision low-false-positive branch (validation FPR 0.117) that pulls the operating point to a safer place. Because the forest also uses part of the shared time budget, the two were compared at equal budget in a dedicated pair of runs: a full-budget CNN with no forest (run 28, holdout AUC 0.927) against the ensemble where the CNN trains slightly less to leave time for the forest (run 29, holdout AUC 0.929). They are close, and the ensemble is chosen for its steady within-run lift and safer false-positive margin. The weight ending up at 0.50, up from 0.40 in the weak-CNN era, reflects the CNN and forest now being about equally informative.

**Calibration protocol.** The decision threshold is set by searching, over the real images of `calibration`, for the threshold whose false-positive rate matches a target, then verified on the held-out validation split. The target is 0.18 rather than 0.20: the realised validation FPR sits a little above the calibration target and varies from run to run because the real subset is small, so a 0.18 target leaves room for that sampling gap, while a higher target would cost recall, the graded quantity. The residual gap between calibration, validation and the hidden predict split is acknowledged as an accepted risk. Calibrating on one split and selecting the model on a separate holdout, both distinct from validation, is the main guard against overfitting the operating point, which the task warns is scored on a hidden holdout.

**Budget handling and timeout safety.** The 5x reference is the basis for all reported metrics because it is hardware-independent. The submitted `train.py`, by contrast, trains to whatever per-script timeout the grader provides (1800s on 8 CPUs): it fits and saves the Random Forest first, then trains the CNN up to a fixed reserve before the deadline, the reserve covering the final evaluation, calibration and artifact saving. Training is deliberately not left to run until the grader kills the process, even though that would use a few more seconds: if the process is stopped while a file is being written the saved model could be corrupted, a complete ensemble would have to be re-calibrated and re-saved on every new best (each one another chance to corrupt a file), and by the deadline the cosine learning rate has decayed to near zero so the final seconds barely move the weights. The best checkpoint is still written incrementally during training as a cheap safeguard, but the complete artifact set is produced once, inside the reserve.

**Capability beyond the reference budget.** The 5x figure is a development constraint, not a ceiling of the model. Running the same recipe through the solution scripts to the full grader timeout, on the same hardware, lifts holdout AUC from 0.929 to 0.951. The model is therefore budget-limited rather than saturated; given more compute it keeps improving, with diminishing returns. These numbers are not used as the headline result because they are hardware-dependent and not comparable across machines, but they show the model could still improve with more time.

### Experiment trajectory

Development started from the small reference-style CNN of Appendix B and proceeded as a long sequence of incremental changes: replacing cross-entropy with focal loss, sweeping the focal gamma and input resolution, adding the Random Forest ensemble, and calibrating the threshold from data. That line did produce passing models (runs 7, 12, 22 and 23 all met the constraint), but only borderline and seed-dependent ones; reruns of the same recipe flipped between pass and fail on random seed and threshold noise. The reason shows in the table below: through every change the ensemble AUC barely moved, staying around 0.88 to 0.91. Since recall at a fixed false-positive ceiling is bounded by the AUC, tuning the loss, threshold or resolution only slid the operating point along the same curve without raising it. Several capacity-style changes were tried in this era and dropped: focal gamma 2.0 (higher single-run AUC but recall swung across seeds, runs 13 to 17), one or two residual skip connections added to the shallow network (run 16) and depthwise-separable convolutions (run 17, worse), upweighting the hard SD3 class (pushed FPR past 0.20, runs 8 to 11), and simply enlarging the input to 224px (run 19) or the width to k=32 (run 20), both of which ran too few steps in budget and underfit.

| Run | CNN arch | img px | gamma | ENS target | ENS val recall | ENS val FPR | ENS val AUC | Result |
|-----|----------|--------|-------|------------|----------------|-------------|-------------|--------|
| 1 | 3-conv | 96 | CE | - | - | - | - | no ensemble |
| 2 | 3-conv | 128 | focal | - | - | - | - | no ensemble |
| 3 | 4-conv | 128 | focal | - | - | - | - | no ensemble |
| 4 | 4-conv | 128 | focal | 0.15 | 0.721 | 0.149 | - | FAIL |
| 5 | 4-conv | 128 | focal | 0.19 | 0.778 | 0.207 | - | FAIL |
| 6 | 4-conv | 128 | focal | 0.20 | 0.799 | 0.223 | - | FAIL |
| 7 | 5-conv+MLP | 128 | 1.5 | 0.19 | 0.809 | 0.197 | 0.884 | PASS |
| 8 | 5-conv+MLP | 128 | 1.5† | 0.19 | 0.833 | 0.234 | - | FAIL |
| 9 | 4-conv | 160 | 1.5† | 0.18 | 0.792 | 0.165 | - | FAIL |
| 10 | 5-conv | 128 | 1.5† | 0.18 | 0.775 | 0.176 | - | FAIL |
| 11 | 5-conv | 128 | 1.5† | 0.19 | 0.817 | 0.250 | - | FAIL |
| 12 | 5-conv+MLP | 128 | 1.5 | 0.19 | 0.803 | 0.191 | 0.891 | PASS |
| 13 | 5-conv+MLP | 128 | 2.0 | 0.19 | 0.827 | 0.207 | 0.896 | FAIL |
| 14 | 5-conv+MLP | 128 | 2.0 | 0.18 | 0.786 | 0.160 | - | FAIL |
| 15 | 5-conv+MLP | 128 | 2.0 | 0.185 | 0.771 | 0.165 | 0.885 | FAIL |
| 16 | ResBlock | 128 | 2.0 | 0.185 | 0.768 | 0.144 | 0.886 | FAIL |
| 17 | DW-sep | 128 | 2.0 | 0.185 | 0.740 | 0.149 | 0.875 | FAIL |
| 18 | 5-conv+MLP | 128 | 1.5 | 0.18 | 0.782 | 0.160 | 0.896 | FAIL |
| 19 | 5-conv+MLP | 224 | 1.5 | 0.18 | 0.772 | 0.144 | 0.878 | FAIL |
| 20 | 5-conv+MLP | 128, k=32 | 1.5 | 0.18 | 0.722 | 0.138 | 0.876 | FAIL |
| 21 | 5-conv+MLP | 160 | 1.5 | 0.18 | 0.795 | 0.176 | 0.878 | FAIL |
| 22 | 5-conv+MLP | 160 | 1.5 | 0.19 | 0.809 | 0.197 | 0.878 | PASS |
| 23 | 5-conv+MLP | 160 | 1.5 | 0.19 | 0.811 | 0.170 | 0.888 | PASS |

*Runs 1 to 3 predate the ensemble (CNN only). † marks runs that additionally upweighted the SD3 class. The ensemble AUC column shows the plateau around 0.88 to 0.91 that motivated the redesign.*

The conclusion was that the limit was the network's discriminative ceiling, not its operating point, which motivated the architecture rework described above. The capacity era (runs 24 onward) then refined it: run 24 was the first pass of the redesign and immediately at a much stronger operating point; run 25 spent more of the budget on the CNN and showed the 0.19 calibration target left no room (validation FPR 0.207); run 26 moved to step-based AUC selection and a 0.18 target for a clean pass; run 27 added a block to the deepest stage and, despite posting the highest in-budget holdout AUC, was rejected as likely noise from a single run, at the cost of 42% more parameters (see model selection above); and runs 28 and 29 ran the equal-budget CNN-only against ensemble comparison that decided which model to submit.

| Run | Change from previous | CNN holdout AUC | ENS holdout AUC | val recall | val FPR | val AUC | Result |
|-----|----------------------|-----------------|-----------------|------------|---------|---------|--------|
| 24 | ResNet-SE redesign, 192px, warmup plus cosine, calibrate 0.19 | 0.915 | 0.929 | 0.849 | 0.186 | 0.903 | PASS |
| 25 | larger CNN share of the budget | 0.919 | 0.929 | 0.879 | 0.207 | 0.914 | FAIL (FPR) |
| 26 | step-based AUC selection, calibrate 0.18 | 0.916 | 0.928 | 0.837 | 0.191 | 0.902 | PASS |
| 27 | extra block in the deepest stage (+42% params) | 0.917 | 0.933 | 0.842 | 0.213 | 0.901 | rejected |
| 28 | full-budget CNN only, no forest | 0.927 | (0.937) | 0.846 | 0.197 | 0.907 | CNN-only reference |
| 29 | ensemble at equal budget | 0.916 | 0.929 | 0.837 | 0.181 | 0.915 | submitted |

*Run 28 has no ensemble, so its validation figures are the CNN-only model's; its parenthesised ensemble holdout AUC is the over-budget reference where the forest is added without returning CNN time. The val AUC column is the ensemble's except for run 28.*

### Final model and results

The submitted model is the run-29 ensemble. Its performance on validation, with the two families alongside:

| model | holdout AUC | val recall_ai | val FPR_real | val AUC |
|-------|-------------|---------------|--------------|---------|
| CNN | 0.916 | 0.825 | 0.186 | ~0.89 |
| Random Forest | 0.885 | 0.650 | 0.117 | 0.854 |
| Ensemble | 0.929 | 0.837 | 0.181 | 0.915 |

The CNN validation AUC is approximate, representative of the capacity era. The ensemble clears both targets, recall 0.837 above the 0.80 goal at FPR 0.181 within the 0.20 limit. Per source class, recall is strong on DALL-E 3 and SDXL (about 0.95 and 0.92) and weakest on SD 3 (about 0.71), the historically hardest generator, with SD 2.1 and Midjourney near the 0.80 line.

![Per-source-class recall of the ensemble on validation, against the 0.8 target](figures/fig6_per_class_recall.png)

The confusion matrices show the two families and their ensemble at the calibrated operating point.

![Confusion matrices for the CNN, the Random Forest, and the ensemble on validation](figures/fig5_confusion_matrices.png)

On `validation_augmented` the same model keeps recall above the 0.60 Task 1.3 bar (around 0.73, representative of the capacity era) but its false-positive rate rises well above the limit (around 0.35), because augmented real images get smoothed and recompressed and start to score like AI. The Random Forest is the more robust branch there, as its hand-crafted features are less sensitive to pixel-level distortion. Closing that gap without losing the performance on clean validation is the problem Task 1.3 addresses.

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
