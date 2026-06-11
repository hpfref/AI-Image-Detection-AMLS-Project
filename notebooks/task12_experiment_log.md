# Task 1.2 — Full Experiment Log

## Setup

**Task:** Binary classification real (0) vs ai_generated (1). Labels 1-5 collapse to 1.
**Splits:** train (~29,700), calibration (~1,924), validation (~1,124), predict (100 unlabeled)
**Metric targets:** recall_ai >= 0.80 AND fpr_real <= 0.20 on data/validation/
**Time budget:** 5x reference (reference = ~155.6s on this machine -> BUDGET_S ~778s)

**Pipeline:**
- `prepare.py` (600s): feature extraction, memmap cache
- `train.py` (1800s): RF/LR training + CNN training + calibration + save
- `predict.py` (600s): load model + threshold, run inference

**Measurement note:** All metrics in this log (runs 1-23) were measured during notebook
development in `notebooks/02_task12_modeling_and_tuning.ipynb`. The 5x reference budget
was enforced end-to-end: RF/LR training (~20s) + CNN training (~713s) + calibration/ensemble
overhead (~45s) = total ~778s <= BUDGET_S. The notebook uses `time.monotonic()` and a
deadline-based training loop so training stops automatically when the budget is exhausted.
The predict split has no labels and is not used for any metric evaluation.

---

## Model Families Compared

### Classical Baseline: LR vs RF comparison — Random Forest selected

**Both LR and RF are trained every run; the winner is selected automatically by holdout recall.**
Each run fits:
- Logistic Regression: StandardScaler + LogisticRegression(class_weight="balanced"), C in {0.1, 1.0, 10.0}
- Random Forest: RandomForestClassifier(class_weight="balanced", max_features="sqrt"), n_estimators in {200, 400}

**Run 22 comparison (holdout, at fpr=0.18):**
| Model | Best param | Holdout recall | Holdout AUC | Winner |
|-------|-----------|---------------|-------------|--------|
| LR | C=10.0 | 0.749 | 0.857 | |
| RF | n=400 | **0.781** | **0.885** | YES |

RF wins by 0.032 recall and 0.028 AUC on holdout. RF consistently outperforms LR across all runs.

**Why RF > LR for this task:** the 101-dim feature space contains non-linear interactions
(e.g. JPEG block score interacts with file source in a non-linear way; spectral bands interact
with color skewness). RF captures these interactions via tree splits; LR cannot without
manual feature crosses. RF also handles the class imbalance and varied feature scales
more robustly than regularized LR at this feature dimensionality.

The variable `LR_PIPE` in the code holds the winning model (RF in all recent runs).
The ensemble uses `p_rf = LR_PIPE.predict_proba(F)[:, 1]` which is therefore RF scores.

**Feature dimensions:** 90-dim (runs 1-11) -> 101-dim (runs 12-23)

**90-dim feature set (runs 1-11):**
- Color mean/std per channel (6)
- Color histograms 8-bin per channel (24)
- Laplacian variance (1), Sobel std (1), FFT high-freq ratio (1)
- Patch stats: mean absolute deviation, range, std per channel (9)
- Noise std per channel (3)
- Total: 45... (approximate, context from early runs)

**101-dim feature set (runs 12-23), added in run 12:**
- All above features
- FFT band ratios: 4 spectral power bands 0-5%, 5-15%, 15-30%, 30-50% of Nyquist (4)
- Channel skewness per channel (3)
- Chroma noise: box-blur residual std in Cb/Cr channels (2)
- JPEG block score: mean absolute discontinuity at 8px JPEG block boundaries (2)
- Total: 101 dims

**Rationale for added features:** AI images (especially from diffusion models) show different spectral profiles, less natural color skewness, different chroma noise patterns, and fewer JPEG block artifacts than real camera images.

**Selected classical model (RF, all recent runs):** RandomForestClassifier(n_estimators=400, class_weight="balanced", max_features="sqrt")
**Threshold calibration:** pick_threshold_for_fpr on data/calibration/, target_fpr=0.19

### Neural Network: CNN

**Architecture evolution:**
- Runs 1-3: 3-4 conv layers, k=16, image size 96px
- Runs 4-11: 4-5 conv layers, k=16, 128px (run 7 added 5th conv + MLP head)
- Run 12-14: 5-conv k=16 128px (original focal gamma, then gamma=2.0)
- Run 15-17: gamma=2.0 experiments (ResBlock, DW-sep - both worse)
- Run 18-23: standard 5-conv k=16, gamma=1.5 reverted, resolution experiments
- Run 24: ResNet-SE redesign (stem stride-4 + 4 residual+SE stages 32->256, 192px,
  warmup+cosine LR) - the current architecture

**Current CNN (runs 24-26) -- IN PROGRESS, may still change after the Planned Improvement Runs.**
ResNet-SE, 2.84M params, 192px; the full architecture and training config are in "Run 24 Full
Metrics" (runs 25-26 keep that architecture, changing only budget reserve, eval/selection and
calibration target). The plain-CNN spec (runs 1-23) lives in the "Runs 22-23" section and the
archived notebook old_02. Checkpoint selection is now holdout AUC with a guaranteed final eval
(run 26); depth/width and resolution are still being decided (see "Planned Improvement Runs");
the shipped choice gets pinned in "Key Decisions (shipped model)".

### Ensemble

`p_ens = alpha * p_cnn + (1-alpha) * p_rf`, alpha selected by AUC sweep on holdout.

**IN PROGRESS -- leaning ENS, decided last.** The plain-CNN era (runs 7-23) used alpha=0.40;
runs 24-26 used 0.50. On the reliable holdout (n=2970) the RF adds a steady +0.014/+0.010/+0.012
AUC across runs 24/25/26 - it is not collapsing. Run-25's val crossover that briefly suggested
CNN-only was a small-val artifact (Finding D). So the current lean is ENS, but it is not sealed:
"CNN-only" has never been trained to the true full budget (Finding E), so the final CNN-only vs
ENS call waits until idea 4, run on the winning architecture. See "Improvement Runs -- Tried",
Findings D and E.

---

## Full Run Table

| Run | CNN arch | Img px | k | Gamma | ENS tgt | CNN val rec | CNN val fpr | RF val rec | RF val fpr | ENS val rec | ENS val fpr | ENS val AUC | Result |
|-----|----------|--------|---|-------|---------|-------------|-------------|------------|------------|-------------|-------------|-------------|--------|
| 1 | 3-conv | 96 | 16 | CE | - | 0.649 | 0.207 | 0.561 | 0.176 | - | - | - | no ENS |
| 2 | 3-conv | 128 | 16 | focal | - | 0.689 | 0.191 | 0.692 | 0.186 | - | - | - | no ENS |
| 3 | 4-conv | 128 | 16 | focal | - | 0.745 | 0.234 | 0.692 | 0.186 | - | - | - | no ENS |
| 4 | 4-conv | 128 | 16 | focal | 0.15 | 0.661 | 0.181 | 0.685 | 0.165 | 0.721 | 0.149 | - | FAIL |
| 5 | 4-conv | 128 | 16 | focal | 0.19 | 0.661 | 0.181 | 0.685 | 0.165 | 0.778 | 0.207 | - | FAIL  |
| 6 | 4-conv | 128 | 16 | focal | 0.20 | 0.720 | 0.176 | 0.685 | 0.165 | 0.799 | 0.223 | - | FAIL  |
| 7 | 5-conv+MLP | 128 | 16 | focal~1.5 | ~0.19 | 0.674 | 0.138 | 0.685 | 0.165 | **0.809** | **0.197** | 0.884 | **PASS** |
| 8 | 5-conv+MLP | 128 | 16 | 1.5+sd3x3 | 0.19 | 0.755 | 0.186 | 0.685 | 0.165 | 0.833 | 0.234 | - | FAIL  |
| 9 | 4-conv | 160 | 16 | 1.5+sd3x2 | 0.18 | 0.674 | 0.176 | 0.685 | 0.165 | 0.792 | 0.165 | - | FAIL  |
| 10 | 5-conv | 128 | 16 | 1.5+sd3x2 | 0.18 | 0.709 | 0.186 | 0.685 | 0.165 | 0.775 | 0.176 | - | FAIL |
| 11 | 5-conv | 128 | 16 | 1.5+sd3x2 | 0.19 | 0.709 | 0.186 | 0.685 | 0.165 | 0.817 | 0.250 | - | FAIL |
| 12 | 5-conv+MLP | 128 | 16 | 1.5 | ~0.19 | 0.761 | 0.218 | 0.650 | 0.117 | **0.803** | **0.191** | 0.891 | **PASS** |
| 13 | 5-conv+MLP | 128 | 16 | **2.0** | 0.19 | 0.779 | 0.223 | 0.650 | 0.117 | 0.827 | 0.207 | **0.896** | FAIL  |
| 14 | 5-conv+MLP | 128 | 16 | 2.0 | 0.18 | 0.779 | 0.223 | 0.650 | 0.117 | 0.786 | 0.160 | - | FAIL  |
| 15 | 5-conv+MLP | 128 | 16 | 2.0 | 0.185 | 0.743 | 0.213 | 0.682 | 0.144 | 0.771 | 0.165 | 0.885 | FAIL |
| 16 | ResBlock | 128 | 16 | 2.0 | 0.185 | 0.729 | 0.165 | 0.682 | 0.144 | 0.768 | 0.144 | 0.886 | FAIL |
| 17 | DW-sep | 128 | 32 | 2.0 | 0.185 | 0.723 | 0.213 | 0.682 | 0.144 | 0.740 | 0.149 | 0.875 | FAIL |
| 18 | 5-conv+MLP | 128 | 16 | **1.5** | 0.18 | 0.738 | 0.154 | 0.650 | 0.117 | 0.782 | 0.160 | **0.896** | FAIL  |
| 19 | 5-conv+MLP | **224** | 16 | 1.5 | 0.18 | 0.669 | 0.186 | 0.650 | 0.117 | 0.772 | 0.144 | 0.878 | FAIL |
| 20 | 5-conv+MLP | 128 | **32** | 1.5 | 0.18 | 0.651 | 0.144 | 0.650 | 0.117 | 0.722 | 0.138 | 0.876 | FAIL |
| 21 | 5-conv+MLP | **160** | 16 | 1.5 | 0.18 | 0.746 | 0.207 | 0.650 | 0.117 | 0.795 | 0.176 | 0.878 | FAIL  |
| 22 | 5-conv+MLP | 160 | 16 | 1.5 | 0.19 | 0.772 | 0.229 | 0.669 | 0.128 | **0.809** | **0.197** | 0.878 | **PASS** |
| 23 | 5-conv+MLP | 160 | 16 | 1.5 | 0.19 | 0.724 | 0.197 | 0.669 | 0.128 | **0.811** | **0.170** | 0.888 | **PASS** |
| 24 | **ResNet-SE** | **192** | 32-256 | 1.5 | 0.19 | 0.833 | 0.186 | 0.669 | 0.128 | **0.849** | **0.186** | **0.903** | **PASS** |
| 25 | ResNet-SE | 192 | 32-256 | 1.5 | 0.19 | 0.859 | 0.218 | 0.669 | 0.128 | 0.879 | 0.207 | **0.914** | FAIL* |
| 26 | ResNet-SE | 192 | 32-256 | 1.5 | 0.18 | 0.817 | 0.165 | 0.650 | 0.117 | **0.837** | **0.191** | **0.902** | **PASS** |

Runs 24-26 are a different architecture family (see "Post-Run-23" section below); the `k` column lists their stage widths (32->64->128->256) rather than a single base channel count.

*Run 25 FAIL is a calibration-margin artifact, not a model regression: the threshold hit the target on the calibration set (cal fpr 0.189 at target 0.19), and the val breach (fpr 0.207) is the cal->val sampling gap on ~400 val reals. Every discriminative metric beat run 24 (CNN holdout AUC 0.915 -> 0.919, ENS val AUC 0.903 -> 0.914, ENS val recall 0.849 -> 0.879). See "Improvement Runs -- Tried".

---

## Runs 22-23: plain-CNN baseline (superseded by run 24)

Runs 22 and 23 are stochastic reruns of one recipe - 5-conv BN k=16, 160px, FocalLoss
gamma=1.5, RF(n=400) ensemble at alpha=0.40, calibration target fpr 0.19. They flip PASS/FAIL
on seed and threshold noise (run 22 ENS val 0.809 @ 0.197; run 23 ENS val 0.811 @ 0.170). Run 23
is the baseline run 24 is measured against.

**Run 23 (validation):**
| model | recall_ai | fpr_real | AUC |
|-------|-----------|----------|-----|
| CNN | 0.724 | 0.197 | 0.850 (holdout AUC 0.879) |
| RF | 0.669 | 0.128 | 0.854 |
| ENS | **0.811** | **0.170** | **0.888** |

Budget: RF ~18s + CNN ~715s = ~734s = 4.7x reference.

**Rationale (plain-CNN era; the capacity and ResBlock conclusions are revised in "Post-Run-23"):**
- gamma=1.5 over 2.0: gamma=2.0 hit a higher single-run AUC (0.896, run 13) but swung recall
  0.74-0.83 across seeds; 1.5 is stable. Retained in run 24.
- 160px and k=16: best balance for that shallow net under budget; 224px and k=32 got too few
  steps and underfit (runs 19-20). Both superseded by the run-24 redesign (192px, ResNet-SE).
- Ensemble: CNN and RF make partly uncorrelated errors (pixel artifacts vs file/statistical
  features), which lifted holdout AUC for the weak CNN. Run 24 narrowed that edge (CNN ~ ENS).
- No SD3 upweighting: runs 8-11 pushed fpr to 0.23-0.25; reverted in run 12.
- Calibration: pick_threshold_for_fpr on calibration reals at target 0.19 - data-driven, no
  hardcoded threshold. Unchanged in run 24.

The early-run optimizer ablation (AdamW lr=3e-4 + class weights won) lived in the archived
notebook 02 and is superseded by the run-24 warmup+cosine schedule.

---

## Post-Run-23: Breaking the AUC Plateau (Capacity CNN)

Run 23 passed, but only barely and only by seed luck (runs 21-23 flip-flop PASS/FAIL on
the same config). Rather than ship a borderline model, we asked a sharper question: what
is actually capping performance, and is there a lever we have not pulled?

### Why the old CNN was not powerful enough -- AUC was the only real lever

Look down the run table: across runs 7-23 we changed loss (CE -> focal, gamma 1.5 vs
2.0), resolution (96/128/160/224), width (k=16 vs 32), depth (3/4/5 conv), architecture
(plain, ResBlock, DW-sep), SD3 upweighting, and calibration target. Through all of it the
ensemble AUC barely moved: holdout AUC stayed in 0.877-0.915, val AUC in 0.848-0.896.

That matters because the metric we are graded on, recall_ai at fpr_real <= 0.20, is an
operating point on the ROC curve, and the best achievable recall at a fixed fpr ceiling is
bounded by the AUC. With the feature set and the threshold-calibration procedure fixed,
threshold and loss tuning only slide the operating point along the same curve -- they trade
recall for fpr, they do not raise the curve. So every "improvement" between runs that did
not move AUC was just sliding along a fixed ROC, and the PASS/FAIL flips were threshold and
seed variance, not real discriminative gains.

Conclusion: the bottleneck was the model's discriminative ceiling (AUC), not the operating
point. No amount of further loss or threshold tuning could break it. The only lever left
was a genuinely more powerful model -- one that raises the ROC itself.

### Why the earlier ResBlock attempt (run 16) did not help

It is fair to ask: we already tried a residual architecture in run 16 and it was worse, so
why expect residuals to help now? Two reasons run 16 was not a real test of capacity:

1. It was run at gamma=2.0, the high-variance loss setting. Runs 15-17 all suffered that
   instability (recall swung 0.72-0.83 across seeds), so the ResBlock comparison was
   confounded by a bad loss setting, not a clean architecture test.
2. More fundamentally, residual connections exist to fix gradient flow in *deep* networks
   (tens of layers). Bolting one or two skip connections onto a 5-conv net adds no capacity
   and solves a problem it never had -- vanishing gradients were not the bottleneck at depth
   5. Run 16 also kept the same compute layout (no early downsampling, 3 convs running at
   40x40), so it had the same low parameter count as the plain net. It was a cosmetic change.

Width scaling (k=32, run 20) was the other obvious capacity knob, and it failed too -- but
for a different reason: 4x the parameters made each step ~3x slower, so it got too few steps
and was still underfitting when the budget ran out (AUC 0.876).

### The new approach and why it should work given everything above

Diagnosis of the old net's compute allocation: it stops downsampling after two MaxPools, so
its three deepest convs run at 40x40 and 56% of its 424M MACs/img sit in a single 128->128
conv. It is FLOPs-heavy but parameter-poor (~250k params). It spends the budget on spatial
resolution it cannot exploit instead of on capacity -- which is exactly why both "more time"
(script runs) and "more width" (k=32) failed.

Redesign (ResNet-style, run 24):
- Stem conv 3x3 stride-2 + MaxPool2 downsamples to stride-4 immediately, so the expensive
  stages run at low resolution.
- Four residual stages widening 32 -> 64 -> 128 -> 256, two BasicBlocks each, with
  squeeze-excitation channel attention (near-free on CPU).
- GAP -> Dropout -> Linear head. 192px input.

This gives 2.84M parameters (11x the old net) at 319M MACs/img (0.75x the old net) -- more
capacity at less compute. It fits ~1700 steps in the same budget and affords 192px instead
of 160px. Now the residual blocks serve their real purpose: the net is 17 convs deep, deep
enough that skip connections genuinely aid stable training. We also added a warmup + cosine
LR schedule, which the constant-LR runs never had.

This is the first change since run 7 that moved AUC: CNN holdout AUC 0.879 -> 0.915, ENS
holdout AUC 0.915 -> 0.929, ENS val AUC 0.888 -> 0.903. And critically, the holdout AUC was
still climbing at the budget deadline (best checkpoint at step 1450 of 1491), whereas run 23
plateaued and early-stopped -- direct evidence the extra capacity is being used, not wasted.

---

## Run 24 Full Metrics (Capacity CNN -- last clean PASS)

Run 24 is the last run that passed the operating point cleanly on validation (recall 0.849 @ fpr
0.186). Run 25 (see "Improvement Runs -- Tried") has a strictly higher ROC (ENS val AUC 0.914 vs
0.903) but breached the val fpr ceiling on a too-tight 0.19 calibration target; it becomes the
model direction once recalibrated at 0.18. The full architecture/training spec below still
applies to run 25 (only the budget reserve changed, 70s -> 45s).

Architecture: ResNet-SE. Stem conv3x3 s2 + MaxPool2; 4 stages [2,2,2,2] BasicBlock+SE,
widths 32/64/128/256; GAP -> Dropout(0.2) -> Linear. 2.84M params, 319M MACs/img at 192px,
channels_last. Training: AdamW peak_lr=1e-3 wd=1e-4, warmup+cosine schedule, FocalLoss
gamma=1.5 + class weights, batch 64, 192px. Trained 1491 steps (3.6 epochs) in ~708s; best
holdout checkpoint at step 1450/1491. RF and ensemble recipe unchanged from run 23.

**CNN (ResNet-SE, 192px, thr=0.413):**
| Split | recall_ai | fpr_real | AUC |
|-------|-----------|----------|-----|
| holdout | 0.824 | 0.168 | 0.915 |
| cal | 0.837 | 0.189 | 0.906 |
| val | 0.833 | 0.186 | 0.889 |
| val_aug | 0.693 | 0.305 | 0.786 |

**RF (RandomForest n=400, 101-dim features, thr=0.853):**
| Split | recall_ai | fpr_real | AUC |
|-------|-----------|----------|-----|
| holdout | 0.680 | 0.101 | 0.885 |
| cal | 0.711 | 0.180 | 0.851 |
| val | 0.669 | 0.128 | 0.854 |
| val_aug | 0.501 | 0.203 | 0.694 |

**Ensemble (alpha=0.50, thr=0.604):**
| Split | recall_ai | fpr_real | AUC |
|-------|-----------|----------|-----|
| holdout | 0.851 | 0.162 | 0.929 |
| cal | 0.868 | 0.189 | 0.914 |
| val | **0.849** | **0.186** | **0.903** |
| val_aug | 0.699 | 0.294 | 0.780 |

Per-class recall on val (ENS, AI only): SD2.1=0.80, SDXL=0.93, SD3=0.74, DALL-E3=0.95, Midjourney=0.83
Budget: RF~11s + CNN~708s = ~719s total = 4.6x reference (limit: 5x)

Notable shifts vs run 23: the CNN alone (val AUC 0.889, recall 0.833 @ fpr 0.186) now
matches what the run-23 *ensemble* achieved, and the ensemble alpha moved 0.40 -> 0.50
(the CNN is now strong enough to carry equal weight with the RF). SD3, the historically
hardest class, jumped from 0.63 to 0.74 recall.

---

## Improvement Runs -- Tried

Each idea gets ONE run, documented here; the best is shipped. Ideas 1-2 are done (runs 25-26);
the rest are under "Improvement Runs -- Planned".

### Idea 1 -- Full CNN budget + CNN-only vs ENS (run 25)

**Motivation (the run-24 observation).** The run-24 CNN alone (val 0.833 @ 0.186, AUC 0.889)
nearly equalled the full ensemble (val 0.849 @ 0.186, AUC 0.903): the ensemble added only ~0.015
val AUC, versus ~0.04 in the k=16 era (CNN 0.850 -> ENS 0.888). At the same time the holdout AUC
was still rising at the deadline (best step 1450/1491). Together these said the CNN, not the
ensemble, was now the engine, and it was undertrained, not saturated - so spending budget on the
RF while the CNN trained less may be a net loss, and CNN-only might beat ENS once the CNN got the
full budget. The run trims the CNN-budget reserve from 70s to 45s (the reserve only has to cover
the in-budget post-CNN work - RF refit + calibration + save - plus slower-grader safety), giving
the CNN ~32s more (708s -> 740s), and compares CNN-only against ENS at that budget.

**Run 25 results.** Trace: 1598 steps in 740s (3.8 epochs), best checkpoint at the LAST step
(1598/1598) - still climbing, no plateau (run 24 was 1450/1491 at 708s). Every discriminative
metric improved.

| model | thr | holdout AUC | val recall | val fpr | val AUC | val_aug rec/fpr/AUC |
|-------|-----|-------------|-----------|---------|---------|---------------------|
| CNN | 0.348 | 0.919 | 0.859 | 0.218 | 0.908 | 0.763 / 0.374 / 0.771 |
| RF | 0.853 | 0.885 | 0.669 | 0.128 | 0.854 | 0.501 / 0.203 / 0.694 |
| ENS (alpha 0.50) | 0.567 | 0.929 | 0.879 | 0.207 | 0.914 | 0.769 / 0.396 / 0.767 |

Budget: CNN 739.8s + RF 9.8s = 749.6s = 4.82x reference (CNN-only 4.75x). Within 5x with margin.

**Finding A - calibration margin too tight (target 0.19).** The threshold hit the target on the
calibration set (cal fpr 0.189 for both CNN and ENS), but the cal->val sampling gap on ~400 val
reals pushed val fpr to 0.207-0.218, over the 0.20 ceiling. This is why run 25 is marked FAIL in
the table - a calibration artifact, not a worse model. 0.19 leaves no headroom for the gap. The
fix carries to the next run: calibrate at target 0.18 (~0.02 headroom). Recall barely moves at a
slightly tighter threshold because the ROC is steep there (holdout recall is already 0.877 at fpr
0.174). Run 25 itself is reported at 0.19 - we do not relabel it.

**Finding B - the RF margin is collapsing; lean toward CNN-only (MAJOR).** On validation the
longer-trained CNN alone (AUC 0.908) now exceeds the *previous* run-24 ensemble (0.903), and the
RF's marginal val AUC fell from +0.014 (run 24: CNN 0.889 -> ENS 0.903) to +0.006 (run 25: 0.908
-> 0.914). The original "ensemble is smart" assumption was correct only while the CNN was weak:
the RF made partly uncorrelated errors that lifted a low ROC. Now the CNN's ROC is high enough
that the RF mostly rides along. Honest caveat: on the larger, less-noisy holdout (n=2970) the
ensemble still leads (CNN 0.919 vs ENS 0.929, RF margin +0.010, also shrinking from +0.014), and
the RF is the low-fpr branch (val fpr 0.128) that tightens the operating point. So the val
crossover is partly noise - the durable signal is the shrinking RF margin plus the still-climbing
CNN, not a clean "CNN beat ENS". Budget-wise, dropping the RF returns ~10s to the CNN, which at
the current marginal rate buys back roughly the AUC the RF added - so CNN-only is about a wash on
score and wins on simplicity and on putting all budget into the part that is still improving. A
strong lean toward shipping CNN-only, confirmed (not sealed) in idea 2, judged on holdout.
**Revised by run 26 (Finding D): this lean does not hold up - the +0.006 was a noisy-val
artifact, and CNN-only has never actually been given the full budget (see Finding E).**

### Idea 2 -- Reproducible step-based eval + AUC selection + 0.18 calibration (run 26)

**What changed.** Step-based, tail-weighted eval (16 evals, dense in the last ~25%) plus a
guaranteed final eval of the last weights; checkpoint selected by holdout AUC (not recall@fpr);
calibration target 0.19 -> 0.18 (Finding A). Architecture, loss, LR and budget unchanged.

**Run 26 results.** Trace: 1607 steps in 743s, best holdout AUC at the last step (1607/1607),
final eval 8.1s. First clean PASS since run 24.

| model | thr | holdout AUC | val recall | val fpr | val AUC | val_aug rec/fpr/AUC |
|-------|-----|-------------|-----------|---------|---------|---------------------|
| CNN | 0.406 | 0.916 | 0.817 | 0.165 | 0.888 | 0.725 / 0.353 / 0.774 |
| RF | 0.858 | 0.885 | 0.650 | 0.117 | 0.854 | 0.501 / 0.203 / 0.694 |
| ENS (alpha 0.50) | 0.605 | 0.928 | 0.837 | 0.191 | 0.902 | 0.726 / 0.348 / 0.771 |

Budget: CNN 743.5s (incl. 8.1s final eval) + RF 11.3s = 754.8s = 4.85x; post-CNN work 19.4s, well
under the 45s reserve. Both candidates pass the operating point at target 0.18 (CNN-only 0.817 @
0.165, ENS 0.837 @ 0.191). The budget assert holds with ~23s of headroom under 5x.

**Finding C - idea 2 buys stability, not AUC.** Holdout AUC is unchanged within noise vs run 25
(CNN 0.919 -> 0.916, ENS 0.929 -> 0.928); the model is plateaued at this capacity across runs
24-26. What idea 2 delivered is reproducible, threshold-independent selection and a clean,
in-budget operating point at 0.18 - the first clean PASS since run 24. The guaranteed final eval
did not change the pick (the endpoint won selection in every run) and cannot lower quality; it
only prevents shipping a stale checkpoint.

**Finding D - the RF-margin "collapse" (Finding B) was noise; ENS is the current pick.** The
holdout ENS-over-CNN AUC margin across runs is +0.014 (24), +0.010 (25), +0.012 (26): the RF
reliably adds ~0.01-0.014 on the n=2970 set. Run-25's val margin of +0.006 that prompted the
CNN-only lean was a small-val artifact (that run's CNN scored unusually high on val, n~188). The
apparent recall "dropoff" (ENS val recall 0.879 run 25 -> 0.837 run 26) is likewise not a
regression: run-25's 0.879 sat at an illegal fpr (0.207 > 0.20 = FAIL); at a legal operating point
it is the same ROC (holdout AUC 0.928 across 24-26). run 26 sits at 0.837 @ 0.191 with ~0.009 fpr
headroom; calibrating slightly looser (target ~0.185 -> val ~0.196) would reclaim ~0.01 recall if
we want to push. Corrected conclusion: keep ENS. CNN-only stays a viable simpler fallback (it also
passes), but the decision is deferred (Finding E).

**Finding E - CNN-only has been handicapped; the fair test comes last.** In every run so far the
CNN trains to `BUDGET_S - 45s` because the 45s reserve is held for the RF. "CNN-only" was then
read off that same RF-reserved CNN, i.e. trained to 5x minus the RF's reserve, ~30s short of its
real budget. So we have never measured a true max-budget CNN-only. Given the plateau (+30s buys
~0.004 holdout AUC) versus the RF's +0.012, ENS very likely still wins, but the comparison as run
is biased toward ENS. The clean test (train the CNN to the full 5x with only a final-eval/save
reserve, then compare to the best ENS) is deferred to after the architecture is settled - see
"Improvement Runs -- Planned".

---

## Improvement Runs -- Planned

Each idea below gets ONE run, moved to "Tried" once executed; the best is shipped. Order: idea 3
next (settle the architecture), then idea 4 (final CNN-only vs ENS on that architecture).

3. **Depth/width A-B (NEXT).** The model is plateaued at holdout AUC ~0.916 CNN / ~0.928 ENS
   across runs 24-26, so capacity (not time, not resolution) is the only remaining AUC lever - and
   even here expect modest movement. Run it one-variable-at-a-time from the run-26 baseline, NOT a
   grid, and stop early:
   - *Variant 3a (first):* `CNN_BLOCKS = (2,2,2,3)` - a 3rd BasicBlock in stage 4 (deepest, runs at
     6x6 @192px so extra params are cheap in MACs). 2.84M -> 4.04M params, 348M -> 390M MACs/img
     (+12%). It costs ~12% throughput, so it trains ~1,430 steps vs run-26's 1,607 in the same
     budget - the test is precisely whether the added capacity outweighs the lost steps within 5x.
   - *Decision gate:* compare holdout AUC (the reliable n=2970 metric) vs run 26 (CNN 0.916 /
     ENS 0.928). If it does not beat run 26 by more than noise (~0.003), capacity at the top is
     tapped out - stop and keep run 26. If it moves, that variant becomes the architecture and we
     try one more (stage-4 width 320). Resolution (176 vs 192) is not its own run: runs 25-26
     already showed more steps do not help (plateau), so 176's only edge (more steps) is moot -
     only revisit if a capacity bump makes 192px drop below ~1,600 benchmarked steps.
   - Whatever wins (or run 26, if nothing beats it) becomes the architecture for idea 4.

4. **Full CNN train time + final CNN-only vs ENS (LAST, on the winning architecture).** Settles
   whether we ship CNN-only or ENS, fairly. Reason (Finding E): every run so far trained the CNN
   to `BUDGET_S - 45s` because the 45s reserve is held for the RF, so "CNN-only" was always ~30s
   short of its real budget - we have never measured a true max-budget CNN-only. Here, on the
   architecture idea 3 picks, run the CNN to the full 5x with only a final-eval/save reserve
   (`CNN_BUDGET_RESERVE_S ~15`, no RF reserve, CNN ~763s) to get the genuine max CNN-only, and
   compare its holdout AUC to the best ENS (run 26 ENS, or idea-3's ENS if better). If max
   CNN-only still trails the ENS (expected, given the plateau: +30s ~+0.004 AUC vs the RF's
   +0.012), ship ENS; if it catches up, ship CNN-only (simpler). Doing this last avoids re-running
   it for every architecture variant - we only need the CNN-vs-ENS call on the final model.

**Will not retry (no improvement in runs 1-23, evidence in the run table above):** gamma=2.0
focal (high seed variance, runs 13-17), depthwise-separable convs (run 17 worse), SD3
upweighting (pushed fpr > 0.20, runs 8-11), 224px and k=32-from-scratch (too few steps,
underfit, runs 19-20), and large width/parameter scaling (overfit risk at ~27k images).

---

## Key Decisions and Rationale (shipped model) -- TO FILL

Placeholder. Fill once the planned improvement runs settle the model we ship for Task 1.2.
Capture only the decisions that define THAT model, each with a a short "why" grounded in the
run that decided it:
- Architecture (ResNet-SE widths/blocks/SE) and input resolution.
- CNN-only vs ensemble (does the RF still add AUC?), and alpha if ensembled. (!NOTE TO LLM WHEN WRITING THIS: DECISION TO EVEN THINK OF AN ENS IS NOT TRIVIAL -> TELL HOW COMBINING these 2 MODELS CAN RESULT IN BETTER PERFROMACNE WHEN CONBINED VS WHEN USED STANDALONE & THIS WAS AN OBVIOUS DECISION EARLY ON WHEN CNN WAS BAD AND ENS ALWAYS SIGNIFICANTLY BETTER)
- LR schedule (warmup+cosine, peak).
- Checkpoint-selection metric (AUC vs recall@fpr) and whether snapshot averaging is used.
- Calibration target and the realised val operating point.

Until then the live rationale lives in "Post-Run-23" (why the capacity redesign) and
"Planned Improvement Runs" (what is still being decided).

---

## What Remains for Task 1.3

Task 1.3 requires robustness to distortions (blur, compression, resize) using
data/calibration_augmented/ and data/validation_augmented/. Whatever 1.2 model we settle on
(after the planned improvement runs above) is the starting point - 1.3 is not tackled here.

The core 1.3 problem shows up consistently across the 1.2 runs: on val_aug the false-positive
rate runs far above the clean-val fpr (roughly 0.29-0.34 vs ~0.17-0.20), because augmented real
images get smoothed/recompressed and start scoring like AI. The RF is the more robust branch
(val_aug fpr ~0.19-0.20 vs the CNN's ~0.30+), since hand-crafted features are less sensitive to
pixel-level distortion. Recall on val_aug is not the blocker - it clears the >=0.60 target in
recent runs; the false-positive rate is.

Task 1.3 target: recall_ai >= 0.60 on val_aug while fpr <= 0.20.
Main work: reduce false positives on augmented real images - augmentation-aware training and
calibration on calibration_augmented - without giving back the clean-val operating point.
