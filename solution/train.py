"""
Task 1.2 -- Modeling and Tuning under Time Constraints  (35/100 points)

Reads:    artifacts/prepared/          (features from prepare.py)
Writes:   artifacts/task02/best.pt         -- CNN state dict + config
          artifacts/task02/rf_model.pkl    -- winning classical model
          artifacts/task02/threshold.json  -- ensemble threshold + alpha

Goals (PDF §1.2):
    - Maximize recall_ai under fpr_real <= 0.20.
    - Calibrate threshold AUTOMATICALLY on data/calibration/ -- no manual values.
    - Report at least TWO model families (classical baseline + neural).
    - CPU only, no internet, no pretrained downloads at runtime.
    - Training time bounded by timeout_seconds CLI argument.

Budget design:
    deadline = start + timeout_seconds - SAFETY_S
    SAFETY_S = 90s covers RF training (~45s on grader) + calibration scoring
    (~20s) + save (~5s) + 20s buffer. CNN gets the rest (~1710s on our machine).

CLI:
    python train.py --timeout_seconds 1800
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from _lib import io as _io
from _lib import seed as _seed
from _lib.calibration import pick_threshold_for_fpr, write_threshold_json
from _lib.model import build_cnn_bn, cnn_scores, train_cnn

# Hyperparameters matching run 23 (pass: recall=0.809, fpr=0.197)
CNN_K        = 16
IMG_TRAIN    = 160
ENS_TARGET   = 0.19   # calibration FPR target for ensemble threshold
SAFETY_S     = 90     # post-training overhead budget (see docstring)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_split(out_dir, name: str):
    n = int(np.load(str(out_dir / f"n_{name}.npy"))[0])
    X = np.lib.format.open_memmap(
        str(out_dir / f"X_{name}.mmap"), mode="r", dtype=np.uint8,
        shape=(n, _io.IMG_SIZE, _io.IMG_SIZE, 3),
    )
    y   = np.load(str(out_dir / f"y_{name}.npy"))
    src = np.load(str(out_dir / f"src_{name}.npy"))
    F   = np.load(str(out_dir / f"F_{name}.npy"))
    return X, y, src, F


# ---------------------------------------------------------------------------
# Classical baseline: LR vs RF, winner selected by holdout recall
# ---------------------------------------------------------------------------

def train_classical(F_fit, y_fit, F_hold, y_hold):
    """Train LR (C grid) and RF (n_estimators grid); return (winner, elapsed_s)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.monotonic()
    print("training classical models...")

    best_lr = {"recall": -1.0, "pipe": None, "label": ""}
    for C in (0.1, 1.0, 10.0):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr",    LogisticRegression(class_weight="balanced", max_iter=2000, C=C)),
        ])
        pipe.fit(F_fit, y_fit)
        p = pipe.predict_proba(F_hold)[:, 1]
        thr = pick_threshold_for_fpr(p[y_hold == 0], target_fpr=0.18)
        rec = float(((p >= thr) & (y_hold == 1)).sum() / max(int((y_hold == 1).sum()), 1))
        print(f"  LR C={C:>5}: holdout recall={rec:.3f}")
        if rec > best_lr["recall"]:
            best_lr = {"recall": rec, "pipe": pipe, "label": f"LR C={C}"}

    best_rf = {"recall": -1.0, "pipe": None, "label": ""}
    for n_est in (200, 400):
        rf = RandomForestClassifier(
            n_estimators=n_est, n_jobs=-1, random_state=0,
            class_weight="balanced", max_features="sqrt",
        )
        rf.fit(F_fit, y_fit)
        p = rf.predict_proba(F_hold)[:, 1]
        thr = pick_threshold_for_fpr(p[y_hold == 0], target_fpr=0.18)
        rec = float(((p >= thr) & (y_hold == 1)).sum() / max(int((y_hold == 1).sum()), 1))
        print(f"  RF n={n_est:>4}: holdout recall={rec:.3f}")
        if rec > best_rf["recall"]:
            best_rf = {"recall": rec, "pipe": rf, "label": f"RF n={n_est}"}

    if best_rf["recall"] >= best_lr["recall"]:
        print(f"  winner: {best_rf['label']} (recall={best_rf['recall']:.3f})")
        winner = best_rf["pipe"]
    else:
        print(f"  winner: {best_lr['label']} (recall={best_lr['recall']:.3f})")
        winner = best_lr["pipe"]

    elapsed = time.monotonic() - t0
    print(f"  classical training: {elapsed:.1f}s")
    return winner, elapsed


# ---------------------------------------------------------------------------
# Ensemble alpha selection (by holdout AUC)
# ---------------------------------------------------------------------------

def select_alpha(p_cnn, p_rf, y):
    from sklearn.metrics import roc_auc_score
    best_alpha, best_auc = 0.5, -1.0
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p_ens = alpha * p_cnn + (1 - alpha) * p_rf
        auc = float(roc_auc_score(y, p_ens))
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
    print(f"  alpha={best_alpha}  holdout ENS AUC={best_auc:.4f}")
    return best_alpha


# ---------------------------------------------------------------------------
# Metrics printer
# ---------------------------------------------------------------------------

def _print_metrics(tag, y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"  [{tag}] recall_ai={rec:.3f}  fpr_real={fpr:.3f}  auc={auc:.3f}  "
          f"thr={thr:.4f}  tp={tp}  fn={fn}  fp={fp}  tn={tn}")
    return rec, fpr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: train + calibrate")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()
    _start = time.monotonic()

    _seed.set_deterministic(0)
    prep_dir = _io.ARTIFACTS_ROOT / "prepared"
    out_dir  = _io.ensure_artifact_dir("task02")

    if not (prep_dir / "n_fit.npy").exists():
        print("ERROR: artifacts/prepared/ missing -- run prepare.py first", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 1: load prepared arrays
    # ------------------------------------------------------------------
    print("loading prepared splits...")
    X_fit,  y_fit,  src_fit,  F_fit  = _load_split(prep_dir, "fit")
    X_hold, y_hold, src_hold, F_hold = _load_split(prep_dir, "hold")
    X_cal,  y_cal,  src_cal,  F_cal  = _load_split(prep_dir, "cal")
    X_val,  y_val,  src_val,  F_val  = _load_split(prep_dir, "val")
    X_va,   y_va,   src_va,   F_va   = _load_split(prep_dir, "va")
    mean = np.load(str(prep_dir / "mean.npy"))
    std  = np.load(str(prep_dir / "std.npy"))
    print(f"  fit={len(X_fit)}  hold={len(X_hold)}  cal={len(X_cal)}  "
          f"val={len(X_val)}  va={len(X_va)}")

    # ------------------------------------------------------------------
    # Step 2: classical model (RF + LR comparison)
    # ------------------------------------------------------------------
    classical_pipe, rf_elapsed = train_classical(F_fit, y_fit, F_hold, y_hold)

    # ------------------------------------------------------------------
    # Step 3: CNN -- deadline accounts for actual RF time + post overhead
    # ------------------------------------------------------------------
    overhead = max(SAFETY_S, rf_elapsed * 2 + 30)
    cnn_deadline = _start + args.timeout_seconds - overhead
    remaining = cnn_deadline - time.monotonic()
    print(f"CNN budget: {remaining:.0f}s  (timeout={args.timeout_seconds}s  overhead={overhead:.0f}s)")

    cnn = build_cnn_bn(k=CNN_K)
    best = train_cnn(
        cnn, X_fit, y_fit, X_hold, y_hold,
        mean=mean, std=std, deadline=cnn_deadline,
        lr=3e-4, weight_decay=1e-4,
        batch=64, eval_every_s=30.0, patience=8,
        target_size=IMG_TRAIN, verbose=True,
    )
    if best["state"] is None:
        print("ERROR: CNN training produced no checkpoint", file=sys.stderr)
        return 1
    cnn.load_state_dict(best["state"])
    print(f"CNN best holdout recall={best['recall']:.3f}")

    # ------------------------------------------------------------------
    # Step 4: ensemble alpha selection on holdout
    # ------------------------------------------------------------------
    print("selecting ensemble alpha...")
    p_cnn_hold = cnn_scores(cnn, X_hold, mean, std, target_size=IMG_TRAIN)
    p_rf_hold  = classical_pipe.predict_proba(F_hold)[:, 1]
    alpha = select_alpha(p_cnn_hold, p_rf_hold, y_hold)

    # ------------------------------------------------------------------
    # Step 5: calibrate ensemble threshold on data/calibration/
    # ------------------------------------------------------------------
    print("calibrating ensemble threshold on calibration split...")
    p_cnn_cal = cnn_scores(cnn, X_cal, mean, std, target_size=IMG_TRAIN)
    p_rf_cal  = classical_pipe.predict_proba(F_cal)[:, 1]
    p_ens_cal = alpha * p_cnn_cal + (1 - alpha) * p_rf_cal
    thr = pick_threshold_for_fpr(p_ens_cal[y_cal == 0], target_fpr=ENS_TARGET)
    print(f"  thr={thr:.4f}  cal_fpr_realised={(p_ens_cal[y_cal==0] >= thr).mean():.3f}")

    # ------------------------------------------------------------------
    # Step 6: evaluate and print metrics on all splits
    # ------------------------------------------------------------------
    print("\n--- Standalone CNN ---")
    thr_cnn = pick_threshold_for_fpr(p_cnn_cal[y_cal == 0], target_fpr=ENS_TARGET)
    for tag, X, y, F in [
        ("holdout", X_hold, y_hold, F_hold),
        ("cal",     X_cal,  y_cal,  F_cal),
        ("val",     X_val,  y_val,  F_val),
        ("va",      X_va,   y_va,   F_va),
    ]:
        p = cnn_scores(cnn, X, mean, std, target_size=IMG_TRAIN)
        _print_metrics(f"cnn  {tag}", y, p, thr_cnn)

    print("\n--- Standalone RF ---")
    thr_rf = pick_threshold_for_fpr(p_rf_cal[y_cal == 0], target_fpr=ENS_TARGET)
    for tag, F, y in [
        ("holdout", F_hold, y_hold),
        ("cal",     F_cal,  y_cal),
        ("val",     F_val,  y_val),
        ("va",      F_va,   y_va),
    ]:
        p = classical_pipe.predict_proba(F)[:, 1]
        _print_metrics(f"rf   {tag}", y, p, thr_rf)

    print("\n--- Ensemble ---")
    for tag, X, y, F in [
        ("holdout", X_hold, y_hold, F_hold),
        ("cal",     X_cal,  y_cal,  F_cal),
        ("val",     X_val,  y_val,  F_val),
        ("va",      X_va,   y_va,   F_va),
    ]:
        p_cnn = cnn_scores(cnn, X, mean, std, target_size=IMG_TRAIN)
        p_rf  = classical_pipe.predict_proba(F)[:, 1]
        p_ens = alpha * p_cnn + (1 - alpha) * p_rf
        rec, fpr = _print_metrics(f"ens  {tag}", y, p_ens, thr)

    # ------------------------------------------------------------------
    # Step 7: save artifacts
    # ------------------------------------------------------------------
    print("\nsaving artifacts...")
    import joblib

    torch.save(
        {"state": best["state"], "k": CNN_K, "img_size": IMG_TRAIN,
         "mean": mean, "std": std},
        str(out_dir / "best.pt"),
    )
    joblib.dump(classical_pipe, str(out_dir / "rf_model.pkl"))
    write_threshold_json(
        out_dir / "threshold.json",
        {"thr": thr, "alpha": alpha, "target_fpr": ENS_TARGET},
    )

    elapsed = time.monotonic() - _start
    print(f"train.py done in {elapsed:.1f}s  (budget={args.timeout_seconds}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
