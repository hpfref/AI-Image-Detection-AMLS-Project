"""
Task 1.2 -- Modeling and Tuning under Time Constraints  (35/100 points)

Reads:    artifacts/prepared/               (features + image caches from prepare.py)
Writes:   artifacts/task02/best.pt          -- CNN state dict + arch config
          artifacts/task02/rf_model.pkl     -- RandomForest (classical family)
          artifacts/task02/threshold.json   -- ensemble threshold + alpha

Shipped model (notebooks/task12_experiment_log.md "Key Decisions"):
    ResNet-SE capacity CNN (build_capacity_cnn, 192px, warmup+cosine LR, checkpoint
    chosen by holdout AUC) ensembled with a RandomForest(400) on 101-dim engineered
    features: p_ens = alpha*p_cnn + (1-alpha)*p_rf, alpha by holdout AUC sweep.
    Threshold calibrated automatically on data/calibration/ at target FPR 0.18.

Budget design:
    The grader hard-kills each script at --timeout_seconds. We train the CNN to
    (start + timeout - OVERHEAD_S) and reserve OVERHEAD_S for the in-budget post-CNN
    work that MUST finish so we can ship: the guaranteed final eval + RF refit +
    calibration + saving the 3 artifacts. OVERHEAD_S=120 covers that even on a
    ~3x-slower grader. The best CNN checkpoint is also written incrementally during
    training (PDF advice), and the optional all-split metric printing is time-guarded
    so documentation can never delay shipping.

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
from _lib.model import build_capacity_cnn, cnn_scores, train_capacity

# Shipped architecture / training config (run 29)
WIDTHS        = (32, 64, 128, 256)
BLOCKS        = (2, 2, 2, 2)
TRAIN_IMG     = 192
CHANNELS_LAST = True
ENS_TARGET    = 0.18    # calibration FPR target (kept at 0.18; see log Finding F)
OVERHEAD_S    = 60      # post-CNN reserve: final eval + 1 calibration pass + save (RF is done pre-CNN)
METRIC_SAFETY = 20      # stop optional metric eval this many seconds before the timeout


# ---------------------------------------------------------------------------
# Data loading
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


def _print_metrics(tag, y_true, scores, thr):
    from sklearn.metrics import roc_auc_score
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"  [{tag}] recall_ai={rec:.3f}  fpr_real={fpr:.3f}  auc={auc:.3f}  "
          f"thr={thr:.4f}  tp={tp}  fn={fn}  fp={fp}  tn={tn}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1.2: train + calibrate")
    parser.add_argument("--timeout_seconds", type=int, required=True)
    args = parser.parse_args()
    _start = time.monotonic()
    hard_deadline = _start + args.timeout_seconds

    _seed.set_deterministic(0)
    prep_dir = _io.ARTIFACTS_ROOT / "prepared"
    out_dir  = _io.ensure_artifact_dir("task02")

    if not (prep_dir / "n_fit.npy").exists():
        print("ERROR: artifacts/prepared/ missing -- run prepare.py first", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 1: load prepared arrays
    # ------------------------------------------------------------------
    print("loading prepared splits...", flush=True)
    X_fit,  y_fit,  src_fit,  F_fit  = _load_split(prep_dir, "fit")
    X_hold, y_hold, src_hold, F_hold = _load_split(prep_dir, "hold")
    X_cal,  y_cal,  src_cal,  F_cal  = _load_split(prep_dir, "cal")
    X_val,  y_val,  src_val,  F_val  = _load_split(prep_dir, "val")
    X_va,   y_va,   src_va,   F_va   = _load_split(prep_dir, "va")
    mean = np.load(str(prep_dir / "mean.npy"))
    std  = np.load(str(prep_dir / "std.npy"))
    print(f"  fit={len(X_fit)}  hold={len(X_hold)}  cal={len(X_cal)}  "
          f"val={len(X_val)}  va={len(X_va)}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: RandomForest FIRST -- deterministic, ~10s; save it immediately so the
    # classical model can never be lost, and keep it out of the post-CNN reserve.
    # ------------------------------------------------------------------
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    rf_t0 = time.monotonic()
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0,
                                class_weight="balanced", max_features="sqrt")
    rf.fit(F_fit, y_fit)
    joblib.dump(rf, str(out_dir / "rf_model.pkl"))
    p_rf_hold = rf.predict_proba(F_hold)[:, 1]
    p_rf_cal  = rf.predict_proba(F_cal)[:, 1]
    print(f"RF trained + saved in {time.monotonic() - rf_t0:.1f}s  "
          f"(pre-CNN, off the post-train reserve)", flush=True)

    # ------------------------------------------------------------------
    # Step 3: train the capacity CNN to (timeout - reserve)
    # ------------------------------------------------------------------
    cnn_deadline = hard_deadline - OVERHEAD_S
    print(f"CNN deadline: {cnn_deadline - time.monotonic():.0f}s of training  "
          f"(timeout={args.timeout_seconds}s  reserve={OVERHEAD_S}s)", flush=True)

    ckpt_meta = {
        "arch": "capacity_v1", "widths": WIDTHS, "blocks": BLOCKS,
        "img_size": TRAIN_IMG, "channels_last": CHANNELS_LAST, "mean": mean, "std": std,
    }
    cnn = build_capacity_cnn(widths=WIDTHS, blocks=BLOCKS)
    best, history, total_steps, final_eval_s = train_capacity(
        cnn, X_fit, y_fit, X_hold, y_hold, mean, std, cnn_deadline,
        target_size=TRAIN_IMG, channels_last=CHANNELS_LAST,
        ckpt_path=out_dir / "best.pt", ckpt_meta=ckpt_meta, verbose=True,
    )
    if best["state"] is None:
        print("ERROR: CNN training produced no checkpoint", file=sys.stderr)
        return 1
    cnn.load_state_dict(best["state"])
    cnn.eval()
    print(f"CNN trained {total_steps} steps; best holdout AUC={best['auc']:.3f} "
          f"recall={best['recall']:.3f} at step {best['step']} (final eval {final_eval_s:.1f}s)",
          flush=True)

    # ------------------------------------------------------------------
    # Step 4: ensemble alpha (reuse the best checkpoint's holdout scores -- no extra CNN
    # pass) + calibrate the threshold on data/calibration/ (the one unavoidable CNN pass)
    # ------------------------------------------------------------------
    p_cnn_hold = best["hold_scores"]
    best_alpha, best_auc = 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 11):
        auc = float(roc_auc_score(y_hold, a * p_cnn_hold + (1 - a) * p_rf_hold))
        if auc > best_auc:
            best_auc, best_alpha = auc, float(a)
    alpha = best_alpha
    print(f"  alpha={alpha:.2f}  holdout ENS AUC={best_auc:.4f}", flush=True)

    t_cal0 = time.monotonic()
    p_cnn_cal = cnn_scores(cnn, X_cal, mean, std, target_size=TRAIN_IMG)
    cal_pass_s = time.monotonic() - t_cal0
    p_ens_cal = alpha * p_cnn_cal + (1 - alpha) * p_rf_cal
    thr = pick_threshold_for_fpr(p_ens_cal[y_cal == 0], target_fpr=ENS_TARGET)
    cal_fpr = float((p_ens_cal[y_cal == 0] >= thr).mean())
    print(f"  ENS thr={thr:.4f}  cal_fpr_realised={cal_fpr:.3f}  (target {ENS_TARGET})", flush=True)

    # ------------------------------------------------------------------
    # Step 5: save best.pt + threshold.json (rf_model.pkl already on disk)
    # ------------------------------------------------------------------
    t_save0 = time.monotonic()
    torch.save({"state": best["state"], "thr": thr, **ckpt_meta}, str(out_dir / "best.pt"))
    write_threshold_json(out_dir / "threshold.json",
                         {"thr": thr, "alpha": alpha, "target_fpr": ENS_TARGET})
    save_s = time.monotonic() - t_save0

    # Post-CNN reserve report: how much of OVERHEAD_S we actually used, so we can judge it.
    post_cnn_s = time.monotonic() - cnn_deadline
    print(f"saved best.pt + threshold.json.  POST-CNN RESERVE USED = {post_cnn_s:.1f}s of {OVERHEAD_S}s "
          f"[final_eval={final_eval_s:.1f}s + cal_pass={cal_pass_s:.1f}s + save={save_s:.1f}s + alpha/overhead]. "
          f"A 2x/3x-slower grader would use ~{2*post_cnn_s:.0f}s / ~{3*post_cnn_s:.0f}s.", flush=True)

    # ------------------------------------------------------------------
    # Step 6: metrics (time-guarded; documentation only, never blocks shipping)
    # ------------------------------------------------------------------
    thr_cnn = pick_threshold_for_fpr(p_cnn_cal[y_cal == 0], target_fpr=ENS_TARGET)
    thr_rf  = pick_threshold_for_fpr(p_rf_cal[y_cal == 0],  target_fpr=ENS_TARGET)
    p_cnn = {"holdout": p_cnn_hold, "cal": p_cnn_cal}   # already computed
    p_rf  = {"holdout": p_rf_hold,  "cal": p_rf_cal}
    Xs = {"val": X_val, "va": X_va}
    Fs = {"val": F_val, "va": F_va}
    for s in ("val", "va"):
        if time.monotonic() > hard_deadline - METRIC_SAFETY:
            print(f"  (skipping {s} metrics -- near timeout)", flush=True)
            continue
        p_cnn[s] = cnn_scores(cnn, Xs[s], mean, std, target_size=TRAIN_IMG)
        p_rf[s]  = rf.predict_proba(Fs[s])[:, 1]

    ys = {"holdout": y_hold, "cal": y_cal, "val": y_val, "va": y_va}
    print("\n--- CNN / RF / ENS metrics (calibrated thresholds) ---", flush=True)
    for s in ("holdout", "cal", "val", "va"):
        if s not in p_cnn:
            continue
        _print_metrics(f"cnn  {s}", ys[s], p_cnn[s], thr_cnn)
        _print_metrics(f"rf   {s}", ys[s], p_rf[s],  thr_rf)
        _print_metrics(f"ens  {s}", ys[s], alpha * p_cnn[s] + (1 - alpha) * p_rf[s], thr)

    print(f"\ntrain.py done in {time.monotonic() - _start:.1f}s  (budget={args.timeout_seconds}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
