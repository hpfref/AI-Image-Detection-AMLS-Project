"""Quick eval of the saved task02 artifacts on all splits. No retraining."""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "../solution")

import numpy as np
import torch
from pathlib import Path

from _lib.calibration import pick_threshold_for_fpr, read_threshold_json
from _lib.model import build_cnn_bn, cnn_scores

ROOT = Path("../solution")
PREP = ROOT / "artifacts" / "prepared"
OUT  = ROOT / "artifacts" / "task02"
IMG  = 224

def load_split(name):
    n   = int(np.load(str(PREP / f"n_{name}.npy"))[0])
    X   = np.lib.format.open_memmap(str(PREP / f"X_{name}.mmap"), mode="r",
                                     dtype=np.uint8, shape=(n, IMG, IMG, 3))
    y   = np.load(str(PREP / f"y_{name}.npy"))
    F   = np.load(str(PREP / f"F_{name}.npy"))
    return X, y, F

def metrics(y_true, scores, thr):
    yp = (scores >= thr).astype(int)
    tp = int(((yp==1)&(y_true==1)).sum()); fn = int(((yp==0)&(y_true==1)).sum())
    fp = int(((yp==1)&(y_true==0)).sum()); tn = int(((yp==0)&(y_true==0)).sum())
    rec = tp/(tp+fn) if tp+fn else 0.0
    fpr = fp/(fp+tn) if fp+tn else 0.0
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true))>1 else float("nan")
    return rec, fpr, auc

# Load model
ckpt    = torch.load(str(OUT/"best.pt"), map_location="cpu", weights_only=False)
cnn     = build_cnn_bn(k=ckpt["k"]); cnn.load_state_dict(ckpt["state"]); cnn.eval()
mean, std = ckpt["mean"], ckpt["std"]
img_sz  = ckpt["img_size"]

import joblib
rf      = joblib.load(str(OUT/"rf_model.pkl"))
thr_d   = read_threshold_json(OUT/"threshold.json")
thr, alpha = float(thr_d["thr"]), float(thr_d["alpha"])
print(f"alpha={alpha}  thr={thr:.4f}")

print(f"\n{'split':8s}  {'cnn_rec':>8} {'cnn_fpr':>8}  {'rf_rec':>7} {'rf_fpr':>7}  {'ens_rec':>8} {'ens_fpr':>8} {'ens_auc':>8}")
for name in ("hold","cal","val","va"):
    X, y, F = load_split(name)
    p_cnn = cnn_scores(cnn, X, mean, std, target_size=img_sz)
    p_rf  = rf.predict_proba(F)[:,1]
    p_ens = alpha*p_cnn + (1-alpha)*p_rf

    thr_cnn = pick_threshold_for_fpr(p_cnn[y==0], target_fpr=0.19)
    thr_rf  = pick_threshold_for_fpr(p_rf[y==0],  target_fpr=0.19)

    cr, cf, _ = metrics(y, p_cnn, thr_cnn)
    rr, rf_, _ = metrics(y, p_rf, thr_rf)
    er, ef, ea = metrics(y, p_ens, thr)
    print(f"{name:8s}  {cr:8.3f} {cf:8.3f}  {rr:7.3f} {rf_:7.3f}  {er:8.3f} {ef:8.3f} {ea:8.4f}")
