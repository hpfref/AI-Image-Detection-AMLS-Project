"""
Model architectures plus training/inference utilities.

Includes the reference CNN from PDF Appendix B as a starting point. Per PDF
§1.2 you must compare at least two model families in the report -- e.g. this
CNN vs. a classical engineered-feature baseline -- but only the single best
pipeline is packaged in solution/.

Final Task 1.2 model: build_cnn_bn (5-conv BN, k=16, 160px, FocalLoss gamma=1.5).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference architecture (PDF Appendix B)
# ---------------------------------------------------------------------------

def build_appendix_b_cnn(k: int = 32) -> nn.Module:
    """Reference CNN from PDF Appendix B.

    Two nn.Sequential blocks (`features` + `classifier`) are wrapped here as a
    single nn.Sequential so it can be used directly with optimizer / loss.
    Treat as a starting point, not a final solution.
    """

    # --- BEGIN: copied from PDF Appendix B ---
    features = nn.Sequential(
        nn.Conv2d(3, k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * k, 2),
    )
    # --- END: copied from PDF Appendix B ---

    return nn.Sequential(features, classifier)


def build_classical_baseline():
    """Engineered-feature baseline (e.g. color/texture stats + logistic reg).

    Required for PDF §1.2 ("at least two different model families") so the
    report can compare a classical baseline against the CNN.
    """
    raise NotImplementedError("Task 1.2: implement classical baseline")


# ---------------------------------------------------------------------------
# Final Task 1.2 architecture
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples to focus training on hard ones.

    gamma=1.5 chosen over 2.0 for stability: lower gamma reduces variance
    across random seeds at the cost of slightly lower peak AUC.
    """

    def __init__(self, gamma: float = 1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets, sample_weight=None):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def build_cnn_bn(k: int = 16, num_classes: int = 2) -> nn.Module:
    """5-conv BN architecture (Task 1.2 final model).

    Standard 3x3 conv preserves joint spatial+channel mixing needed to detect
    per-pixel AI artifacts. k=16 fits ~1700s of CPU training within the
    timeout_seconds=1800 budget.
    """
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    return nn.Sequential(
        _block(3,    k),   nn.MaxPool2d(2),
        _block(k,  2*k),   nn.MaxPool2d(2),
        _block(2*k, 4*k),
        _block(4*k, 8*k),
        _block(8*k, 8*k),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(8 * k, 4 * k), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(4 * k, num_classes),
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def class_weights(y: np.ndarray) -> torch.Tensor:
    """Balanced class weights: n / (2 * class_count)."""
    counts = np.bincount(y, minlength=2).astype(np.float64)
    n = counts.sum()
    w = n / (2.0 * np.maximum(counts, 1))
    return torch.from_numpy(w.astype(np.float32))


def batch_to_chw(
    idx_chunk: np.ndarray,
    X_u8: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    target_size: int = 160,
) -> torch.Tensor:
    """uint8 memmap slice -> normalized CHW float tensor, downsampled to target_size.

    Normalization happens before permuting to match eval-time transform exactly.
    """
    block = X_u8[idx_chunk].astype(np.float32) / 255.0
    block = (block - mean) / std                             # (N, H, W, 3)
    t = torch.from_numpy(block).permute(0, 3, 1, 2).contiguous()
    if t.shape[-1] != target_size:
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t


@torch.no_grad()
def cnn_scores(
    model: nn.Module,
    X_u8: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch: int = 128,
    target_size: int = 160,
) -> np.ndarray:
    """Run inference and return AI-class softmax probabilities."""
    model.eval()
    out = []
    for i in range(0, len(X_u8), batch):
        idx = np.arange(i, min(i + batch, len(X_u8)))
        xt = batch_to_chw(idx, X_u8, mean, std, target_size=target_size)
        logits = model(xt)
        out.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out)


def train_cnn(
    model: nn.Module,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    deadline: float,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch: int = 64,
    eval_every_s: float = 30.0,
    patience: int = 8,
    target_size: int = 160,
    verbose: bool = True,
) -> dict:
    """Train CNN until deadline; return best checkpoint dict by holdout recall.

    Returned dict keys: 'recall', 'state' (state_dict copy), 'thr'.
    """
    from _lib.calibration import pick_threshold_for_fpr

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights(y_fit)
    loss_fn = FocalLoss(gamma=1.5, weight=w)
    yt_fit = torch.from_numpy(y_fit).long()
    n = len(X_fit)

    best = {"recall": -1.0, "state": None, "thr": 0.5}
    last_eval = time.monotonic()
    no_improve = 0
    step = 0
    loss_win: list[float] = []
    WINDOW = 50

    model.train()
    while time.monotonic() < deadline:
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            if time.monotonic() >= deadline:
                break
            ix = perm[i:i+batch]
            xt = batch_to_chw(ix, X_fit, mean, std, target_size=target_size)
            logits = model(xt)
            loss = loss_fn(logits, yt_fit[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            loss_win.append(float(loss.item()))
            if len(loss_win) > WINDOW:
                loss_win.pop(0)

            if time.monotonic() - last_eval >= eval_every_s:
                scores = cnn_scores(model, X_hold, mean, std, target_size=target_size)
                thr = pick_threshold_for_fpr(scores[y_hold == 0], target_fpr=0.20)
                y_pred = (scores >= thr).astype(int)
                tp = int(((y_pred == 1) & (y_hold == 1)).sum())
                fn = int(((y_pred == 0) & (y_hold == 1)).sum())
                fp = int(((y_pred == 1) & (y_hold == 0)).sum())
                tn = int(((y_pred == 0) & (y_hold == 0)).sum())
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                loss_ma = float(np.mean(loss_win)) if loss_win else 0.0
                if verbose:
                    print(f"  step={step:5d}  loss={loss_ma:.3f}  holdout recall={rec:.3f}  fpr={fpr:.3f}")
                if rec > best["recall"] + 1e-4:
                    best = {
                        "recall": rec,
                        "state":  {k: v.clone() for k, v in model.state_dict().items()},
                        "thr":    thr,
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
                last_eval = time.monotonic()
                if no_improve >= patience:
                    if verbose:
                        print(f"  early stop after {patience} evals without improvement")
                    return best
    return best
