"""
Model architectures plus training/inference utilities.

Final Task 1.2 model (shipped): a ResNet-SE capacity CNN (build_capacity_cnn) trained
with FocalLoss(gamma=1.5) + class weights and a warmup+cosine LR schedule, the checkpoint
selected by holdout AUC. At inference it is ensembled with a RandomForest on 101-dim
engineered features. See notebooks/task12_experiment_log.md "Key Decisions" for the full
rationale. The plain Appendix-B / 5-conv baselines from earlier runs were removed once the
capacity CNN superseded them (history lives in the experiment log + the archived notebook).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss
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


# ---------------------------------------------------------------------------
# Final Task 1.2 architecture: ResNet-SE capacity CNN
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-excitation channel attention. Cheap on CPU, recalibrates channels."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):
        s = x.mean(dim=(2, 3))                 # GAP -> (N, C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class BasicBlock(nn.Module):
    """ResNet basic block: 2x conv3x3-BN with SE on the residual branch, then skip."""

    def __init__(self, in_c, out_c, stride=1, se_reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.se    = SEBlock(out_c, se_reduction)
        self.down  = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + idt, inplace=True)


class CapacityCNN(nn.Module):
    """Stem conv3x3 s2 + MaxPool2 (stride-4) -> 4 residual+SE stages -> GAP -> Linear.

    Downsampling to stride-4 immediately puts the expensive stages at low resolution,
    buying ~11x the parameters of the old 5-conv net at less compute (more capacity where
    it is usable). Standard 3x3 convs (not depthwise-separable): AI-artifact detection
    needs joint spatial+channel mixing.
    """

    def __init__(self, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2),
                 se_reduction=8, dropout=0.2, num_classes=2):
        super().__init__()
        c0 = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        stages, in_c = [], c0
        for si, (w, nb) in enumerate(zip(widths, blocks)):
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                stages.append(BasicBlock(in_c, w, stride=stride, se_reduction=se_reduction))
                in_c = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(in_c, num_classes),
        )

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_capacity_cnn(**kw) -> nn.Module:
    """Factory for the shipped ResNet-SE CNN (defaults = the run-29 architecture)."""
    return CapacityCNN(**kw)


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
    target_size: int = 192,
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
    target_size: int = 192,
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


def cosine_lr(step, peak, floor, warmup, total):
    """Linear warmup for `warmup` steps, then cosine decay from peak to floor over `total`."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * prog))


def train_capacity(
    model: nn.Module,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    deadline: float,
    *,
    peak_lr: float = 1e-3,
    lr_floor_frac: float = 0.01,
    warmup_cap: int = 300,
    batch: int = 64,
    n_evals: int = 16,
    tail_power: float = 2.0,
    patience: int = 12,
    target_size: int = 192,
    channels_last: bool = True,
    ckpt_path=None,
    ckpt_meta: dict | None = None,
    verbose: bool = True,
) -> tuple:
    """Train to `deadline`; return (best, history, total_steps, final_eval_s).

    Selection: the single checkpoint with the highest HOLDOUT AUC (smooth, threshold-
    independent); recall@fpr.20 is logged for the trace only. The schedule (warmup+cosine)
    self-calibrates to measured steps/s at step 30, eval is step-based and tail-weighted
    (gaps shrink toward the deadline so the still-improving end is sampled densely), and the
    final weights are always scored once after the loop (the model improves to the last step).

    If `ckpt_path` is given, the best checkpoint is written to disk on each new best (PDF
    advice: regularly persist the best checkpoint). The payload is
    {"state", **ckpt_meta} so it is directly loadable by predict.py; the deadline is set by
    the caller with enough reserve that the final RF + calibration + save still complete.
    """
    from sklearn.metrics import roc_auc_score

    from _lib.calibration import pick_threshold_for_fpr

    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)
    loss_fn = FocalLoss(gamma=1.5, weight=class_weights(y_fit))
    yt = torch.from_numpy(y_fit).long()
    floor = lr_floor_frac * peak_lr
    n = len(X_fit)

    best = {"auc": -1.0, "recall": -1.0, "fpr": 1.0, "state": None, "thr": 0.5, "step": 0,
            "hold_scores": None}
    history: list[dict] = []
    no_improve = step = 0
    loss_win: list[float] = []
    warmup = warmup_cap            # provisional until steps/s measured at step 30
    total_est = None
    schedule, si = [], 0           # step-based eval steps, gaps shrinking toward the deadline
    t_start = time.monotonic()

    def _persist():
        if ckpt_path is not None and best["state"] is not None:
            payload = {"state": best["state"]}
            if ckpt_meta:
                payload.update(ckpt_meta)
            torch.save(payload, str(ckpt_path))

    def do_eval(tag, lr_val):
        nonlocal no_improve
        scores = cnn_scores(model, X_hold, mean, std, target_size=target_size)
        thr = pick_threshold_for_fpr(scores[y_hold == 0], target_fpr=0.20)
        yp = (scores >= thr).astype(int)
        tp = int(((yp == 1) & (y_hold == 1)).sum()); fn = int(((yp == 0) & (y_hold == 1)).sum())
        fp = int(((yp == 1) & (y_hold == 0)).sum()); tn = int(((yp == 0) & (y_hold == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        auc = float(roc_auc_score(y_hold, scores))
        lma = float(np.mean(loss_win)) if loss_win else float("nan")
        history.append({"step": step, "lr": lr_val, "loss": lma,
                        "recall": rec, "fpr": fpr, "auc": auc, "tag": tag})
        if verbose:
            print(f"  step={step:5d} lr={lr_val:.2e} loss={lma:.3f} "
                  f"hold recall={rec:.3f} fpr={fpr:.3f} auc={auc:.3f} ({tag})", flush=True)
        if auc > best["auc"] + 1e-4:
            best.update(auc=auc, recall=rec, fpr=fpr, thr=thr, step=step,
                        state={k: v.clone() for k, v in model.state_dict().items()},
                        hold_scores=scores)   # reused for the ensemble alpha sweep (no extra pass)
            no_improve = 0
            _persist()
        else:
            no_improve += 1
        model.train()

    model.train()
    lr = peak_lr
    while time.monotonic() < deadline:
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            if time.monotonic() >= deadline:
                break
            if step == 30:  # calibrate LR schedule + eval plan to measured throughput
                sps = (time.monotonic() - t_start) / 30.0
                total_est = step + int((deadline - time.monotonic()) / sps)
                warmup = min(warmup_cap, int(0.05 * total_est))
                # tail-weighted eval steps: gaps shrink toward total_est so the still-improving
                # end of training is sampled densely; total stays ~n_evals (bounded eval cost).
                raw = [int(total_est * (1 - (1 - k / n_evals) ** tail_power))
                       for k in range(1, n_evals + 1)]
                schedule = sorted(s for s in set(raw) if s > step)
                if verbose:
                    print(f"  [sched] {sps:.3f}s/step  total_est={total_est}  warmup={warmup}  "
                          f"evals={len(schedule)}  last_steps={schedule[-3:]}", flush=True)
            lr = cosine_lr(step, peak_lr, floor, warmup, total_est or 2000)
            for g in opt.param_groups:
                g["lr"] = lr

            ix = perm[i:i+batch]
            xt = batch_to_chw(ix, X_fit, mean, std, target_size=target_size)
            if channels_last:
                xt = xt.contiguous(memory_format=torch.channels_last)
            loss = loss_fn(model(xt), yt[torch.from_numpy(ix)])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            loss_win.append(float(loss.item()))
            if len(loss_win) > 50:
                loss_win.pop(0)

            if si < len(schedule) and step >= schedule[si]:
                do_eval("periodic", lr)
                si += 1
                if no_improve >= patience:   # guard only; AUC is smooth so this rarely trips
                    if verbose:
                        print(f"  early stop: no AUC improvement in {patience} evals", flush=True)
                    deadline = 0.0           # force the outer while to exit
                    break

    # Guaranteed final eval: the model improves to the very last step, so always score the
    # final weights once and let AUC-selection consider them.
    t_fe = time.monotonic()
    do_eval("final", lr)
    final_eval_s = time.monotonic() - t_fe
    return best, history, step, final_eval_s
