"""Cross-day compensation model: Leave-One-Day-Out evaluation.

Deployment scenario
-------------------
The model is trained on one set of recording days and used on a *different*
day. Days are therefore never mixed inside a split. Each fold trains on whole
sessions, validates on one held-out session, and tests on another held-out
session.

Why this differs from the previous version
------------------------------------------
1. Leave-One-Day-Out CV       : 5 honest cross-day estimates instead of 1.
2. Per-recording sEMG scaling : each file is normalised by its own sEMG
                                statistics. Leakage-free, because sEMG is
                                available at deployment time.
3. Per-session force gain     : optional short calibration segment at the start
                                of each recording (``--calib-seconds``). This
                                mirrors the real protocol of calibrating when
                                electrodes are fitted. The calibration window is
                                excluded from all reported metrics.
4. Base model with dynamics   : TANH -> LPF restored, so the model can represent
                                the hysteresis loops visible in Force-vs-sEMG.
5. Scale-relative target      : the compensator predicts error normalised by the
                                recording's own base-output scale, so no
                                training-day offset is baked into predictions.
6. Per-fold base fitting      : the base model is refitted on each fold's
                                training sessions only.
"""

from pathlib import Path
import argparse
import copy
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# Settings
# =========================================================
SEED = 42
Ts = 0.0005

SEQ = 100
WINDOW_STRIDE = 5  # use 1 for all windows (slow on CPU)
BATCH = 64
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR = 3e-4
WD = 1e-4
GRAD_CLIP = 1.0
WARMUP_STEPS = 500
SKIP_BATCH_ABORT_RATIO = 0.20

CONV_C = 32
KERNEL = 3
LSTM_H = 48
D_MODEL = 48
HEADS = 4
TF_LAYERS = 2
FF = 96
DROP = 0.2

XCOL = "Cali_LPF_PASF_sEMG"
YCOL = "Force"
TCOL = "Time"

# Base model LPF cutoff bounds (Hz).
FC_MIN = 1.0
FC_MAX = 200.0

# Recording sessions, in chronological order. Splits never mix these.
SESSIONS = ["1st_Comp", "2nd_Comp", "3rd_Base", "4th_Valid", "5th_Comp"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent / "Estimation"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def session_files(name):
    return sorted((DATA / name).glob("Data*.csv"))


# =========================================================
# Data helpers
# =========================================================
def assert_finite(array, name):
    """Guard: fail fast with a clear source name if data is non-finite."""
    arr = np.asarray(array)
    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(
            f"Non-finite values in {name}: {bad} bad value(s) out of {arr.size}"
        )


def tensor_from_numpy(array, name):
    """Convert to torch tensor and confirm the numpy bridge did not corrupt data."""
    t = torch.tensor(array, dtype=torch.float32)
    if not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum().item())
        raise ValueError(
            f"Non-finite values in {name} after torch.tensor: "
            f"{bad} bad value(s) out of {t.numel()}"
        )
    return t


def load(path):
    df = pd.read_csv(path)
    values = df[[XCOL, YCOL]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1)

    x = values.loc[valid, XCOL].to_numpy(float)
    y = values.loc[valid, YCOL].to_numpy(float)

    # Guard: raw columns must be finite after parsing.
    assert_finite(x, f"{path} -> {XCOL}")
    assert_finite(y, f"{path} -> {YCOL}")

    if TCOL in df.columns:
        t = pd.to_numeric(df.loc[valid, TCOL], errors="coerce").to_numpy(float)
    else:
        t = np.arange(len(x)) * Ts

    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        t = np.arange(len(x)) * Ts

    return t, x, y


def metrics(y, y_hat):
    mse = mean_squared_error(y, y_hat)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, y_hat)),
        "R2": float(r2_score(y, y_hat)),
    }


# =========================================================
# Base model: TANH -> LPF
#
#   z(k) = gain * tanh(slope * x_n(k))
#   G(s) = wc / (s + wc),  wc = 2*pi*fc      (bilinear discretisation)
#
# The LPF gives the model dynamics, so it can represent the hysteresis loops
# seen in the Force-vs-sEMG plots. A static tanh cannot: it is single-valued.
# =========================================================
def lpf(x, fc):
    K = 2.0 / Ts
    fc = float(np.clip(fc, FC_MIN, FC_MAX))
    wc = 2.0 * np.pi * fc
    b0 = wc / (K + wc)
    a1 = (wc - K) / (K + wc)
    return lfilter([b0, b0], [1.0, a1], x)


def base_model(theta, x_n, use_lpf=True):
    if use_lpf:
        gain, slope, fc = theta
        return lpf(gain * np.tanh(slope * x_n), fc)
    gain, slope = theta[0], theta[1]
    return gain * np.tanh(slope * x_n)


def semg_scale_for(x, q=99.0):
    """Per-recording sEMG scale. Uses only the input signal -> no leakage."""
    positive = x[x > 0]
    if len(positive) == 0:
        scale = float(np.max(np.abs(x)))
        return scale if scale > 1e-9 else 1.0
    scale = float(np.percentile(positive, q))
    return scale if scale > 1e-9 else 1.0


def fit_base_model(paths, use_lpf=True, max_nfev=600):
    """Fit the base model on the given (training-only) recordings."""
    data = []
    for path in paths:
        _, x, y = load(path)
        data.append((x / semg_scale_for(x), y))

    def residual(theta):
        out = []
        for x_n, y in data:
            out.append(base_model(theta, x_n, use_lpf) - y)
        return np.concatenate(out)

    if use_lpf:
        x0 = np.array([1.0, 1.0, 20.0])
        lower = np.array([0.0, 0.0, FC_MIN])
        upper = np.array([50.0, 50.0, FC_MAX])
    else:
        x0 = np.array([1.0, 1.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([50.0, 50.0])

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
    )
    return result.x.copy(), result


GAIN_MIN, GAIN_MAX = 1e-3, 1e3
ACTIVITY_FRACTION = 0.2  # of the recording's 95th-percentile base output


def calibration_masks(y_base_raw, calib_seconds):
    """Split a recording into a calibration segment and an evaluation segment.

    The calibration segment collects the first ``calib_seconds`` worth of
    *active* samples, where activity is detected from the base output - which
    is derived from sEMG alone, so choosing the window never looks at force.
    Keying off activity rather than wall-clock matters: several recordings sit
    at rest for the first ten seconds, and a rest-only window makes the gain
    fit noise-dominated and collapses it toward zero.

    Everything up to the end of the calibration segment is excluded from
    evaluation, so calibration force is never scored.
    """
    n = len(y_base_raw)
    if calib_seconds <= 0:
        return None, np.ones(n, dtype=bool)

    magnitude = np.abs(y_base_raw)
    threshold = ACTIVITY_FRACTION * float(np.percentile(magnitude, 95))
    active_idx = np.flatnonzero(magnitude > threshold)

    if len(active_idx) == 0:
        return None, np.ones(n, dtype=bool)

    # Contiguous window starting at the first detected contraction, so the
    # calibration stays a short bounded segment rather than spanning the record.
    start = int(active_idx[0])
    requested = max(1, int(round(calib_seconds / Ts)))

    # Always leave enough of the recording to evaluate on. Short recordings
    # (the 15 s files) would otherwise be consumed entirely by the window.
    for min_eval in (max(SEQ + 1, n // 4), SEQ + 1):
        end = min(start + requested, n - min_eval)
        if end > start:
            break
    else:
        return None, np.ones(n, dtype=bool)

    if end <= start:
        return None, np.ones(n, dtype=bool)

    calib_mask = np.zeros(n, dtype=bool)
    calib_mask[start:end] = True

    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[end:] = True

    return calib_mask, eval_mask


def estimate_session_gain(y, y_base_raw, calib_mask):
    """Least-squares scalar gain from the calibration segment.

    Mirrors a real deployment calibration: a brief reference contraction is
    recorded when the electrodes are fitted, and its force is known. Returns
    (gain, ok) where ok is False if the segment carried too little signal to
    identify a gain.
    """
    if calib_mask is None or not calib_mask.any():
        return 1.0, False

    y_c = y[calib_mask]
    yb_c = y_base_raw[calib_mask]

    den = float(np.dot(yb_c, yb_c))
    if den < 1e-12:
        return 1.0, False

    gain = float(np.dot(y_c, yb_c)) / den
    if not np.isfinite(gain) or gain <= 0:
        return 1.0, False

    # A gain pinned at the clip bounds means the fit did not identify anything.
    clipped = float(np.clip(gain, GAIN_MIN, GAIN_MAX))
    return clipped, GAIN_MIN < clipped < GAIN_MAX


def build_records(paths, theta, calib_seconds, use_lpf=True):
    """Build per-recording arrays with per-session normalisation applied."""
    records = []
    for path in paths:
        t, x, y = load(path)

        semg_scale = semg_scale_for(x)
        x_n = x / semg_scale
        y_base_raw = base_model(theta, x_n, use_lpf)

        calib_mask, eval_mask = calibration_masks(y_base_raw, calib_seconds)
        gain, gain_ok = estimate_session_gain(y, y_base_raw, calib_mask)
        y_base = gain * y_base_raw

        if calib_seconds > 0 and not gain_ok:
            print(
                f"  WARNING: {path.parent.name}/{path.stem}: calibration did not "
                f"identify a gain (gain={gain:.5g}); using it unscaled.",
                file=sys.stderr,
            )
        if not eval_mask.any():
            raise ValueError(
                f"{path}: calibration consumed the whole recording; "
                f"reduce --calib-seconds"
            )

        # Per-recording output scale, derived from the base output only, so it
        # is computable at deployment. Keeps the compensation proportional to
        # the session instead of carrying a training-day offset.
        scale = float(np.percentile(np.abs(y_base), 95))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        records.append(
            {
                "path": path,
                "session": path.parent.name,
                "t": t,
                "x": x,
                "x_n": x_n,
                "y": y,
                "y_base": y_base,
                "error": y - y_base,
                "semg_scale": semg_scale,
                "session_gain": gain,
                "scale": scale,
                "eval_mask": eval_mask,
            }
        )
    return records


def record_arrays(record):
    """Session-normalised features and target for one recording."""
    feat = np.column_stack([record["x_n"], record["y_base"] / record["scale"]])
    target = record["error"] / record["scale"]
    return feat, target


def windows(x, y, stride):
    if len(x) < SEQ:
        return (
            np.empty((0, SEQ, x.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.array([], dtype=int),
        )
    index = np.arange(SEQ - 1, len(x), stride)
    X_all = np.lib.stride_tricks.sliding_window_view(x, SEQ, axis=0)
    X_all = np.transpose(X_all, (0, 2, 1))
    win_idx = index - (SEQ - 1)
    X = X_all[win_idx].astype(np.float32)
    Y = y[index].astype(np.float32)
    return X, Y, index


def record_windows(record, x_scaler, stride):
    """Windows for one recording, with the calibration segment removed."""
    feat, target = record_arrays(record)
    if x_scaler is not None:
        feat = x_scaler.transform(feat)
    X, Y, index = windows(feat, target, stride)
    if len(index) == 0:
        return X, Y, index
    keep = record["eval_mask"][index]
    return X[keep], Y[keep], index[keep]


def fix_scaler_scale(scaler):
    """Guard: zero-variance features must not divide by zero during transform."""
    scale = np.asarray(scaler.scale_, dtype=float)
    scale[np.abs(scale) < 1e-8] = 1.0
    scaler.scale_ = scale


def fit_feature_scaler(records):
    """StandardScaler on inputs only, fitted on training recordings."""
    scaler = StandardScaler()
    scaler.fit(np.vstack([record_arrays(r)[0] for r in records]))
    fix_scaler_scale(scaler)
    return scaler


def records_to_xy(records, x_scaler, stride, label=""):
    X_parts, Y_parts = [], []
    for r in records:
        X, Y, _ = record_windows(r, x_scaler, stride)
        if len(X) == 0:
            continue
        assert_finite(X, f"{r['path']} features (windows)")
        assert_finite(Y, f"{r['path']} targets (windows)")
        X_parts.append(X)
        Y_parts.append(Y)

    if not X_parts:
        raise ValueError(f"No usable windows for split {label or '?'}")

    X_all = np.concatenate(X_parts)
    Y_all = np.concatenate(Y_parts)
    if label:
        print(f"  {label}: {X_all.shape}", flush=True)
    return X_all, Y_all


# =========================================================
# Model
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CompensationModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, CONV_C, KERNEL, padding="same"),
            nn.GroupNorm(num_groups=8, num_channels=CONV_C),
            nn.GELU(),
            nn.Conv1d(CONV_C, CONV_C, KERNEL, padding="same"),
            nn.GroupNorm(num_groups=8, num_channels=CONV_C),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(CONV_C, LSTM_H, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * LSTM_H, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL, SEQ)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=FF,
            dropout=DROP,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=TF_LAYERS, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROP),
            nn.Linear(D_MODEL, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


def apply_warmup_lr(optimizer, step, warmup_steps, base_lr):
    """Linear LR warmup for the first `warmup_steps` optimizer steps."""
    if step <= warmup_steps:
        lr = base_lr * (step / warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr


def run_epoch(model, loader, optimizer, loss_fn, train=True, warmup_state=None):
    model.train(mode=train)
    losses = []
    skipped = 0
    total_batches = len(loader)

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        # Guard: skip batches that already produce a non-finite loss.
        if not torch.isfinite(loss):
            skipped += 1
            continue

        if train:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Guard: do not step if gradients exploded to non-finite values.
            if not torch.isfinite(grad_norm):
                skipped += 1
                continue

            if warmup_state is not None:
                warmup_state["step"] += 1
                apply_warmup_lr(
                    optimizer,
                    warmup_state["step"],
                    warmup_state["warmup_steps"],
                    warmup_state["base_lr"],
                )

            optimizer.step()

        losses.append(loss.item())

    if not losses:
        return float("nan"), skipped, total_batches
    return float(np.mean(losses)), skipped, total_batches


def state_has_nonfinite(state):
    """Return True if any tensor in a state_dict contains NaN/Inf."""
    for value in state.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return True
    return False


def save_checkpoint(state, path):
    """Atomically save only finite checkpoints; surface OS errors clearly."""
    target = Path(path)
    if state_has_nonfinite(state):
        print(
            f"ERROR: Refusing to save checkpoint with non-finite weights to "
            f"{target.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)

    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, target)
    except OSError as exc:
        print(
            f"ERROR: Failed to save checkpoint to {target.resolve()}\n"
            f"OS error: {exc}",
            file=sys.stderr,
        )
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


@torch.no_grad()
def predict_records(model, records, x_scaler, stride):
    """Predict compensation and rebuild physical force for each recording."""
    model.eval()
    rows = []

    for r in records:
        X, _, index = record_windows(r, x_scaler, stride)
        if len(X) == 0:
            continue
        assert_finite(X, f"{r['path']} inference features (windows)")

        pred_parts = []
        for i in range(0, len(X), BATCH):
            xb = tensor_from_numpy(X[i : i + BATCH], f"{r['path']} inference batch")
            pred_parts.append(model(xb.to(DEVICE)).cpu().numpy())
        pred_norm = np.concatenate(pred_parts)

        # Undo the per-recording normalisation: the compensation scales with
        # this session, not with the training days.
        compensation = pred_norm * r["scale"]
        y_true = r["y"][index]
        y_base = r["y_base"][index]
        y_hat = y_base + compensation

        rows.append(
            {
                "path": r["path"],
                "session": r["session"],
                "index": index,
                "t": r["t"][index],
                "x": r["x"][index],
                "y": y_true,
                "y_base": y_base,
                "compensation": compensation,
                "y_hat": y_hat,
                "session_gain": r["session_gain"],
                "base_metrics": metrics(y_true, y_base),
                "final_metrics": metrics(y_true, y_hat),
            }
        )
    return rows


@torch.no_grad()
def split_rmse(model, records, x_scaler, stride):
    """Mean physical RMSE over a split (base and final)."""
    rows = predict_records(model, records, x_scaler, stride)
    if not rows:
        return float("nan"), float("nan")
    base = float(np.mean([r["base_metrics"]["RMSE"] for r in rows]))
    final = float(np.mean([r["final_metrics"]["RMSE"] for r in rows]))
    return base, final


# =========================================================
# Plots
# =========================================================
def save_training_loss_plot(history, run_summary, out_path, title_suffix=""):
    """Save train/val loss curves with best and early-stop epochs."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, [h["train_loss"] for h in history], label="Train loss", linewidth=1.5)
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val loss", linewidth=1.5)

    best_ep = run_summary["best_epoch"]
    ax.axvline(
        best_ep,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"Best epoch ({best_ep})",
    )
    if run_summary["early_stopped"]:
        stop_ep = run_summary["early_stop_epoch"]
        ax.axvline(
            stop_ep,
            color="red",
            linestyle=":",
            linewidth=1.2,
            label=f"Early stop ({stop_ep})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (normalised)")
    ax.set_title(f"Training / validation loss per epoch{title_suffix}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_force_plots(rows, out_dir):
    """Save real / EDM base / predicted force line plots with metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for row in rows:
        stem = row["path"].stem
        t = row["t"]
        y_true = row["y"]
        y_hat = row["y_hat"]
        y_base = row["y_base"]
        final_m = row["final_metrics"]
        base_m = row["base_metrics"]

        step = max(1, len(t) // 5000)
        sl = slice(None, None, step)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t[sl], y_true[sl], label="Real Force", linewidth=1.2)
        ax.plot(
            t[sl],
            y_base[sl],
            label="EDM Output (base model)",
            linewidth=1.0,
            alpha=0.85,
            linestyle="--",
        )
        ax.plot(
            t[sl],
            y_hat[sl],
            label="Predicted Force (base + compensation)",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force")
        ax.set_title(f"{row['session']} / {stem}: Real vs EDM vs Predicted Force")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        metric_text = (
            f"Final  RMSE={final_m['RMSE']:.4f}  MAE={final_m['MAE']:.4f}  "
            f"R2={final_m['R2']:.4f}     "
            f"Base   RMSE={base_m['RMSE']:.4f}  MAE={base_m['MAE']:.4f}  "
            f"R2={base_m['R2']:.4f}     "
            f"session_gain={row['session_gain']:.4f}"
        )
        fig.text(0.5, 0.02, metric_text, ha="center", va="bottom", fontsize=9)
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

        summary.append(
            {
                "Session": row["session"],
                "Data": stem,
                "Session_Gain": row["session_gain"],
                **{f"Base_{k}": v for k, v in base_m.items()},
                **{f"Final_{k}": v for k, v in final_m.items()},
            }
        )

    return summary


def save_prediction_csvs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pd.DataFrame(
            {
                TCOL: row["t"],
                XCOL: row["x"],
                YCOL: row["y"],
                "BaseModel_Output": row["y_base"],
                "Compensation_Output": row["compensation"],
                "Final_Output": row["y_hat"],
                "BaseModel_Error": row["y"] - row["y_base"],
                "Final_Error": row["y"] - row["y_hat"],
            }
        ).to_csv(out_dir / f"{row['path'].stem}.csv", index=False)


# =========================================================
# One LODO fold
# =========================================================
def check_epoch_losses(train_loss, val_loss, train_skipped, train_batches, epoch):
    """Abort training when an epoch produces no usable finite losses."""
    if not np.isfinite(train_loss):
        print(
            f"ERROR: All {train_batches} training batches were skipped in epoch "
            f"{epoch}; no finite train loss available.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not np.isfinite(val_loss):
        print(
            f"ERROR: Validation loss is non-finite after epoch {epoch}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if train_batches and (train_skipped / train_batches) > SKIP_BATCH_ABORT_RATIO:
        ratio = 100.0 * train_skipped / train_batches
        print(
            f"ERROR: Skipped {train_skipped}/{train_batches} training batches "
            f"({ratio:.1f}%) in epoch {epoch}. Training is unstable.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_fold(test_session, val_session, args):
    """Train on the remaining sessions, validate on one, test on a held-out day."""
    set_seed(SEED)

    train_sessions = [
        s for s in SESSIONS if s not in (test_session, val_session)
    ]
    train_paths = [p for s in train_sessions for p in session_files(s)]
    val_paths = session_files(val_session)
    test_paths = session_files(test_session)

    print(f"\n{'=' * 62}")
    print(f"FOLD  test={test_session}  val={val_session}")
    print(f"  train sessions : {', '.join(train_sessions)} ({len(train_paths)} files)")
    print(f"  val   session  : {val_session} ({len(val_paths)} files)")
    print(f"  test  session  : {test_session} ({len(test_paths)} files)")
    print(f"{'=' * 62}", flush=True)

    use_lpf = not args.no_lpf

    # Base model is refitted on this fold's training sessions only.
    print("Fitting base model on training sessions...", flush=True)
    theta, base_result = fit_base_model(train_paths, use_lpf=use_lpf)
    if use_lpf:
        print(
            f"  TANH gain={theta[0]:.4f} slope={theta[1]:.4f} | LPF fc={theta[2]:.2f} Hz"
            f"  (cost={base_result.cost:.4f})",
            flush=True,
        )
    else:
        print(
            f"  TANH gain={theta[0]:.4f} slope={theta[1]:.4f}"
            f"  (cost={base_result.cost:.4f})",
            flush=True,
        )

    train_records = build_records(train_paths, theta, args.calib_seconds, use_lpf)
    val_records = build_records(val_paths, theta, args.calib_seconds, use_lpf)
    test_records = build_records(test_paths, theta, args.calib_seconds, use_lpf)

    gains = [r["session_gain"] for r in test_records]
    print(
        f"  test session gains: min={min(gains):.4f} max={max(gains):.4f} "
        f"mean={np.mean(gains):.4f}",
        flush=True,
    )

    x_scaler = fit_feature_scaler(train_records)

    print("Building windows...", flush=True)
    X_train, Y_train = records_to_xy(train_records, x_scaler, args.stride, "train")
    X_val, Y_val = records_to_xy(val_records, x_scaler, args.stride, "val")

    train_loader = DataLoader(
        TensorDataset(
            tensor_from_numpy(X_train, "X_train"),
            tensor_from_numpy(Y_train, "Y_train"),
        ),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,  # Avoid batch-size-1 normalisation edge cases.
    )
    val_loader = DataLoader(
        TensorDataset(
            tensor_from_numpy(X_val, "X_val"),
            tensor_from_numpy(Y_val, "Y_val"),
        ),
        batch_size=BATCH,
        shuffle=False,
    )

    model = CompensationModel(X_train.shape[2]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=WD, eps=1e-7
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    loss_fn = nn.HuberLoss()
    warmup_state = {
        "step": 0,
        "warmup_steps": WARMUP_STEPS,
        "base_lr": LR,
        "warmup_finished": False,
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    early_stopped = False
    early_stop_epoch = None

    print("Training...", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_skipped, train_batches = run_epoch(
            model, train_loader, optimizer, loss_fn, train=True,
            warmup_state=warmup_state,
        )
        val_loss, _, _ = run_epoch(
            model, val_loader, optimizer, loss_fn, train=False
        )
        check_epoch_losses(
            train_loss, val_loss, train_skipped, train_batches, epoch
        )

        # Apply plateau scheduling only after linear warmup completes.
        if warmup_state["step"] >= warmup_state["warmup_steps"]:
            if not warmup_state["warmup_finished"]:
                for group in optimizer.param_groups:
                    group["lr"] = LR
                warmup_state["warmup_finished"] = True
            scheduler.step(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
            + (" *" if improved else ""),
            flush=True,
        )

        if stale >= EARLY_STOP_PATIENCE:
            early_stopped = True
            early_stop_epoch = epoch
            print(f"  Early stopping at epoch {epoch} (best = {best_epoch})")
            break

    if early_stop_epoch is None:
        early_stop_epoch = len(history)

    if not np.isfinite(best_val) or state_has_nonfinite(best_state):
        print(
            "ERROR: Training finished without a finite validation loss or with "
            "non-finite best weights; refusing to continue.",
            file=sys.stderr,
        )
        sys.exit(1)

    model.load_state_dict(best_state)

    run_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "early_stop_epoch": early_stop_epoch,
        "early_stopped": early_stopped,
        "total_epochs": len(history),
    }

    fold_dir = OUT / "lodo" / f"test_{test_session}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    save_training_loss_plot(
        history,
        run_summary,
        fold_dir / "training_loss.png",
        title_suffix=f"  (test = {test_session})",
    )

    val_base_rmse, val_final_rmse = split_rmse(
        model, val_records, x_scaler, args.stride
    )

    print("Predicting held-out test day...", flush=True)
    test_rows = predict_records(model, test_records, x_scaler, args.stride)
    file_summary = save_force_plots(test_rows, fold_dir / "test")
    if not args.images_only:
        pd.DataFrame(file_summary).to_csv(
            fold_dir / "per_file_metrics.csv", index=False
        )

    if args.save_all:
        save_prediction_csvs(test_rows, fold_dir / "test_csv")
        save_checkpoint(best_state, fold_dir / "model.pt")
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        (fold_dir / "fold_config.json").write_text(
            json.dumps(
                {
                    "test_session": test_session,
                    "val_session": val_session,
                    "train_sessions": train_sessions,
                    "base_theta": theta.tolist(),
                    "use_lpf": use_lpf,
                    "calib_seconds": args.calib_seconds,
                    "stride": args.stride,
                    **run_summary,
                },
                indent=2,
            )
        )

    base_rmse = float(np.mean([r["Base_RMSE"] for r in file_summary]))
    final_rmse = float(np.mean([r["Final_RMSE"] for r in file_summary]))
    base_r2 = float(np.mean([r["Base_R2"] for r in file_summary]))
    final_r2 = float(np.mean([r["Final_R2"] for r in file_summary]))
    improved_files = sum(
        1 for r in file_summary if r["Final_RMSE"] < r["Base_RMSE"]
    )

    print(
        f"\n  FOLD RESULT (test = {test_session})\n"
        f"    val  : base RMSE={val_base_rmse:.4f}  final RMSE={val_final_rmse:.4f}\n"
        f"    test : base RMSE={base_rmse:.4f}  final RMSE={final_rmse:.4f}\n"
        f"    test : base R2  ={base_r2:.4f}  final R2  ={final_r2:.4f}\n"
        f"    files improved by compensation: {improved_files}/{len(file_summary)}",
        flush=True,
    )

    return {
        "Test_Session": test_session,
        "Val_Session": val_session,
        "Train_Sessions": "+".join(train_sessions),
        "N_Test_Files": len(file_summary),
        "Best_Epoch": best_epoch,
        "Val_Base_RMSE": val_base_rmse,
        "Val_Final_RMSE": val_final_rmse,
        "Base_RMSE": base_rmse,
        "Final_RMSE": final_rmse,
        "Base_R2": base_r2,
        "Final_R2": final_r2,
        "Files_Improved": improved_files,
    }


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Leave-One-Day-Out cross-day compensation training"
    )
    p.add_argument(
        "--stride", type=int, default=WINDOW_STRIDE,
        help="Window stride (higher = less RAM/time). Try 10 or 20 on a laptop.",
    )
    p.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max epochs per fold.")
    p.add_argument(
        "--calib-seconds", type=float, default=0.0,
        help=(
            "Seconds at the start of each recording used to calibrate the "
            "per-session force gain, mirroring a real deployment calibration. "
            "0 disables it (no-calibration baseline). The calibration window is "
            "excluded from all reported metrics."
        ),
    )
    p.add_argument(
        "--no-lpf", action="store_true",
        help="Ablation: use a static TANH base model with no dynamics.",
    )
    p.add_argument(
        "--test-sessions", nargs="+", default=None, metavar="SESSION",
        help=f"Run only these folds. Choices: {' '.join(SESSIONS)}",
    )
    outputs = p.add_mutually_exclusive_group()
    outputs.add_argument(
        "--save-all", action="store_true",
        help="Also save model checkpoints, per-file prediction CSVs and history.",
    )
    outputs.add_argument(
        "--images-only", action="store_true",
        help=(
            "Save only PNGs (result plots and the training-loss curve). "
            "Writes no CSV files; metrics are still printed to the console."
        ),
    )
    p.add_argument(
        "--debug-anomaly", action="store_true",
        help="Enable torch autograd anomaly detection for NaN traceback.",
    )
    return p.parse_args()


def main():
    global BATCH

    args = parse_args()
    args.stride = max(1, args.stride)
    BATCH = max(1, args.batch)

    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    test_sessions = args.test_sessions or SESSIONS
    unknown = [s for s in test_sessions if s not in SESSIONS]
    if unknown:
        print(
            f"ERROR: unknown session(s): {', '.join(unknown)}\n"
            f"Choices: {', '.join(SESSIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = [s for s in SESSIONS if not session_files(s)]
    if missing:
        print(
            f"ERROR: no Data*.csv found for session(s): {', '.join(missing)}\n"
            f"Looked under: {DATA.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    print("DEVICE =", DEVICE)
    print("Sessions:", ", ".join(SESSIONS))
    print(f"Folds to run: {', '.join(test_sessions)}")
    print(
        f"stride={args.stride} batch={BATCH} epochs={args.epochs} "
        f"calib_seconds={args.calib_seconds} lpf={not args.no_lpf}"
    )

    results = []
    for test_session in test_sessions:
        # Validation is a different held-out day, rotating deterministically.
        i = SESSIONS.index(test_session)
        val_session = SESSIONS[(i + 1) % len(SESSIONS)]
        results.append(run_fold(test_session, val_session, args))

    df = pd.DataFrame(results)
    if not args.images_only:
        summary_path = OUT / "lodo" / "lodo_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 62}")
    print("LEAVE-ONE-DAY-OUT SUMMARY")
    print(f"{'=' * 62}")
    print(
        df[
            [
                "Test_Session", "Base_RMSE", "Final_RMSE",
                "Base_R2", "Final_R2", "Files_Improved", "N_Test_Files",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nMean across folds: base RMSE={df['Base_RMSE'].mean():.4f} "
        f"(+-{df['Base_RMSE'].std():.4f})  "
        f"final RMSE={df['Final_RMSE'].mean():.4f} "
        f"(+-{df['Final_RMSE'].std():.4f})"
    )

    total_improved = int(df["Files_Improved"].sum())
    total_files = int(df["N_Test_Files"].sum())
    folds_improved = int((df["Final_RMSE"] < df["Base_RMSE"]).sum())
    print(
        f"Folds where compensation beat the base model: "
        f"{folds_improved}/{len(df)}"
    )
    print(f"Files where compensation beat the base model: {total_improved}/{total_files}")

    if folds_improved == 0:
        print(
            "\nGATE FAILED: compensation did not beat the base model on any fold.\n"
            "Do not tune the architecture yet - the scale/gain handling is still "
            "wrong. Try --calib-seconds 10 first."
        )
    print(f"\nSaved to: {(OUT / 'lodo').resolve()}")


if __name__ == "__main__":
    main()
