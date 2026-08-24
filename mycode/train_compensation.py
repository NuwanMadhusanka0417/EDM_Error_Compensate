"""Compensation model training with early stopping and Pattern 3 inputs.

Train : 1st_Comp + 2nd_Comp + 3rd_Base
Val   : 4th_Valid
Test  : 5th_Comp
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# Settings
# =========================================================
SEED = 42
Ts = 0.0005
PATTERN = 3

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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent / "Estimation"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def csv_files(folder):
    return sorted((DATA / folder).glob("Data*.csv"))


TRAIN_FILES = (
    csv_files("1st_Comp")
    + csv_files("2nd_Comp")
    + csv_files("3rd_Base")
)
VAL_FILES = csv_files("4th_Valid")
TEST_FILES = csv_files("5th_Comp")
BASE_FILES = csv_files("3rd_Base")


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
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y, y_hat),
        "R2": r2_score(y, y_hat),
    }


def base_model(theta, x):
    gain, slope = theta
    return gain * np.tanh(slope * x)


def features(x, y_base):
    return np.column_stack([x, y_base])


def windows(x, y):
    if len(x) < SEQ:
        return (
            np.empty((0, SEQ, x.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.array([], dtype=int),
        )
    index = np.arange(SEQ - 1, len(x), WINDOW_STRIDE)
    X_all = np.lib.stride_tricks.sliding_window_view(x, SEQ, axis=0)
    X_all = np.transpose(X_all, (0, 2, 1))
    win_idx = index - (SEQ - 1)
    X = X_all[win_idx].astype(np.float32)
    Y = y[index].astype(np.float32)
    return X, Y, index


def semg_scale_from_files(paths, q=99.0):
    vals = []
    for path in paths:
        _, x, _ = load(path)
        vals.append(x)
    all_x = np.concatenate(vals)
    positive = all_x[all_x > 0]
    if len(positive) == 0:
        return float(np.max(np.abs(all_x)) or 1.0)
    return float(np.percentile(positive, q))


def fit_base_model(paths, scale):
    data = [load(p) for p in paths]

    def residual(theta):
        out = []
        for _, x, y in data:
            out.append(base_model(theta, x / scale) - y)
        return np.concatenate(out)

    result = least_squares(
        residual,
        x0=np.array([1.0, 1.0]),
        bounds=([0.0, 0.0], [20.0, 20.0]),
        method="trf",
        max_nfev=3000,
    )
    return result.x.copy(), result


def build_records(paths, base_theta, scale):
    records = []
    for path in paths:
        t, x, y = load(path)
        x_n = x / scale
        y_base = base_model(base_theta, x_n)
        records.append(
            {
                "path": path,
                "t": t,
                "x": x,
                "x_n": x_n,
                "y": y,
                "y_base": y_base,
                "error": y - y_base,
            }
        )
    return records


def fix_scaler_scale(scaler):
    """Guard: zero-variance features must not divide by zero during transform."""
    scale = np.asarray(scaler.scale_, dtype=float)
    scale[np.abs(scale) < 1e-8] = 1.0
    scaler.scale_ = scale


def fit_scalers(records):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(
        np.vstack([features(r["x_n"], r["y_base"]) for r in records])
    )
    y_scaler.fit(
        np.concatenate([r["error"] for r in records])[:, None]
    )
    fix_scaler_scale(x_scaler)
    fix_scaler_scale(y_scaler)
    return x_scaler, y_scaler


def records_to_xy(records, x_scaler, y_scaler, label=""):
    X_parts, Y_parts = [], []
    for r in records:
        feat = x_scaler.transform(features(r["x_n"], r["y_base"]))
        err = y_scaler.transform(r["error"][:, None]).ravel()
        X, Y, _ = windows(feat, err)
        # Guard: scaled windows must stay finite before training.
        assert_finite(X, f"{label or r['path']} features (windows)")
        assert_finite(Y, f"{label or r['path']} targets (windows)")
        X_parts.append(X)
        Y_parts.append(Y)
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


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    train=True,
    log_every=0,
    warmup_state=None,
):
    model.train(mode=train)
    losses = []
    skipped = 0
    total_batches = len(loader)

    for batch_i, (xb, yb) in enumerate(loader, start=1):
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
        if log_every and batch_i % log_every == 0:
            print(
                f"    batch {batch_i}/{total_batches} loss={loss.item():.6f}",
                flush=True,
            )

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
def eval_split_loss_and_rmse(model, records, x_scaler, y_scaler, loss_fn):
    """Compute mean scaled loss and physical RMSE over a split (no file IO)."""
    losses = []
    rmses = []

    for r in records:
        feat = x_scaler.transform(features(r["x_n"], r["y_base"]))
        err_scaled = y_scaler.transform(r["error"][:, None]).ravel()
        X, Y, index = windows(feat, err_scaled)
        if len(X) == 0:
            continue

        assert_finite(X, f"{r['path']} eval features (windows)")
        assert_finite(Y, f"{r['path']} eval targets (windows)")

        pred_parts = []
        for i in range(0, len(X), BATCH):
            xb = tensor_from_numpy(X[i : i + BATCH], f"{r['path']} eval batch")
            xb = xb.to(DEVICE)
            pred_parts.append(model(xb).cpu())
        pred_scaled = torch.cat(pred_parts)

        target = torch.tensor(Y, dtype=torch.float32)
        split_loss = loss_fn(pred_scaled, target).item()
        if np.isfinite(split_loss):
            losses.append(split_loss)

        compensation = y_scaler.inverse_transform(pred_scaled.numpy()[:, None]).ravel()
        y_true = r["y"][index]
        y_hat = r["y_base"][index] + compensation
        rmses.append(float(np.sqrt(mean_squared_error(y_true, y_hat))))

    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(rmses))


@torch.no_grad()
def predict_compensation(model, records, x_scaler, y_scaler, scale):
    model.eval()
    rows = []

    for r in records:
        feat = x_scaler.transform(features(r["x_n"], r["y_base"]))
        X, _, index = windows(feat, np.zeros(len(feat)))
        # Guard: inference features must remain finite after scaling/windowing.
        assert_finite(X, f"{r['path']} inference features (windows)")

        pred_parts = []
        for i in range(0, len(X), BATCH):
            xb = tensor_from_numpy(X[i : i + BATCH], f"{r['path']} inference batch")
            xb = xb.to(DEVICE)
            pred_parts.append(model(xb).cpu().numpy())
        pred_scaled = np.concatenate(pred_parts)

        compensation = y_scaler.inverse_transform(pred_scaled[:, None]).ravel()
        y_true = r["y"][index]
        y_base = r["y_base"][index]
        y_hat = y_base + compensation

        rows.append(
            {
                "path": r["path"],
                "index": index,
                "t": r["t"][index],
                "x": r["x"][index],
                "y": y_true,
                "y_base": y_base,
                "compensation": compensation,
                "y_hat": y_hat,
                "base_metrics": metrics(y_true, y_base),
                "final_metrics": metrics(y_true, y_hat),
            }
        )
    return rows


def save_training_loss_plot(history, run_summary, out_path):
    """Always save train/val/test loss curves with best and early-stop epochs."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, [h["train_loss"] for h in history], label="Train loss", linewidth=1.5)
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val loss", linewidth=1.5)
    ax.plot(epochs, [h["test_loss"] for h in history], label="Test loss", linewidth=1.5)

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
    ax.set_ylabel("Huber loss (scaled)")
    ax.set_title("Training / validation / test loss per epoch")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_test_force_plots(rows, out_dir):
    """Always save real vs predicted force line plots with metrics at the bottom."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for row in rows:
        stem = row["path"].stem
        t = row["t"]
        y_true = row["y"]
        y_hat = row["y_hat"]
        y_base = row["y_base"]
        final_m = metrics(y_true, y_hat)
        base_m = metrics(y_true, y_base)

        step = max(1, len(t) // 5000)
        sl = slice(None, None, step)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t[sl], y_true[sl], label="Real Force", linewidth=1.0)
        ax.plot(t[sl], y_hat[sl], label="Predicted Force", linewidth=1.0, alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force")
        ax.set_title(f"{stem}: Real vs Predicted Force")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        metric_text = (
            f"Final  RMSE={final_m['RMSE']:.4f}  MAE={final_m['MAE']:.4f}  "
            f"R2={final_m['R2']:.4f}  MSE={final_m['MSE']:.4f}     "
            f"Base   RMSE={base_m['RMSE']:.4f}  MAE={base_m['MAE']:.4f}  "
            f"R2={base_m['R2']:.4f}  MSE={base_m['MSE']:.4f}"
        )
        fig.text(0.5, 0.02, metric_text, ha="center", va="bottom", fontsize=9)
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

        summary.append({"Data": stem, **final_m})

    return summary


def save_test_predictions_csv(rows, out_dir):
    """CSV export for test predictions (only used with --save-all)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for row in rows:
        stem = row["path"].stem
        y_true = row["y"]
        y_hat = row["y_hat"]

        pd.DataFrame(
            {
                TCOL: row["t"],
                "Force": y_true,
                "Predicted_Force": y_hat,
            }
        ).to_csv(out_dir / f"{stem}.csv", index=False)

        summary.append(
            {
                "Data": stem,
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_hat))),
                "MAE": float(mean_absolute_error(y_true, y_hat)),
                "R2": float(r2_score(y_true, y_hat)),
            }
        )

    pd.DataFrame(summary).to_csv(out_dir / "metrics_summary.csv", index=False)
    return summary


def save_predictions(rows, split_dir):
    summary = []
    for row in rows:
        stem = row["path"].stem
        save_dir = split_dir / stem
        save_dir.mkdir(parents=True, exist_ok=True)

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
        ).to_csv(save_dir / "prediction.csv", index=False)

        result = {
            "Data": stem,
            **{f"Base_{k}": v for k, v in row["base_metrics"].items()},
            **{f"Final_{k}": v for k, v in row["final_metrics"].items()},
        }
        pd.DataFrame([result]).to_csv(save_dir / "metrics.csv", index=False)
        summary.append(result)

    pd.DataFrame(summary).to_csv(split_dir / "metrics_summary.csv", index=False)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train compensation model")
    p.add_argument(
        "--stride",
        type=int,
        default=WINDOW_STRIDE,
        help="Window stride (higher = less RAM). Try 10 for ~4GB, 20 for ~2GB.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="Batch size (lower = less RAM). Try 32 or 16 on small machines.",
    )
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument(
        "--debug-anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection for NaN traceback.",
    )
    p.add_argument(
        "--save-all",
        action="store_true",
        help=(
            "Also save CSV outputs (training history, run summary, model, "
            "config, validation CSVs, detailed prediction CSVs)."
        ),
    )
    return p.parse_args()


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


def main():
    global WINDOW_STRIDE, BATCH, MAX_EPOCHS

    args = parse_args()
    WINDOW_STRIDE = max(1, args.stride)
    BATCH = max(1, args.batch)
    MAX_EPOCHS = args.epochs

    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    OUT.mkdir(parents=True, exist_ok=True)
    print("DEVICE =", DEVICE, flush=True)
    print(
        f"Train files: {len(TRAIN_FILES)} | Val: {len(VAL_FILES)} | Test: {len(TEST_FILES)}",
        flush=True,
    )
    print(
        f"Low-RAM settings: stride={WINDOW_STRIDE}, batch={BATCH}",
        flush=True,
    )

    # Use training statistics only; do not leak validation into sEMG scale.
    semg_scale = semg_scale_from_files(TRAIN_FILES)
    base_theta, base_result = fit_base_model(BASE_FILES, semg_scale)

    print("\n===== Base model (all 3rd_Base files) =====")
    print(f"sEMG scale = {semg_scale:.4f}")
    print(f"gain  = {base_theta[0]:.4f}")
    print(f"slope = {base_theta[1]:.4f}")

    train_records = build_records(TRAIN_FILES, base_theta, semg_scale)
    val_records = build_records(VAL_FILES, base_theta, semg_scale)
    test_records = build_records(TEST_FILES, base_theta, semg_scale)

    print("Fitting scalers on training data...", flush=True)
    x_scaler, y_scaler = fit_scalers(train_records)
    print("Building windows...", flush=True)
    X_train, Y_train = records_to_xy(train_records, x_scaler, y_scaler, "train")
    X_val, Y_val = records_to_xy(val_records, x_scaler, y_scaler, "val")
    print(f"Pattern {PATTERN} ready", flush=True)

    X_train_t = tensor_from_numpy(X_train, "X_train")
    Y_train_t = tensor_from_numpy(Y_train, "Y_train")
    X_val_t = tensor_from_numpy(X_val, "X_val")
    Y_val_t = tensor_from_numpy(Y_val, "Y_val")

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,  # Avoid batch-size-1 BatchNorm/GroupNorm edge cases.
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
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

    print("\n===== Training =====")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_skipped, train_batches = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            train=True,
            log_every=500,
            warmup_state=warmup_state,
        )
        val_loss, val_skipped, val_batches = run_epoch(
            model, val_loader, optimizer, loss_fn, train=False
        )
        val_rmse = eval_split_loss_and_rmse(
            model, val_records, x_scaler, y_scaler, loss_fn
        )[1]
        test_loss, test_rmse = eval_split_loss_and_rmse(
            model, test_records, x_scaler, y_scaler, loss_fn
        )

        if train_skipped:
            print(
                f"  train skipped batches: {train_skipped}/{train_batches}",
                flush=True,
            )
        if val_skipped:
            print(
                f"  val skipped batches: {val_skipped}/{val_batches}",
                flush=True,
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
                "test_loss": test_loss,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "train_skipped": train_skipped,
                "val_skipped": val_skipped,
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )

        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} val={val_loss:.6f} test={test_loss:.6f} "
            f"val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            + (" *" if improved else ""),
            flush=True,
        )

        if stale >= EARLY_STOP_PATIENCE:
            early_stopped = True
            early_stop_epoch = epoch
            print(f"Early stopping at epoch {epoch} (best epoch = {best_epoch})")
            break

    total_epochs = len(history)
    if early_stop_epoch is None:
        early_stop_epoch = total_epochs

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
        "total_epochs": total_epochs,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOP_PATIENCE,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_test_loss": history[-1]["test_loss"],
        "final_val_rmse": history[-1]["val_rmse"],
        "final_test_rmse": history[-1]["test_rmse"],
        "best_val_rmse": history[best_epoch - 1]["val_rmse"],
        "best_test_rmse": history[best_epoch - 1]["test_rmse"],
    }

    loss_plot_path = OUT / "training_loss.png"
    save_training_loss_plot(history, run_summary, loss_plot_path)

    print("\n===== Test =====")
    test_rows = predict_compensation(
        model, test_records, x_scaler, y_scaler, semg_scale
    )
    test_plot_dir = OUT / "test"
    test_summary = save_test_force_plots(test_rows, test_plot_dir)

    print(f"Best epoch: {best_epoch}")
    print(
        f"Early stopping: {'yes' if early_stopped else 'no'} "
        f"(stopped at epoch {early_stop_epoch})"
    )
    print(f"Saved loss plot         -> {loss_plot_path}")
    print(f"Saved test force plots  -> {test_plot_dir}")

    if args.save_all:
        history_df = pd.DataFrame(history)
        history_df.to_csv(OUT / "training_history.csv", index=False)
        (OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

        save_checkpoint(best_state, OUT / "model.pt")

        pd.DataFrame(
            [
                {
                    "Order": 1,
                    "Function": "TANH",
                    "Parameter1": base_theta[0],
                    "Parameter2": base_theta[1],
                    "sEMG_Scale": semg_scale,
                }
            ]
        ).to_csv(OUT / "base_model_parameters.csv", index=False)

        meta = {
            "pattern": PATTERN,
            "semg_scale": semg_scale,
            "base_theta": base_theta.tolist(),
            "x_scaler_mean": x_scaler.mean_.tolist(),
            "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": float(y_scaler.mean_[0]),
            "y_scaler_scale": float(y_scaler.scale_[0]),
            **run_summary,
        }
        (OUT / "run_config.json").write_text(json.dumps(meta, indent=2))

        print("\n===== Validation (full save) =====")
        val_summary = save_predictions(
            predict_compensation(
                model, val_records, x_scaler, y_scaler, semg_scale
            ),
            OUT / "validation",
        )
        pd.DataFrame(val_summary).to_csv(
            OUT / "validation_summary.csv", index=False
        )

        print("\n===== Test CSVs (--save-all) =====")
        save_test_predictions_csv(test_rows, OUT / "test_csv")
        pd.DataFrame(test_summary).to_csv(
            OUT / "test_csv_summary.csv", index=False
        )
        print(f"Saved CSV history       -> {OUT / 'training_history.csv'}")
        print(f"Saved run summary       -> {OUT / 'run_summary.json'}")

    print("\nTest mean RMSE:", np.mean([r["RMSE"] for r in test_summary]))
    print(f"\nSaved to:\n{OUT.resolve()}")


if __name__ == "__main__":
    main()
