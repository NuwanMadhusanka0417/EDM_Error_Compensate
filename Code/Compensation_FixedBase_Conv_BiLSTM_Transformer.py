from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.signal import lfilter
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# Settings
# =========================================================
SEED = 42
Ts = 0.0005

SEQ = 100
BATCH = 64
EPOCHS = 1

CONV_C = 32
KERNEL = 3
LSTM_H = 48

D_MODEL = 48
HEADS = 4
TF_LAYERS = 2
FF = 96

DROP = 0.2
LR = 1e-3
WD = 1e-4

# Pattern
# 1: Preprocessed sEMG only
# 2: Base model estimation only
# 3: Preprocessed sEMG + Base model estimation
PATTERNS = [2]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("DEVICE =", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Files
# =========================================================
data_dir = Path("../Data")

# ---------------------------------------------------------
# Base model fitting data
# Fixed structure: TANH -> LPF
# Adjust the files here as needed.
# ---------------------------------------------------------
base_files = [
    data_dir / f"3rd_Base/Data{i}.csv"
    for i in [1]
]

# ---------------------------------------------------------
# Compensation training data
# ---------------------------------------------------------
comp_day1_files = [
    data_dir / f"1st_Comp/Data{i}.csv"
    # for i in range (1, 6)
    for i in [1, 3]
]

comp_day2_files = [
    data_dir / f"2nd_Comp/Data{i}.csv"
    # for i in range(1, 10)
    for i in [7]
]

comp_day3_files = [
    data_dir / f"5th_Comp/Data{i}.csv"
    # for i in range(1, 11)
    for i in [4, 8]
]

# comp_files = comp_day1_files + comp_day2_files + comp_day3_files
comp_files = comp_day1_files + comp_day3_files

# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------
test_files = [
    data_dir / f"4th_Valid/Data{i}.csv"
    for i in range(1, 14)
]

out = data_dir.parent / "Estimation"
out.mkdir(parents=True, exist_ok=True)

XCOL = "Cali_LPF_PASF_sEMG"
YCOL = "Force"
TCOL = "Time"

FC_MIN = 10
FC_MAX = 500.0


# =========================================================
# Common
# =========================================================
def load(path):
    df = pd.read_csv(path)

    values = df[[XCOL, YCOL]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1)

    x = values.loc[valid, XCOL].to_numpy(float)
    y = values.loc[valid, YCOL].to_numpy(float)

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
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y, y_hat),
        "R2": r2_score(y, y_hat),
    }


# =========================================================
# Fixed Base Model
# TANH -> LPF
#
# TANH: z = gain * tanh(slope * x)
# LPF : G(s) = wc / (s + wc), wc = 2*pi*fc
#
# Only [gain, slope, fc] are fitted once.
# After fitting, BASE_THETA is frozen.
# =========================================================
def lpf(x, fc):
    K = 2.0 / Ts
    fc = np.clip(float(fc), FC_MIN, FC_MAX)
    wc = 2.0 * np.pi * fc

    b0 = wc / (K + wc)
    a1 = (wc - K) / (K + wc)

    return lfilter([b0, b0], [1.0, a1], x)

def base_model(theta, x):
    gain, slope = theta
    return gain * np.tanh(slope * x)
# def base_model(theta, x):
#     gain, slope, fc = theta
#     z = gain * np.tanh(slope * x)
#     return lpf(z, fc)


# =========================================================
# 1. Base model fitting, then freeze
# =========================================================
base_data = [load(path) for path in base_files]


def base_residual(theta):
    residuals = []
    for _, x, y in base_data:
        residuals.append(base_model(theta, x) - y)
    return np.concatenate(residuals)

BASE_X0 = np.array([1.0, 1.0])
BASE_LOWER = np.array([0, 0])
BASE_UPPER = np.array([10.0, 10.0])
# BASE_X0 = np.array([1.0, 1.0, 10.0])
# BASE_LOWER = np.array([0.0, 0.0, FC_MIN])
# BASE_UPPER = np.array([100.0, 100.0, FC_MAX])

print("\n===== Base model fitting: TANH -> LPF =====")

base_result = least_squares(
    base_residual,
    BASE_X0,
    bounds=(BASE_LOWER, BASE_UPPER),
    method="trf",
    x_scale="jac",
    loss="linear",
    max_nfev=2000,
)

BASE_THETA = base_result.x.copy()

print("Frozen Base Model")
print("TANH gain  =", BASE_THETA[0])
print("TANH slope =", BASE_THETA[1])
# print("LPF fc     =", BASE_THETA[2], "Hz")

pd.DataFrame([
    {
        "Order": 1,
        "Function": "TANH",
        "Parameter1": BASE_THETA[0],
        "Parameter2": BASE_THETA[1],
        "Unit": "",
    },
    # {
    #     "Order": 2,
    #     "Function": "LPF",
    #     "Parameter1": BASE_THETA[2],
    #     "Parameter2": np.nan,
    #     "Unit": "Hz",
    # },
]).to_csv(out / "base_model_parameters.csv", index=False)

pd.DataFrame([{
    "Success": base_result.success,
    "Status": base_result.status,
    "Cost": base_result.cost,
    "Optimality": base_result.optimality,
    "NFEV": base_result.nfev,
    "Message": base_result.message,
}]).to_csv(out / "base_optimizer_info.csv", index=False)


# =========================================================
# 2. Prepare compensation data with frozen Base model
# =========================================================
comp_data = []

for path in comp_files:
    t, x, y = load(path)
    y_base = base_model(BASE_THETA, x)

    comp_data.append({
        "path": path,
        "t": t,
        "x": x,
        "y": y,
        "y_base": y_base,
        "error": y - y_base,
    })


# =========================================================
# Input patterns
# =========================================================
def features(x, y_base, pattern):
    if pattern == 1:
        return x[:, None]

    if pattern == 2:
        return y_base[:, None]

    if pattern == 3:
        return np.column_stack([x, y_base])

    raise ValueError("pattern must be 1, 2, or 3")


# =========================================================
# Sliding windows
# =========================================================
def windows(x, y):
    index = np.arange(SEQ - 1, len(x))

    X = np.stack([
        x[k - SEQ + 1:k + 1]
        for k in index
    ]).astype(np.float32)

    Y = y[index].astype(np.float32)

    return X, Y, index


# =========================================================
# Positional encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================================================
# Compensation Model
# Conv + BiLSTM + Transformer
# =========================================================
class CompensationModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, CONV_C, KERNEL, padding="same"),
            nn.BatchNorm1d(CONV_C),
            nn.GELU(),
            nn.Conv1d(CONV_C, CONV_C, KERNEL, padding="same"),
            nn.BatchNorm1d(CONV_C),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            CONV_C,
            LSTM_H,
            batch_first=True,
            bidirectional=True,
        )

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
            layer,
            num_layers=TF_LAYERS,
            enable_nested_tensor=False,
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


# =========================================================
# 3. Prepare compensation training data
# =========================================================
def prepare(pattern):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(np.vstack([
        features(d["x"], d["y_base"], pattern)
        for d in comp_data
    ]))

    y_scaler.fit(np.concatenate([
        d["error"]
        for d in comp_data
    ])[:, None])

    X_list = []
    Y_list = []

    for d in comp_data:
        feature_scaled = x_scaler.transform(
            features(d["x"], d["y_base"], pattern)
        )

        error_scaled = y_scaler.transform(
            d["error"][:, None]
        ).ravel()

        X, Y, _ = windows(feature_scaled, error_scaled)

        X_list.append(X)
        Y_list.append(Y)

    return (
        np.concatenate(X_list),
        np.concatenate(Y_list),
        x_scaler,
        y_scaler,
    )


# =========================================================
# Train compensation model
# =========================================================
def train_model(X, Y):
    model = CompensationModel(X.shape[2]).to(DEVICE)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    loss_fn = nn.HuberLoss()

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for batch, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(xb)
            loss = loss_fn(prediction, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if batch == 1 or batch % 10 == 0 or batch == len(loader):
                print(
                    f"  Batch {batch:4d}/{len(loader):4d} | "
                    f"Loss: {loss.item():.6f}"
                )

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Mean Loss: {np.mean(losses):.6f}"
        )

    return model


# =========================================================
# 4. Train each input pattern
# =========================================================
models = {}

for pattern in PATTERNS:
    print(f"\n===== Pattern {pattern} =====")

    X, Y, xs, ys = prepare(pattern)

    print("Training data:", X.shape, Y.shape)

    model = train_model(X, Y)

    models[pattern] = (model, xs, ys)

    save_dir = out / f"Pattern{pattern}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model.pt")


# =========================================================
# 5. Test inference with frozen Base model
# =========================================================
summary = []

for path in test_files:
    t, x, y = load(path)

    y_base = base_model(BASE_THETA, x)

    for pattern in PATTERNS:
        model, xs, ys = models[pattern]

        feature = xs.transform(
            features(x, y_base, pattern)
        )

        X, _, index = windows(
            feature,
            np.zeros(len(feature)),
        )

        model.eval()
        pred_scaled_parts = []

        with torch.no_grad():
            for i in range(0, len(X), BATCH):
                xb = torch.tensor(
                    X[i:i + BATCH],
                    dtype=torch.float32,
                ).to(DEVICE)

                pred_scaled_parts.append(
                    model(xb).cpu().numpy()
                )

        pred_scaled = np.concatenate(pred_scaled_parts)

        compensation = ys.inverse_transform(
            pred_scaled[:, None]
        ).ravel()

        y_true = y[index]
        y_base_eval = y_base[index]
        y_hat = y_base_eval + compensation

        base_metrics = metrics(y_true, y_base_eval)
        final_metrics = metrics(y_true, y_hat)

        save_dir = out / f"Pattern{pattern}" / path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            TCOL: t[index],
            XCOL: x[index],
            YCOL: y_true,
            "BaseModel_Output": y_base_eval,
            "Compensation_Output": compensation,
            "Final_Output": y_hat,
            "BaseModel_Error": y_true - y_base_eval,
            "Final_Error": y_true - y_hat,
        }).to_csv(
            save_dir / "prediction.csv",
            index=False,
        )

        result = {
            "Model": "Conv_BiLSTM_Transformer",
            "Pattern": pattern,
            "Data": path.stem,
            **{f"Base_{k}": v for k, v in base_metrics.items()},
            **{f"Final_{k}": v for k, v in final_metrics.items()},
        }

        pd.DataFrame([result]).to_csv(
            save_dir / "metrics.csv",
            index=False,
        )

        summary.append(result)


# =========================================================
# Summary
# =========================================================
pd.DataFrame(summary).to_csv(
    out / "test_metrics_summary.csv",
    index=False,
)

print(f"\nSaved to:\n{out}")