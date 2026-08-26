"""Compensation-model architectures for cross-day sEMG force estimation.

Every model maps a window of shape ``(batch, SEQ, n_features)`` to a single
scalar: the normalised compensation at the last sample of the window. They are
therefore interchangeable inside the training/evaluation pipeline.

Widths are chosen so the models sit in the same parameter range (roughly
60k-200k), which keeps the comparison about architecture rather than capacity.
With ~10^5 training windows of length 100 and only 2 input channels, models
much larger than this overfit the day-to-day variation rather than the
sEMG-to-force relationship.

A model may expose an auxiliary loss (used by the PINN) by setting
``self._aux`` during ``forward``; the training loop adds it to the data loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ = 100
DROP = 0.2


# =========================================================
# Shared pieces
# =========================================================
class CompensationBase(nn.Module):
    """Common interface: optional physics/auxiliary loss from the last forward."""

    def __init__(self):
        super().__init__()
        self._aux = None

    def aux_loss(self):
        return self._aux


def make_head(d_in, drop=DROP):
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Dropout(drop),
        nn.Linear(d_in, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ):
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
        return x + self.pe[:, : x.size(1)]


def conv_stem(n_features, channels, kernel=3):
    return nn.Sequential(
        nn.Conv1d(n_features, channels, kernel, padding="same"),
        nn.GroupNorm(8, channels),
        nn.GELU(),
        nn.Conv1d(channels, channels, kernel, padding="same"),
        nn.GroupNorm(8, channels),
        nn.GELU(),
    )


# =========================================================
# 1. Conv + BiLSTM + Transformer  (the original model)
# =========================================================
class ConvBiLSTMTransformer(CompensationBase):
    def __init__(self, n_features, conv_c=32, lstm_h=48, d_model=48,
                 heads=4, layers=2, ff=96):
        super().__init__()
        self.conv = conv_stem(n_features, conv_c)
        self.lstm = nn.LSTM(conv_c, lstm_h, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * lstm_h, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff, dropout=DROP,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = make_head(d_model)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================================================
# 2. Conv + BiLSTM  (no attention)
# =========================================================
class ConvBiLSTM(CompensationBase):
    def __init__(self, n_features, conv_c=32, lstm_h=64, layers=2):
        super().__init__()
        self.conv = conv_stem(n_features, conv_c)
        self.lstm = nn.LSTM(
            conv_c, lstm_h, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=DROP if layers > 1 else 0.0,
        )
        self.head = make_head(2 * lstm_h)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================================================
# 3. Plain LSTM  (stacked, unidirectional - causal baseline)
# =========================================================
class LSTMNet(CompensationBase):
    def __init__(self, n_features, hidden=96, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=DROP if layers > 1 else 0.0,
        )
        self.head = make_head(hidden)

    def forward(self, x):
        x, _ = self.lstm(x)
        # Causal model: the last state summarises the window.
        return self.head(x[:, -1]).squeeze(-1)


# =========================================================
# 4. Transformer encoder only
# =========================================================
class TransformerNet(CompensationBase):
    def __init__(self, n_features, d_model=64, heads=4, layers=3, ff=128):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff, dropout=DROP,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = make_head(d_model)

    def forward(self, x):
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================================================
# 5. Synergy-SSM network
#
# A synergy layer projects the inputs onto a small set of non-negative
# activation primitives, and each primitive drives a diagonal linear
# state-space model. This mirrors the muscle-synergy view of motor control:
# a few low-dimensional activations, each low-pass filtered by muscle
# activation dynamics.
#
# NOTE: classical muscle-synergy decomposition needs *many* EMG channels.
# This dataset has a single sEMG channel (plus the base-model output), so the
# synergy layer here is a low-rank non-negative bottleneck rather than a true
# multi-muscle synergy extraction. Interpret it accordingly.
# =========================================================
class SynergySSM(CompensationBase):
    def __init__(self, n_features, n_synergies=16, d_state=128, d_out=128):
        super().__init__()
        self.synergy = nn.Linear(n_features, n_synergies)
        self.to_state = nn.Linear(n_synergies, d_state, bias=False)
        # Per-state decay in (0, 1) via sigmoid: first-order activation dynamics.
        # The spread of initial time constants lets different states capture
        # fast twitch response and slow force build-up.
        self.decay_logit = nn.Parameter(torch.linspace(-2.0, 3.0, d_state))
        self.mix = nn.Sequential(
            nn.Linear(d_state, 2 * d_out),
            nn.GELU(),
            nn.Linear(2 * d_out, d_out),
        )
        self.head = make_head(d_out)

    def forward(self, x):
        # Non-negative synergy activations.
        u = F.softplus(self.synergy(x))          # (B, T, K)
        v = self.to_state(u)                     # (B, T, S)

        a = torch.sigmoid(self.decay_logit)      # (S,)
        h = torch.zeros(v.size(0), v.size(2), device=v.device, dtype=v.dtype)
        states = []
        for t in range(v.size(1)):
            h = a * h + (1.0 - a) * v[:, t]
            states.append(h)
        h_seq = torch.stack(states, dim=1)       # (B, T, S)

        z = self.mix(h_seq)
        return self.head(z.mean(dim=1)).squeeze(-1)


# =========================================================
# 6. Mamba-style selective structured SSM (S6)
#
# Diagonal SSM with input-dependent step size, B and C, plus the short causal
# convolution and gated residual branch of a Mamba block. This is a compact
# PyTorch reimplementation - it does not use the fused CUDA `mamba-ssm`
# kernels, so it is correct but slower than the reference implementation.
# =========================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner
        )
        self.x_proj = nn.Linear(d_inner, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, d_inner)

        A = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Short causal depthwise conv.
        x_in = self.conv(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_in = F.silu(x_in)

        proj = self.x_proj(x_in)
        dt_raw, B_t, C_t = torch.split(
            proj, [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))    # (B, T, d_inner)

        A = -torch.exp(self.A_log)               # (d_inner, d_state)

        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        states = []
        for t in range(T):
            dt_t = dt[:, t].unsqueeze(-1)                  # (B, d_inner, 1)
            a_bar = torch.exp(dt_t * A)                    # (B, d_inner, S)
            u_t = x_in[:, t].unsqueeze(-1)                 # (B, d_inner, 1)
            b_t = B_t[:, t].unsqueeze(1)                   # (B, 1, S)
            h = a_bar * h + (dt_t * b_t) * u_t
            states.append(h)
        h_seq = torch.stack(states, dim=1)                 # (B, T, d_inner, S)

        y = (h_seq * C_t.unsqueeze(2)).sum(-1) + self.D * x_in
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class MambaSSM(CompensationBase):
    # One block by default: each block runs a 100-step sequential scan, which is
    # the dominant cost without the fused CUDA kernels. A second block roughly
    # doubles training time for a small accuracy change on this dataset.
    def __init__(self, n_features, d_model=64, d_state=8, layers=1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = make_head(d_model)

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================================================
# 7. Physics-Informed Neural Network
#
# The network predicts the compensation at every timestep of the window. On top
# of the usual data loss at the last sample, a physics residual penalises
# departure from the first-order activation dynamics the EDM base model assumes:
#
#     tau * d(f)/dt + f  =  w * sEMG + b
#
# where f is the reconstructed (normalised) force and tau, w, b are learnable.
# Inputs arrive standard-scaled, so the constraint holds up to an affine
# transform, which w and b absorb.
# =========================================================
class PINN(CompensationBase):
    def __init__(self, n_features, hidden=64, tau_init=20.0, aux_weight=1e-2):
        super().__init__()
        self.aux_weight = aux_weight
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, 5, padding="same"),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding="same"),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            hidden, hidden // 2, batch_first=True, bidirectional=True
        )
        self.out = nn.Linear(hidden, 1)

        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        seq = self.net(x.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.gru(seq)
        comp = self.out(seq).squeeze(-1)          # (B, T) compensation per step

        # Reconstructed normalised force: base output channel + compensation.
        f = x[:, :, 1] + comp

        tau = torch.exp(self.log_tau)
        df = f[:, 1:] - f[:, :-1]
        drive = self.w * x[:, 1:, 0] + self.b
        residual = tau * df + f[:, 1:] - drive
        self._aux = self.aux_weight * (residual ** 2).mean()

        return comp[:, -1]


# =========================================================
# Registry
# =========================================================
MODEL_REGISTRY = {
    "conv_bilstm_transformer": ConvBiLSTMTransformer,
    "conv_bilstm": ConvBiLSTM,
    "lstm": LSTMNet,
    "transformer": TransformerNet,
    "synergy_ssm": SynergySSM,
    "mamba": MambaSSM,
    "pinn": PINN,
}

MODEL_NAMES = list(MODEL_REGISTRY)

DISPLAY_NAMES = {
    "conv_bilstm_transformer": "Conv-BiLSTM-Transformer",
    "conv_bilstm": "Conv-BiLSTM",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "synergy_ssm": "Synergy-SSM",
    "mamba": "Mamba (SSM)",
    "pinn": "PINN",
}


def build_model(name, n_features):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choices: {', '.join(MODEL_NAMES)}"
        )
    return MODEL_REGISTRY[name](n_features)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
