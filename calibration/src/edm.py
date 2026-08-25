"""Frozen EDM forward model and parameter I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from src.config import AppConfig


@dataclass(frozen=True)
class EDMParams:
    """Frozen EDM parameters b1..b6."""

    b1: float
    b2: float
    b3: float
    b4: float
    b5: float
    b6: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.b1, self.b2, self.b3, self.b4, self.b5, self.b6], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "EDMParams":
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "EDMParams":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**raw)

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def assert_frozen(params: EDMParams, expected_sha256: str) -> None:
    """Raise if loaded parameters differ from the committed hash."""
    if params.sha256() != expected_sha256:
        raise ValueError(
            f"EDM parameters changed. expected={expected_sha256} got={params.sha256()}"
        )


def edm_forward(envelope: np.ndarray, params: EDMParams, fs: float) -> np.ndarray:
    """Discrete-time EDM: delay -> offset -> tanh -> scale -> first-order lag."""
    x = np.asarray(envelope, dtype=float)
    delay = int(round(params.b3 * fs))
    if delay > 0:
        x = np.concatenate([np.full(delay, x[0]), x[:-delay]])

    z = params.b2 * (x - params.b4)
    y = params.b1 * np.tanh(z)

    # First-order lag: b5 / (s + b6) via bilinear-like one-pole recursion
    dt = 1.0 / fs
    alpha = dt * params.b6 / (1.0 + dt * params.b6)
    gain = params.b5 * dt / (1.0 + dt * params.b6)
    out = np.empty_like(y)
    out[0] = gain * y[0]
    for i in range(1, len(y)):
        out[i] = (1.0 - alpha) * out[i - 1] + gain * y[i]
    return out


def fit_vector_bounds(cfg: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    b = cfg.edm_identify.bounds
    names = ["b1", "b2", "b3", "b4", "b5", "b6"]
    lower = np.array([b[n][0] for n in names])
    upper = np.array([b[n][1] for n in names])
    return lower, upper
