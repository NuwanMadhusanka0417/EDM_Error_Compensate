"""sEMG and force preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

from src.config import PreprocessConfig


def comb_filter(x: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Digital comb filter for power-line interference removal."""
    n = int(round(fs / cfg.comb_f0))
    n = max(n, 2)
    b = cfg.comb_c1 * np.array([1.0] + [0.0] * (n - 1) + [-1.0])
    a = np.array([1.0] + [0.0] * (n - 1) + [-cfg.comb_c2])
    return lfilter(b, a, x)


def pasf_filter(x: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Simple low-pass proxy for periodic/aperiodic separation (PASF)."""
    wc = 2.0 * np.pi * cfg.pasf_cutoff_hz / fs
    alpha = wc / (wc + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def bandpass(x: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Band-pass filter using zero-phase Butterworth."""
    lo, hi = cfg.bandpass_hz
    nyq = fs / 2.0
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Second-order low-pass filter."""
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def preprocess_emg(emg_mv: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Full sEMG chain: PLI -> band-pass -> rectify -> envelope LPF."""
    x = emg_mv.astype(float)
    if cfg.pli_method == "comb":
        x = comb_filter(x, fs, cfg)
    else:
        x = pasf_filter(x, fs, cfg)
    x = bandpass(x, fs, cfg)
    x = np.abs(x)
    return lowpass(x, fs, cfg.envelope_lpf_hz)


def preprocess_force(force_n: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """Force low-pass at configured cutoff."""
    return lowpass(force_n.astype(float), fs, cfg.force_lpf_hz)


def apply_baseline_and_trim(
    t: np.ndarray,
    emg: np.ndarray,
    force: np.ndarray,
    fs: float,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard filter transients and subtract DC baseline from first window."""
    discard = int(round(cfg.discard_start_s * fs))
    discard = min(discard, max(0, len(t) - 2))
    t = t[discard:]
    emg = emg[discard:]
    force = force[discard:]

    base_n = int(round(cfg.baseline_window_s * fs))
    base_n = min(base_n, len(force))
    if base_n > 0:
        force = force - np.mean(force[:base_n])
    return t, emg, force


def preprocess_trial(
    emg_mv: np.ndarray,
    force_n: np.ndarray,
    fs: float,
    cfg: PreprocessConfig,
    envelope: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (envelope, filtered_force). Uses provided envelope if given."""
    if envelope is None:
        env = preprocess_emg(emg_mv, fs, cfg)
    else:
        env = envelope.astype(float)
    f = preprocess_force(force_n, fs, cfg)
    return env, f
