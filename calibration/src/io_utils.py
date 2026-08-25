"""Manifest building, trial loading, and run sidecars."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)


class TrialRole(str, Enum):
    REST = "rest"
    REFERENCE = "reference"
    CALIBRATION = "calibration"
    TASK = "task"


class LeakageError(RuntimeError):
    """Raised when a calibrator is fit on evaluation-only data."""


@dataclass
class TrialData:
    """One contiguous trial segment."""

    time_s: np.ndarray
    emg_mv: np.ndarray
    force_n: np.ndarray
    envelope: np.ndarray | None
    role: TrialRole
    subject_id: str
    session_id: str
    trial_id: str
    source_file: str

    @property
    def fs(self) -> float:
        if len(self.time_s) < 2:
            return 2000.0
        dt = np.median(np.diff(self.time_s))
        return 1.0 / dt if dt > 0 else 2000.0


def _session_csvs(data_root: Path, session: str) -> list[Path]:
    folder = data_root / session
    if not folder.is_dir():
        return []
    return sorted(folder.glob("Data*.csv"))


def build_manifest(cfg: AppConfig) -> pd.DataFrame:
    """Scan Data/<session>/Data*.csv and register pseudo-trial segments."""
    rows = []
    for session in (cfg.data.day0, *cfg.data.train, cfg.data.val, cfg.data.test):
        for path in _session_csvs(cfg.data.root, session):
            rows.append(
                {
                    "subject_id": "sub-01",
                    "session_id": session,
                    "source_file": str(path),
                    "stem": path.stem,
                }
            )
    if not rows:
        logger.warning("No CSV files found under %s", cfg.data.root)
    return pd.DataFrame(rows)


def _load_csv(path: Path, cfg: AppConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    t = pd.to_numeric(df[cfg.data.time_col], errors="coerce").to_numpy(float)
    if cfg.data.use_preprocessed_envelope:
        emg = pd.to_numeric(df[cfg.data.emg_env_col], errors="coerce").to_numpy(float)
    else:
        emg = pd.to_numeric(df[cfg.data.emg_raw_col], errors="coerce").to_numpy(float)
    force = pd.to_numeric(df[cfg.data.force_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(emg) & np.isfinite(force)
    return t[valid], emg[valid], force[valid]


def _segment_bounds(n: int, fs: float, cfg: AppConfig) -> list[tuple[TrialRole, int, int]]:
    """Split one recording into rest / reference / calibration / task segments."""
    start = int(round(cfg.data.discard_start_s * fs))
    start = min(start, max(0, n - 4))
    rest_end = start + int(round(cfg.data.rest_s * fs))
    ref_end = rest_end + int(round(cfg.data.reference_s * fs))
    cal_end = ref_end + int(round(cfg.data.calibration_s * fs))
    cal_end = min(cal_end, n)
    ref_end = min(ref_end, cal_end)
    rest_end = min(rest_end, ref_end)

    segments: list[tuple[TrialRole, int, int]] = []
    if rest_end > start:
        segments.append((TrialRole.REST, start, rest_end))
    if ref_end > rest_end:
        segments.append((TrialRole.REFERENCE, rest_end, ref_end))
    if cal_end > ref_end:
        segments.append((TrialRole.CALIBRATION, ref_end, cal_end))
    if n > cal_end:
        segments.append((TrialRole.TASK, cal_end, n))
    return segments


def load_session_trials(session: str, cfg: AppConfig) -> list[TrialData]:
    """Load all pseudo-trials for one session folder."""
    trials: list[TrialData] = []
    for path in _session_csvs(cfg.data.root, session):
        t, emg, force = _load_csv(path, cfg)
        if len(t) < 10:
            continue
        fs = 1.0 / np.median(np.diff(t))
        for role, i0, i1 in _segment_bounds(len(t), fs, cfg):
            env = emg[i0:i1].copy() if cfg.data.use_preprocessed_envelope else None
            trials.append(
                TrialData(
                    time_s=t[i0:i1],
                    emg_mv=emg[i0:i1],
                    force_n=force[i0:i1],
                    envelope=env,
                    role=role,
                    subject_id="sub-01",
                    session_id=session,
                    trial_id=f"{path.stem}_{role.value}",
                    source_file=str(path),
                )
            )
    return trials


def trials_for_split(split: str, cfg: AppConfig) -> list[TrialData]:
    """Return trials for train / val / test / day0 splits."""
    if split == "day0":
        sessions = [cfg.data.day0]
    elif split == "train":
        sessions = list(cfg.data.train)
    elif split == "val":
        sessions = [cfg.data.val]
    elif split == "test":
        sessions = [cfg.data.test]
    else:
        raise ValueError(f"Unknown split: {split}")

    out: list[TrialData] = []
    for session in sessions:
        out.extend(load_session_trials(session, cfg))
    return out


def trials_by_role(trials: Iterable[TrialData], role: TrialRole) -> list[TrialData]:
    return [t for t in trials if t.role == role]


def write_run_sidecar(
    out_path: Path,
    cfg: AppConfig,
    script_name: str,
    inputs: list[str],
    extra: dict | None = None,
) -> None:
    """Record config hash, timestamp, and inputs for reproducibility."""
    payload = {
        "script": script_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.config_hash(),
        "config_path": str(cfg.config_path),
        "seed": cfg.seed,
        "inputs": inputs,
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_manifest(manifest: pd.DataFrame, cfg: AppConfig) -> Path:
    out = cfg.path("interim") / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    return out
