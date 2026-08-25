"""Load config.yaml into frozen dataclasses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    root: Path
    day0: str
    train: tuple[str, ...]
    val: str
    test: str
    time_col: str
    emg_raw_col: str
    emg_env_col: str
    force_col: str
    use_preprocessed_envelope: bool
    rest_s: float
    reference_s: float
    calibration_s: float
    discard_start_s: float
    baseline_window_s: float


@dataclass(frozen=True)
class PreprocessConfig:
    pli_method: str
    comb_c1: float
    comb_c2: float
    comb_f0: float
    pasf_cutoff_hz: float
    bandpass_hz: tuple[float, float]
    envelope_lpf_hz: float
    force_lpf_hz: float
    discard_start_s: float
    baseline_window_s: float


@dataclass(frozen=True)
class EDMIdentifyConfig:
    bounds: dict[str, tuple[float, float]]
    method: str
    maxiter: int
    popsize: int


@dataclass(frozen=True)
class CalibrationConfig:
    stage1_reference_stat: str
    stage2_method: str
    alpha_min: float
    alpha_max: float


@dataclass(frozen=True)
class ArmConfig:
    id: str
    name: str
    stage1: bool
    stage2: bool
    oracle: bool = False


@dataclass(frozen=True)
class AppConfig:
    seed: int
    fs: float
    data: DataConfig
    preprocess: PreprocessConfig
    edm_identify: EDMIdentifyConfig
    calibration: CalibrationConfig
    arms: tuple[ArmConfig, ...]
    synth: dict[str, Any]
    paths: dict[str, str]
    config_path: Path = DEFAULT_CONFIG

    def path(self, key: str) -> Path:
        return ROOT / self.paths[key]

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fs": self.fs,
            "data": {
                "root": str(self.data.root),
                "day0": self.data.day0,
                "train": list(self.data.train),
                "val": self.data.val,
                "test": self.data.test,
            },
            "paths": self.paths,
        }


def _bounds(raw: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate config.yaml."""
    import os

    if path is None and os.environ.get("CALIBRATION_CONFIG"):
        path = Path(os.environ["CALIBRATION_CONFIG"])
    cfg_path = Path(path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    data_root = (cfg_path.parent / raw["data"]["root"]).resolve()
    seg = raw["data"]["segment_seconds"]

    data = DataConfig(
        root=data_root,
        day0=raw["data"]["sessions"]["day0"],
        train=tuple(raw["data"]["sessions"]["train"]),
        val=raw["data"]["sessions"]["val"],
        test=raw["data"]["sessions"]["test"],
        time_col=raw["data"]["csv"]["time_col"],
        emg_raw_col=raw["data"]["csv"]["emg_raw_col"],
        emg_env_col=raw["data"]["csv"]["emg_env_col"],
        force_col=raw["data"]["csv"]["force_col"],
        use_preprocessed_envelope=bool(raw["data"]["csv"]["use_preprocessed_envelope"]),
        rest_s=float(seg["rest"]),
        reference_s=float(seg["reference"]),
        calibration_s=float(seg["calibration"]),
        discard_start_s=float(seg["discard_start"]),
        baseline_window_s=float(seg["baseline_window"]),
    )

    pp = raw["preprocess"]
    preprocess = PreprocessConfig(
        pli_method=pp["pli_method"],
        comb_c1=float(pp["comb"]["c1"]),
        comb_c2=float(pp["comb"]["c2"]),
        comb_f0=float(pp["comb"]["f0"]),
        pasf_cutoff_hz=float(pp["pasf"]["cutoff_hz"]),
        bandpass_hz=(float(pp["bandpass_hz"][0]), float(pp["bandpass_hz"][1])),
        envelope_lpf_hz=float(pp["envelope_lpf_hz"]),
        force_lpf_hz=float(pp["force_lpf_hz"]),
        discard_start_s=float(pp["discard_start_s"]),
        baseline_window_s=float(pp["baseline_window_s"]),
    )

    edm_identify = EDMIdentifyConfig(
        bounds=_bounds(raw["edm"]["bounds"]),
        method=raw["edm"]["identify"]["method"],
        maxiter=int(raw["edm"]["identify"]["maxiter"]),
        popsize=int(raw["edm"]["identify"]["popsize"]),
    )

    cal = raw["calibration"]
    calibration = CalibrationConfig(
        stage1_reference_stat=cal["stage1"]["reference_stat"],
        stage2_method=cal["stage2"]["method"],
        alpha_min=float(cal["stage2"]["alpha_min"]),
        alpha_max=float(cal["stage2"]["alpha_max"]),
    )

    arms = tuple(
        ArmConfig(
            id=a["id"],
            name=a["name"],
            stage1=bool(a.get("stage1", False)),
            stage2=bool(a.get("stage2", False)),
            oracle=bool(a.get("stage2") == "oracle"),
        )
        for a in raw["arms"]
    )

    return AppConfig(
        seed=int(raw["seed"]),
        fs=float(raw["fs"]),
        data=data,
        preprocess=preprocess,
        edm_identify=edm_identify,
        calibration=calibration,
        arms=arms,
        synth=dict(raw["synth"]),
        paths=dict(raw["paths"]),
        config_path=cfg_path,
    )
