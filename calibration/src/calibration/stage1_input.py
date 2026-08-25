"""Stage-1 input normalizer (sensor-free)."""

from __future__ import annotations

import logging

import numpy as np

from src.calibration.base import Calibrator
from src.config import CalibrationConfig
from src.io_utils import TrialData, TrialRole, trials_by_role

logger = logging.getLogger(__name__)


class Stage1Normalizer(Calibrator):
    domain = "input"
    requires_force = False

    def __init__(self, cfg: CalibrationConfig, b0: float, r0: float):
        self.cfg = cfg
        self.b0 = float(b0)
        self.r0 = float(r0)
        self.b_k: float | None = None
        self.r_k: float | None = None
        self.drift_factor: float | None = None

    @classmethod
    def from_day0(cls, cfg: CalibrationConfig, day0_trials: list[TrialData]) -> "Stage1Normalizer":
        rest = trials_by_role(day0_trials, TrialRole.REST)
        ref = trials_by_role(day0_trials, TrialRole.REFERENCE)
        b0 = float(np.mean([np.mean(t.envelope if t.envelope is not None else t.emg_mv) for t in rest]))
        if cfg.stage1_reference_stat == "p95":
            r0 = float(np.mean([np.percentile(t.envelope if t.envelope is not None else t.emg_mv, 95) for t in ref]))
        else:
            r0 = float(np.mean([np.mean(t.envelope if t.envelope is not None else t.emg_mv) for t in ref]))
        return cls(cfg, b0, r0)

    def fit(self, trials: list[TrialData]) -> "Stage1Normalizer":
        self._check_fit_trials(trials)
        rest = trials_by_role(trials, TrialRole.REST)
        ref = trials_by_role(trials, TrialRole.REFERENCE)
        self.b_k = float(np.mean([np.mean(t.envelope if t.envelope is not None else t.emg_mv) for t in rest]))
        if self.cfg.stage1_reference_stat == "p95":
            self.r_k = float(np.mean([np.percentile(t.envelope if t.envelope is not None else t.emg_mv, 95) for t in ref]))
        elif self.cfg.stage1_reference_stat == "quantile_matching":
            self.r_k = float(np.mean([np.percentile(t.envelope if t.envelope is not None else t.emg_mv, 95) for t in ref]))
        else:
            self.r_k = float(np.mean([np.mean(t.envelope if t.envelope is not None else t.emg_mv) for t in ref]))
        denom = self.r_k - self.b_k
        num = self.r0 - self.b0
        self.drift_factor = num / denom if abs(denom) > 1e-12 else 1.0
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.b_k is None or self.r_k is None or self.drift_factor is None:
            raise RuntimeError("Stage1Normalizer.fit must be called before apply")
        denom = self.r_k - self.b_k
        if abs(denom) < 1e-12:
            return x.copy()
        return (x - self.b_k) * (self.r0 - self.b0) / denom + self.b0

    def params_(self) -> dict:
        return {
            "b0": self.b0,
            "r0": self.r0,
            "b_k": self.b_k,
            "r_k": self.r_k,
            "drift_factor": self.drift_factor,
        }
