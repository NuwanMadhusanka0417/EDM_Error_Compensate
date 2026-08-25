"""Cascade: Stage1 -> EDM -> Stage2."""

from __future__ import annotations

import numpy as np

from src.calibration.stage1_input import Stage1Normalizer
from src.calibration.stage2_output import Stage2Affine
from src.config import AppConfig
from src.edm import EDMParams, edm_forward
from src.io_utils import TrialData, TrialRole
from src.preprocess import preprocess_trial


class Cascade:
    def __init__(
        self,
        cfg: AppConfig,
        edm: EDMParams,
        stage1: Stage1Normalizer,
        stage2: Stage2Affine,
    ):
        self.cfg = cfg
        self.edm = edm
        self.stage1 = stage1
        self.stage2 = stage2

    def fit(self, trials: list[TrialData]) -> "Cascade":
        self.stage1.fit([t for t in trials if t.role != TrialRole.TASK])
        cal = [t for t in trials if t.role == TrialRole.CALIBRATION]
        f_hat_parts, f_true_parts = [], []
        for trial in cal:
            env, force = preprocess_trial(
                trial.emg_mv, trial.force_n, trial.fs, self.cfg.preprocess, envelope=trial.envelope
            )
            env_n = self.stage1.apply(env)
            pred = edm_forward(env_n, self.edm, self.cfg.fs)
            f_hat_parts.append(pred)
            f_true_parts.append(force)
        self.stage2.fit_from_arrays(np.concatenate(f_hat_parts), np.concatenate(f_true_parts))
        return self

    def predict_envelope(self, env: np.ndarray) -> np.ndarray:
        return self.stage1.apply(env)

    def predict_force(self, env: np.ndarray) -> np.ndarray:
        env_n = self.stage1.apply(env)
        f_hat = edm_forward(env_n, self.edm, self.cfg.fs)
        return self.stage2.apply(f_hat)
