"""Stage-2 output affine correction."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor

from src.calibration.base import Calibrator
from src.config import CalibrationConfig
from src.io_utils import TrialData, TrialRole, trials_by_role

logger = logging.getLogger(__name__)


class Stage2Affine(Calibrator):
    domain = "output"
    requires_force = True

    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.alpha: float | None = None
        self.beta: float | None = None

    def fit(self, trials: list[TrialData]) -> "Stage2Affine":
        self._check_fit_trials(trials)
        cal = trials_by_role(trials, TrialRole.CALIBRATION)
        if not cal:
            raise ValueError("Stage2Affine requires calibration trials")

        f_hat = np.concatenate([getattr(t, "_last_force_hat", np.zeros_like(t.force_n)) for t in cal])
        f_true = np.concatenate([t.force_n for t in cal])
        self._fit_affine(f_hat, f_true)
        return self

    def fit_from_arrays(self, f_hat: np.ndarray, f_true: np.ndarray) -> "Stage2Affine":
        self._fit_affine(f_hat, f_true)
        return self

    def _fit_affine(self, f_hat: np.ndarray, f_true: np.ndarray) -> None:
        x = f_hat.reshape(-1, 1)
        y = f_true
        if self.cfg.stage2_method == "huber":
            reg = HuberRegressor()
            reg.fit(x, y)
            alpha = float(reg.coef_[0])
            beta = float(reg.intercept_)
        elif self.cfg.stage2_method == "ransac":
            reg = RANSACRegressor()
            reg.fit(x, y)
            alpha = float(reg.estimator_.coef_[0])
            beta = float(reg.estimator_.intercept_)
        else:
            var = float(np.var(f_hat))
            alpha = float(np.cov(f_hat, f_true, bias=True)[0, 1] / var) if var > 1e-12 else 1.0
            beta = float(np.mean(f_true) - alpha * np.mean(f_hat))

        if alpha < self.cfg.alpha_min or alpha > self.cfg.alpha_max:
            logger.warning("Stage2 alpha %.4f clamped to [%.4f, %.4f]", alpha, self.cfg.alpha_min, self.cfg.alpha_max)
            alpha = float(np.clip(alpha, self.cfg.alpha_min, self.cfg.alpha_max))
        self.alpha = alpha
        self.beta = beta

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.beta is None:
            raise RuntimeError("Stage2Affine.fit must be called before apply")
        return self.alpha * x + self.beta

    def params_(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}
