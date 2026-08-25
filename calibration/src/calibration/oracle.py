"""Oracle affine fitted on the entire test day (analysis ceiling only)."""

from __future__ import annotations

import numpy as np

from src.calibration.stage2_output import Stage2Affine
from src.config import CalibrationConfig
from src.io_utils import TrialData


class OracleAffine(Stage2Affine):
    """Deliberately violates the no-leakage rule for analysis-only ceilings."""

    @classmethod
    def for_analysis_only(cls, cfg: CalibrationConfig) -> "OracleAffine":
        return cls(cfg)

    def fit(self, trials: list[TrialData]) -> "OracleAffine":
        # Oracle is allowed to use TASK trials; documented violation.
        f_hat = []
        f_true = []
        for trial in trials:
            if not hasattr(trial, "_last_force_hat"):
                continue
            f_hat.append(trial._last_force_hat)
            f_true.append(trial.force_n)
        if not f_hat:
            raise ValueError("OracleAffine requires _last_force_hat on trials")
        self.fit_from_arrays(np.concatenate(f_hat), np.concatenate(f_true))
        return self
