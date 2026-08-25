"""Calibrator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.io_utils import LeakageError, TrialData, TrialRole


class Calibrator(ABC):
    domain: str
    requires_force: bool

    @abstractmethod
    def fit(self, trials: list[TrialData]) -> "Calibrator":
        ...

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def params_(self) -> dict:
        ...

    def _check_fit_trials(self, trials: list[TrialData]) -> None:
        for trial in trials:
            if trial.role == TrialRole.TASK:
                raise LeakageError(
                    f"{self.__class__.__name__}.fit received TASK trial {trial.trial_id}"
                )
