"""Day-0 EDM identification."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from src.config import AppConfig
from src.edm import EDMParams, edm_forward, fit_vector_bounds
from src.io_utils import TrialData, TrialRole
from src.preprocess import preprocess_trial

logger = logging.getLogger(__name__)


def _collect_xy(trials: list[TrialData], cfg: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    env_parts, force_parts = [], []
    for trial in trials:
        if trial.role == TrialRole.TASK:
            continue
        env, force = preprocess_trial(
            trial.emg_mv,
            trial.force_n,
            trial.fs,
            cfg.preprocess,
            envelope=trial.envelope,
        )
        env_parts.append(env)
        force_parts.append(force)
    if not env_parts:
        raise ValueError("No trials available for EDM identification")
    return np.concatenate(env_parts), np.concatenate(force_parts)


def identify_edm(day0_trials: list[TrialData], cfg: AppConfig) -> EDMParams:
    """Identify EDM on day-0 data using differential evolution."""
    env, force = _collect_xy(trials=day0_trials, cfg=cfg)
    lower, upper = fit_vector_bounds(cfg)
    fs = cfg.fs

    def objective(vec: np.ndarray) -> float:
        params = EDMParams.from_vector(vec)
        pred = edm_forward(env, params, fs)
        return float(np.mean((pred - force) ** 2))

    if cfg.edm_identify.method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=list(zip(lower, upper)),
            seed=cfg.seed,
            maxiter=cfg.edm_identify.maxiter,
            popsize=cfg.edm_identify.popsize,
            polish=True,
        )
        vec = result.x
    else:
        x0 = 0.5 * (lower + upper)

        def residual(vec: np.ndarray) -> np.ndarray:
            params = EDMParams.from_vector(vec)
            return edm_forward(env, params, fs) - force

        result = least_squares(residual, x0, bounds=(lower, upper))
        vec = result.x

    params = EDMParams.from_vector(vec)
    logger.info("Identified EDM params: %s", params)
    return params
