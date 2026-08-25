"""Metrics and arm execution."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.cascade import Cascade
from src.calibration.oracle import OracleAffine
from src.calibration.stage1_input import Stage1Normalizer
from src.calibration.stage2_output import Stage2Affine
from src.config import AppConfig
from src.edm import EDMParams, edm_forward
from src.io_utils import TrialData, TrialRole, trials_by_role
from src.preprocess import preprocess_trial

logger = logging.getLogger(__name__)


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def fit_cascade_stages(
    cfg: AppConfig,
    edm: EDMParams,
    day0_trials: list[TrialData],
    session_trials: list[TrialData],
) -> tuple[Stage1Normalizer, Stage2Affine]:
    """Fit cascade (arm D) calibrators on non-task segments."""
    stage1_day0 = Stage1Normalizer.from_day0(cfg.calibration, day0_trials)
    fit_trials = [t for t in session_trials if t.role != TrialRole.TASK]
    stage1 = Stage1Normalizer(cfg.calibration, stage1_day0.b0, stage1_day0.r0)
    stage1.fit(fit_trials)

    stage2 = Stage2Affine(cfg.calibration)
    cal = trials_by_role(session_trials, TrialRole.CALIBRATION)
    f_hat_parts, f_true_parts = [], []
    for trial in cal:
        env, force = preprocess_trial(
            trial.emg_mv, trial.force_n, trial.fs, cfg.preprocess, envelope=trial.envelope
        )
        env_n = stage1.apply(env)
        f_hat_parts.append(edm_forward(env_n, edm, cfg.fs))
        f_true_parts.append(force)
    if f_hat_parts:
        stage2.fit_from_arrays(np.concatenate(f_hat_parts), np.concatenate(f_true_parts))
    return stage1, stage2


def task_force_predictions(
    trial: TrialData,
    cfg: AppConfig,
    edm: EDMParams,
    stage1: Stage1Normalizer,
    stage2: Stage2Affine,
) -> dict[str, np.ndarray]:
    """Per-sample force traces for one task trial."""
    env, force = preprocess_trial(
        trial.emg_mv, trial.force_n, trial.fs, cfg.preprocess, envelope=trial.envelope
    )
    force_base = edm_forward(env, edm, cfg.fs)
    force_calibrated = stage2.apply(edm_forward(stage1.apply(env), edm, cfg.fs))
    return {
        "time_s": trial.time_s,
        "force_true": force,
        "force_base": force_base,
        "force_calibrated": force_calibrated,
    }


def collect_task_force_rows(
    cfg: AppConfig,
    edm: EDMParams,
    day0_trials: list[TrialData],
    session_trials: list[TrialData],
    *,
    split: str = "",
) -> list[dict]:
    """Build plot-ready rows for all task trials in a session."""
    task_trials = trials_by_role(session_trials, TrialRole.TASK)
    if not task_trials:
        return []

    stage1, stage2 = fit_cascade_stages(cfg, edm, day0_trials, session_trials)
    rows: list[dict] = []
    for trial in task_trials:
        preds = task_force_predictions(trial, cfg, edm, stage1, stage2)
        rows.append(
            {
                "split": split,
                "session_id": trial.session_id,
                "stem": Path(trial.source_file).stem,
                "trial_id": trial.trial_id,
                **preds,
            }
        )
    return rows


def run_arm(
    arm_id: str,
    cfg: AppConfig,
    edm: EDMParams,
    day0_trials: list[TrialData],
    session_trials: list[TrialData],
    stage1_day0: Stage1Normalizer | None = None,
) -> tuple[float, list[TrialData]]:
    """Fit calibrators on session trials and return RMSE on TASK trials."""
    stage1_day0 = stage1_day0 or Stage1Normalizer.from_day0(cfg.calibration, day0_trials)
    fit_trials = [t for t in session_trials if t.role != TrialRole.TASK]
    task_trials = trials_by_role(session_trials, TrialRole.TASK)

    stage1 = Stage1Normalizer(cfg.calibration, stage1_day0.b0, stage1_day0.r0)
    stage2 = Stage2Affine(cfg.calibration)

    use_s1 = arm_id in ("B", "D")
    use_s2 = arm_id in ("C", "D")
    oracle = arm_id == "E"

    if use_s1:
        stage1.fit(fit_trials)

    if use_s2 and not oracle:
        cal = trials_by_role(session_trials, TrialRole.CALIBRATION)
        f_hat_parts, f_true_parts = [], []
        for trial in cal:
            env, force = preprocess_trial(
                trial.emg_mv, trial.force_n, trial.fs, cfg.preprocess, envelope=trial.envelope
            )
            env_in = stage1.apply(env) if use_s1 else env
            f_hat_parts.append(edm_forward(env_in, edm, cfg.fs))
            f_true_parts.append(force)
        stage2.fit_from_arrays(np.concatenate(f_hat_parts), np.concatenate(f_true_parts))

    for trial in session_trials:
        env, force = preprocess_trial(
            trial.emg_mv, trial.force_n, trial.fs, cfg.preprocess, envelope=trial.envelope
        )
        env_in = stage1.apply(env) if use_s1 else env
        f_hat = edm_forward(env_in, edm, cfg.fs)
        trial._last_force_hat = f_hat  # type: ignore[attr-defined]
        trial._last_force_true = force  # type: ignore[attr-defined]

    if oracle:
        oracle_fit = OracleAffine.for_analysis_only(cfg.calibration)
        oracle_fit.fit(session_trials)
        stage2 = oracle_fit

    errs = []
    for trial in task_trials:
        f_hat = trial._last_force_hat.copy()
        f_true = trial._last_force_true
        if use_s2 or oracle:
            f_hat = stage2.apply(f_hat)
        errs.append(rmse(f_true, f_hat))

    return float(np.mean(errs)), task_trials


def evaluate_all_arms(cfg: AppConfig, edm: EDMParams) -> pd.DataFrame:
    """Run arms A–E on val and test sessions."""
    from src.io_utils import trials_for_split

    day0 = trials_for_split("day0", cfg)
    stage1_day0 = Stage1Normalizer.from_day0(cfg.calibration, day0)
    rows = []

    for split in ("val", "test"):
        for session in ([cfg.data.val] if split == "val" else [cfg.data.test]):
            session_trials = [t for t in trials_for_split(split, cfg)]
            e_floor = None
            for arm in ("A", "B", "C", "D", "E"):
                e_arm, _ = run_arm(arm, cfg, edm, day0, session_trials, stage1_day0)
                if split == "test" and arm == "A":
                    # approximate floor from day0 task if available
                    d0_task = trials_by_role(day0, TrialRole.TASK)
                    if d0_task:
                        e_floor = run_arm("A", cfg, edm, day0, day0, stage1_day0)[0]
                rows.append(
                    {
                        "split": split,
                        "session": session,
                        "arm": arm,
                        "rmse": e_arm,
                        "E_floor": e_floor,
                    }
                )
    return pd.DataFrame(rows)
