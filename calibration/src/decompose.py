"""Loss decomposition for drift-induced error."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.evaluate import rmse, run_arm
from src.io_utils import TrialRole, trials_by_role, trials_for_split

logger = logging.getLogger(__name__)


def oracle_affine_rmse(f_true: np.ndarray, f_hat: np.ndarray) -> float:
    """RMSE after best affine fit on the test day."""
    var = float(np.var(f_hat))
    alpha = float(np.cov(f_hat, f_true, bias=True)[0, 1] / var) if var > 1e-12 else 1.0
    beta = float(np.mean(f_true) - alpha * np.mean(f_hat))
    return rmse(f_true, alpha * f_hat + beta)


def _collect_task_predictions(cfg, edm, day0, session_trials, stage1_day0, arm: str):
    run_arm(arm, cfg, edm, day0, session_trials, stage1_day0)
    task = trials_by_role(session_trials, TrialRole.TASK)
    f_true = np.concatenate([t._last_force_true for t in task])
    f_hat = np.concatenate([t._last_force_hat for t in task])
    if arm in ("C", "D", "E"):
        from src.calibration.stage2_output import Stage2Affine
        from src.calibration.oracle import OracleAffine

        if arm == "E":
            s2 = OracleAffine.for_analysis_only(cfg.calibration)
            s2.fit(session_trials)
        else:
            s2 = Stage2Affine(cfg.calibration)
            cal = trials_by_role(session_trials, TrialRole.CALIBRATION)
            s2.fit_from_arrays(
                np.concatenate([t._last_force_hat for t in cal]),
                np.concatenate([t._last_force_true for t in cal]),
            )
        f_hat = s2.apply(f_hat)
    return f_true, f_hat


def decompose_session(cfg, edm, day0_trials, session_trials, stage1_day0, session_id) -> pd.DataFrame:
    e_floor, _ = run_arm("A", cfg, edm, day0_trials, day0_trials, stage1_day0)
    e_a, _ = run_arm("A", cfg, edm, day0_trials, session_trials, stage1_day0)
    l_total = e_a - e_floor

    rows = []
    for arm in ("A", "B", "C", "D", "E"):
        e_arm, _ = run_arm(arm, cfg, edm, day0_trials, session_trials, stage1_day0)
        f_true, f_hat = _collect_task_predictions(
            cfg, edm, day0_trials, session_trials, stage1_day0, arm
        )
        nonaff = oracle_affine_rmse(f_true, f_hat)
        r_arm = (e_a - e_arm) / l_total if abs(l_total) > 1e-12 else np.nan
        rows.append(
            {
                "subject_id": "sub-01",
                "session": session_id,
                "arm": arm,
                "E_floor": e_floor,
                "E_arm": e_arm,
                "L_total": l_total,
                "R_arm": r_arm,
                "nonaffine": nonaff,
                "L_affine": e_a - nonaff if arm == "A" else np.nan,
                "L_nonaffine": nonaff - e_floor if arm == "A" else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    a_non = float(df.loc[df.arm == "A", "nonaffine"].iloc[0])
    c_non = float(df.loc[df.arm == "C", "nonaffine"].iloc[0])
    if not np.isclose(a_non, c_non, rtol=0.05, atol=1e-3):
        logger.warning("nonaffine(C)=%.4f vs nonaffine(A)=%.4f", c_non, a_non)
    return df


def run_decomposition(cfg: AppConfig, edm) -> pd.DataFrame:
    from src.calibration.stage1_input import Stage1Normalizer

    day0 = trials_for_split("day0", cfg)
    stage1_day0 = Stage1Normalizer.from_day0(cfg.calibration, day0)
    parts = []
    for split, session in (("val", cfg.data.val), ("test", cfg.data.test)):
        session_trials = trials_for_split(split, cfg)
        if session_trials:
            parts.append(
                decompose_session(cfg, edm, day0, session_trials, stage1_day0, session)
            )
    out = pd.concat(parts, ignore_index=True)
    out_path = cfg.path("results_metrics") / "decomposition.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
