"""Line plots: real force vs EDM base vs cascade-calibrated force."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.edm import EDMParams
from src.evaluate import collect_task_force_rows, mae, r2, rmse
from src.io_utils import trials_for_split

logger = logging.getLogger(__name__)


def _plot_one_task(
    row: dict,
    out_path: Path,
    *,
    dpi: int = 150,
    max_points: int = 5000,
) -> dict:
    """Save one PNG and return summary metrics for CSV."""
    t = row["time_s"]
    y_true = row["force_true"]
    y_base = row["force_base"]
    y_cal = row["force_calibrated"]

    step = max(1, len(t) // max_points)
    sl = slice(None, None, step)

    base_m = {
        "RMSE": rmse(y_true, y_base),
        "MAE": mae(y_true, y_base),
        "R2": r2(y_true, y_base),
    }
    cal_m = {
        "RMSE": rmse(y_true, y_cal),
        "MAE": mae(y_true, y_cal),
        "R2": r2(y_true, y_cal),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t[sl], y_true[sl], label="Real Force", linewidth=1.2)
    ax.plot(
        t[sl],
        y_base[sl],
        label="EDM base (uncalibrated)",
        linewidth=1.0,
        alpha=0.85,
        linestyle="--",
    )
    ax.plot(
        t[sl],
        y_cal[sl],
        label="Calibrated (cascade, arm D)",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title(
        f"{row['session_id']} / {row['stem']} ({row['split']} task): "
        "Real vs EDM base vs Calibrated"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    metric_text = (
        f"Calibrated  RMSE={cal_m['RMSE']:.4f}  MAE={cal_m['MAE']:.4f}  "
        f"R2={cal_m['R2']:.4f}     "
        f"EDM base  RMSE={base_m['RMSE']:.4f}  MAE={base_m['MAE']:.4f}  "
        f"R2={base_m['R2']:.4f}     "
        f"RMSE improvement={base_m['RMSE'] - cal_m['RMSE']:.4f}"
    )
    fig.text(0.5, 0.02, metric_text, ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "split": row["split"],
        "session_id": row["session_id"],
        "stem": row["stem"],
        "trial_id": row["trial_id"],
        **{f"Base_{k}": v for k, v in base_m.items()},
        **{f"Calibrated_{k}": v for k, v in cal_m.items()},
        "RMSE_improvement": base_m["RMSE"] - cal_m["RMSE"],
    }


def save_session_force_plots(
    cfg: AppConfig,
    edm: EDMParams,
    split: str,
    session_id: str,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Plot task-segment force traces for one session."""
    day0 = trials_for_split("day0", cfg)
    session_trials = [t for t in trials_for_split(split, cfg) if t.session_id == session_id]
    rows = collect_task_force_rows(cfg, edm, day0, session_trials, split=split)

    if not rows:
        logger.warning("No task segments to plot for %s / %s", split, session_id)
        return pd.DataFrame()

    base_out = out_dir or (cfg.path("results_figures") / split / session_id)
    summaries = []
    for row in rows:
        out_path = base_out / f"{row['stem']}.png"
        summaries.append(_plot_one_task(row, out_path))
        logger.info("Saved force plot -> %s", out_path)

    return pd.DataFrame(summaries)


def save_all_force_plots(cfg: AppConfig, edm: EDMParams) -> pd.DataFrame:
    """Generate force comparison plots for val and test splits."""
    summaries: list[pd.DataFrame] = []
    for split, session in (
        ("val", cfg.data.val),
        ("test", cfg.data.test),
    ):
        df = save_session_force_plots(cfg, edm, split, session)
        if not df.empty:
            summaries.append(df)

    if not summaries:
        logger.warning("No force plots were generated (check segment_seconds vs recording length)")
        return pd.DataFrame()

    combined = pd.concat(summaries, ignore_index=True)
    out_csv = cfg.path("results_figures") / "force_plot_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    logger.info("Saved plot metrics -> %s", out_csv)
    return combined
