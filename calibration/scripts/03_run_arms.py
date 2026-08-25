#!/usr/bin/env python3
"""Run calibration arms and decomposition on val/test splits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.decompose import run_decomposition
from src.edm import EDMParams, assert_frozen
from src.evaluate import evaluate_all_arms
from src.figures.force_plots import save_all_force_plots
from src.io_utils import write_run_sidecar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("03_run_arms")


def main() -> None:
    cfg = load_config()
    edm_path = cfg.path("processed") / "edm_params.json"
    hash_path = cfg.path("processed") / "edm_params.sha256"
    if not edm_path.exists():
        raise SystemExit("Run scripts/02_identify_day0.py first")
    edm = EDMParams.from_json(edm_path)
    assert_frozen(edm, hash_path.read_text(encoding="utf-8").strip())

    metrics = evaluate_all_arms(cfg, edm)
    metrics_path = cfg.path("results_metrics") / "arm_rmse.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    decomp = run_decomposition(cfg, edm)
    plot_metrics = save_all_force_plots(cfg, edm)
    write_run_sidecar(
        cfg.path("results_metrics") / "03_run_arms.json",
        cfg,
        "03_run_arms.py",
        [str(edm_path)],
        {
            "n_metrics_rows": len(metrics),
            "n_decomp_rows": len(decomp),
            "n_force_plots": len(plot_metrics),
        },
    )
    logger.info("Saved metrics -> %s", metrics_path)
    logger.info("Saved decomposition -> %s", cfg.path("results_metrics") / "decomposition.csv")
    logger.info("Saved force plots -> %s", cfg.path("results_figures"))


if __name__ == "__main__":
    main()
