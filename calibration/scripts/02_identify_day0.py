#!/usr/bin/env python3
"""Identify frozen EDM on day-0 (3rd_Base) data."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.identify import identify_edm
from src.io_utils import trials_for_split, write_run_sidecar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("02_identify_day0")


def main() -> None:
    cfg = load_config()
    day0 = trials_for_split("day0", cfg)
    if not day0:
        raise SystemExit(
            f"No day-0 data in {cfg.data.root / cfg.data.day0}. "
            "Run synth/generate.py or add CSVs to ../Data/3rd_Base/"
        )
    params = identify_edm(day0, cfg)
    out = cfg.path("processed") / "edm_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    params.to_json(out)
    hash_path = cfg.path("processed") / "edm_params.sha256"
    hash_path.write_text(params.sha256(), encoding="utf-8")
    write_run_sidecar(
        cfg.path("processed") / "02_identify_day0.json",
        cfg,
        "02_identify_day0.py",
        [str(cfg.data.root)],
        {"edm_sha256": params.sha256()},
    )
    logger.info("Saved EDM params -> %s", out)


if __name__ == "__main__":
    main()
