#!/usr/bin/env python3
"""End-to-end pipeline: synth (optional) -> manifest -> identify -> run arms."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--use-synth", action="store_true", help="Generate synthetic data first")
    p.add_argument("--synth-config", type=str, default=None)
    args = p.parse_args()

    if args.use_synth:
        subprocess.check_call([sys.executable, str(ROOT / "synth" / "generate.py")])
        # Point config at synthetic data via env override handled in wrapper config
        synth_cfg = ROOT / "config.synth.yaml"
        if not synth_cfg.exists():
            base = (ROOT / "config.yaml").read_text(encoding="utf-8")
            synth_cfg.write_text(base.replace("../Data", "data/synthetic"), encoding="utf-8")

    cfg_flag = ["--synth"] if args.use_synth else []
    for script in ("00_make_manifest.py", "02_identify_day0.py", "03_run_arms.py"):
        cmd = [sys.executable, str(ROOT / "scripts" / script)]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
