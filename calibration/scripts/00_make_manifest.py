#!/usr/bin/env python3
"""Build manifest from Data/ session folders."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import build_manifest, save_manifest, write_run_sidecar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("00_make_manifest")


def main() -> None:
    cfg = load_config()
    manifest = build_manifest(cfg)
    out = save_manifest(manifest, cfg)
    sidecar = cfg.path("interim") / "00_make_manifest.json"
    write_run_sidecar(sidecar, cfg, "00_make_manifest.py", [str(cfg.data.root)])
    logger.info("Wrote manifest (%d files) -> %s", len(manifest), out)


if __name__ == "__main__":
    main()
