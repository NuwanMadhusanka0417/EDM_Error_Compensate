"""Synthetic dataset generator matching ../Data session layout."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.edm import EDMParams, edm_forward

logger = logging.getLogger(__name__)


def _burst_envelope(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    env = 0.05 + 0.02 * rng.standard_normal(n)
    for _ in range(8):
        c = rng.integers(int(0.1 * n), int(0.9 * n))
        w = int(rng.integers(0.5 * fs, 2.0 * fs))
        i0 = max(0, c - w // 2)
        i1 = min(n, c + w // 2)
        env[i0:i1] += rng.uniform(0.2, 1.0)
    return np.clip(env, 0.0, None)


def generate(cfg_path: Path | None = None) -> Path:
    cfg = load_config(cfg_path)
    rng = np.random.default_rng(cfg.seed)
    synth_root = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    gt: dict = {"sessions": {}, "true_edm": None}

    true_edm = EDMParams(1.2, 3.0, 0.02, 0.01, 8.0, 20.0)
    gt["true_edm"] = {
        "b1": true_edm.b1,
        "b2": true_edm.b2,
        "b3": true_edm.b3,
        "b4": true_edm.b4,
        "b5": true_edm.b5,
        "b6": true_edm.b6,
    }

    n = int(cfg.synth["duration_s"] * cfg.fs)
    sessions = [
        cfg.data.day0,
        *cfg.data.train,
        cfg.data.val,
        cfg.data.test,
    ]
    sessions = list(dict.fromkeys(sessions))

    g0, c0 = 1.0, 0.0
    e0 = _burst_envelope(n, cfg.fs, rng)

    for i, session in enumerate(sessions):
        if session == cfg.data.day0:
            g_k, c_k = g0, c0
        else:
            gr = cfg.synth["drift_gain_range"]
            cr = cfg.synth["drift_offset_range"]
            g_k = rng.uniform(gr[0], gr[1])
            c_k = rng.uniform(cr[0], cr[1])

        env = g_k * e0 + c_k
        force = edm_forward(env, true_edm, cfg.fs)
        force += 0.02 * rng.standard_normal(n)
        t = np.arange(n) / cfg.fs

        out_dir = synth_root / session
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "Data1.csv"
        pd.DataFrame(
            {
                "Time": t,
                "Cali_Rectified_PASF_sEMG": env,
                "Cali_LPF_PASF_sEMG": env,
                "Force": force,
            }
        ).to_csv(path, index=False)

        gt["sessions"][session] = {"g_k": g_k, "c_k": c_k, "file": str(path)}

    gt_path = Path(__file__).resolve().parent / "ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    logger.info("Synthetic data written to %s", synth_root)
    return synth_root


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate()
