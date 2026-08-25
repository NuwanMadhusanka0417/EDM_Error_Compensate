"""Synthetic recovery tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_pipeline_end_to_end():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, str(ROOT / "synth" / "generate.py")])
    cfg_path = ROOT / "config.synth.yaml"

    from src.config import load_config
    from src.identify import identify_edm
    from src.io_utils import trials_for_split
    from src.evaluate import run_arm
    from src.calibration.stage1_input import Stage1Normalizer

    cfg = load_config(cfg_path)
    day0 = trials_for_split("day0", cfg)
    assert day0
    edm = identify_edm(day0, cfg)
    test_trials = trials_for_split("test", cfg)
    assert test_trials
    s1d0 = Stage1Normalizer.from_day0(cfg.calibration, day0)
    e_a, _ = run_arm("A", cfg, edm, day0, test_trials, s1d0)
    e_d, _ = run_arm("D", cfg, edm, day0, test_trials, s1d0)
    assert e_d <= e_a * 1.1 + 0.05
