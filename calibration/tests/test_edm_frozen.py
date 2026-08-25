"""EDM frozen hash tests."""

from pathlib import Path

import numpy as np

from src.config import load_config
from src.edm import EDMParams, assert_frozen, edm_forward


def test_edm_forward_finite():
    cfg = load_config()
    p = EDMParams(1.0, 2.0, 0.01, 0.0, 5.0, 10.0)
    env = np.linspace(0, 1, 1000)
    out = edm_forward(env, p, cfg.fs)
    assert np.isfinite(out).all()


def test_assert_frozen_raises(tmp_path: Path):
    p = EDMParams(1.0, 2.0, 0.01, 0.0, 5.0, 10.0)
    try:
        assert_frozen(p, "deadbeef")
        raised = False
    except ValueError:
        raised = True
    assert raised
