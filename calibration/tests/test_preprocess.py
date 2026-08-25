"""Tests for preprocessing."""

import numpy as np

from src.config import load_config
from src.preprocess import comb_filter, preprocess_emg, preprocess_force


def test_preprocess_finite():
    cfg = load_config()
    fs = cfg.fs
    x = np.random.randn(int(fs))
    env = preprocess_emg(x, fs, cfg.preprocess)
    f = preprocess_force(x, fs, cfg.preprocess)
    assert np.isfinite(env).all()
    assert np.isfinite(f).all()


def test_comb_filter_same_length():
    cfg = load_config()
    x = np.ones(4000)
    y = comb_filter(x, cfg.fs, cfg.preprocess)
    assert len(y) == len(x)
