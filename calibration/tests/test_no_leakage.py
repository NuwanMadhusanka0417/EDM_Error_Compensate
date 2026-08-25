"""Leakage guard tests."""

import numpy as np
import pytest

from src.calibration.stage1_input import Stage1Normalizer
from src.config import load_config
from src.io_utils import LeakageError, TrialData, TrialRole


def _trial(role: TrialRole) -> TrialData:
    n = 100
    return TrialData(
        time_s=np.arange(n) / 2000.0,
        emg_mv=np.ones(n),
        force_n=np.ones(n),
        envelope=np.ones(n),
        role=role,
        subject_id="sub-01",
        session_id="test",
        trial_id=f"t_{role.value}",
        source_file="x.csv",
    )


def test_stage1_rejects_task_trial():
    cfg = load_config()
    s1 = Stage1Normalizer(cfg.calibration, 0.1, 0.5)
    with pytest.raises(LeakageError):
        s1.fit([_trial(TrialRole.TASK)])
