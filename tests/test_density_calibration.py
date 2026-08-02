import json
from pathlib import Path

import numpy as np
import pytest

from biohub_baseline.density_calibration import DensityCalibration, detect_density_calibrated


def test_density_calibration_is_deterministic_and_anisotropy_aware():
    volume = np.zeros((8, 32, 32), dtype=float)
    volume[3:5, 10:13, 10:13] = 20
    volume[3:5, 10:13, 14:17] = 18
    config = DensityCalibration(3, 0.5, 3, 1, 3.0, 0.5)
    first = detect_density_calibrated(volume, config, 0, 2)
    assert first == detect_density_calibrated(volume, config, 0, 2)
    assert len(first) == 1


def test_density_calibration_rejects_invalid_inputs():
    config = DensityCalibration(3, 0.5, 3, 1, 3.0, 0.5)
    with pytest.raises(ValueError, match="3-D"):
        detect_density_calibrated(np.zeros((3, 3)), config, 0, 1)
    with pytest.raises(ValueError, match="frame index"):
        detect_density_calibrated(np.zeros((3, 3, 3)), config, 1, 1)


def test_real_screen_runs_are_deterministic_and_do_not_touch_confirm():
    root = Path(__file__).resolve().parents[1]
    runs = [
        json.loads((root / f"artifacts/sot-2274-density-calibration-run{run}.json").read_text())
        for run in (1, 2)
    ]
    for ledger in runs:
        assert ledger["split"]["disjoint"] is True
        assert ledger["confirm_accessed"] is False
        assert ledger["champion_updated"] is False
        assert ledger["kaggle_submission_executed"] is False
        assert ledger["tracker_id"] == "daughter-geometry-v1"
        assert ledger["screen_passed"] is False
        assert ledger["winner"] is None
        assert all(item["reference_consistency_errors"] == 0 for item in ledger["candidates"])

    stable_fields = ("id", "detections_sha256", "graph_sha256", "pred_nodes", "pred_edges", "score")
    first = [{key: item[key] for key in stable_fields} for item in runs[0]["candidates"]]
    second = [{key: item[key] for key in stable_fields} for item in runs[1]["candidates"]]
    assert first == second
