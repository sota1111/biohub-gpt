import json
from pathlib import Path

import numpy as np
import pytest

from biohub_baseline.detect import detect_centroids
from biohub_baseline.experiment import deterministic_split, promotion_decision
from biohub_baseline.submission import build_rows, validate_rows


def moving_cell_frames():
    frames = []
    for x in (4, 5, 6):
        frame = np.zeros((8, 12, 12), dtype=np.float32)
        frame[2:4, 5:7, x : x + 2] = 10
        frames.append(frame)
    return frames


def test_detect_and_track_is_deterministic():
    config = {"threshold_percentile": 99.0, "min_voxels": 4, "max_link_distance": 3}
    first = build_rows("fixture", moving_cell_frames(), **config)
    second = build_rows("fixture", moving_cell_frames(), **config)
    assert first == second
    assert [row["row_type"] for row in first].count("node") == 3
    assert [row["row_type"] for row in first].count("edge") == 2
    validate_rows(first)


def test_detect_rejects_non_volume():
    with pytest.raises(ValueError, match="3-D"):
        detect_centroids(np.zeros((3, 3)))


def test_split_is_seeded_and_disjoint():
    ids = ["a", "b", "c", "d", "e"]
    split = deterministic_split(ids, seed=1988, screen_fraction=0.4)
    assert split == deterministic_split(list(reversed(ids)), seed=1988, screen_fraction=0.4)
    assert set(split["screen"]).isdisjoint(split["confirm"])
    assert set(split["screen"] + split["confirm"]) == set(ids)


def test_promotion_requires_both_stages():
    champion = {
        "screen": {"composite": 0.7},
        "confirm": {"composite": 0.7},
    }
    candidate = {
        "screen": {"composite": 0.72, "detection_f1": 0.8, "edge_f1": 0.6},
        "confirm": {"composite": 0.71, "detection_f1": 0.8, "edge_f1": 0.6},
    }
    gates = {
        "screen_min_delta": 0.01,
        "confirm_min_delta": 0.005,
        "min_detection_f1": 0.5,
        "min_edge_f1": 0.3,
    }
    assert promotion_decision(candidate, champion, gates)["promote"] is True
    candidate["confirm"]["composite"] = 0.7
    assert promotion_decision(candidate, champion, gates)["promote"] is False


def test_champion_metadata_is_valid():
    champion = json.loads(Path("config/champion.json").read_text(encoding="utf-8"))
    metrics = json.loads(Path(champion["artifact"]).read_text(encoding="utf-8"))
    assert metrics["champion_id"] == champion["champion_id"]
