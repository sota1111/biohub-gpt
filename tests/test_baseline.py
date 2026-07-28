import json
from pathlib import Path

import numpy as np
import pytest

from biohub_baseline.detect import detect_adaptive_centroids, detect_centroids
from biohub_baseline.evaluate import combine_metrics, count_identity_switches, validate_lineage
from biohub_baseline.experiment import deterministic_split, promotion_decision
from biohub_baseline.preprocess import (
    SpatialTransform,
    estimate_phase_shift,
    estimate_reference_transforms,
)
from biohub_baseline.submission import (
    build_rows,
    extract_appearance_descriptor,
    validate_rows,
)
from biohub_baseline.track import (
    Detection,
    LinkConfig,
    build_candidate_edges,
    link_constrained,
)


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


def test_adaptive_detector_splits_close_peaks_and_refines_coordinates():
    coordinates = np.indices((10, 20, 20), dtype=float)
    expected = [(4.2, 9.3, 7.1), (4.4, 9.5, 11.0)]
    volume = np.zeros((10, 20, 20), dtype=float)
    for center in expected:
        distance_squared = sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
        volume += 8 * np.exp(-distance_squared / (2 * 1.0**2))
    detected = detect_adaptive_centroids(
        volume, threshold_percentile=85, peak_distance=1, nms_distance=2
    )
    assert len(detected) == 2
    assert all(
        min(np.linalg.norm(np.subtract(point, target)) for point in detected) < 0.5
        for target in expected
    )


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


def test_mutual_knn_candidates_are_sparse_and_reproducible():
    previous = [Detection(1, 0, 0, 0, 0), Detection(2, 0, 0, 10, 0)]
    current = [Detection(3, 1, 0, 1, 0), Detection(4, 1, 0, 9, 0)]
    config = LinkConfig(max_distance=12, k_neighbors=1)
    first = build_candidate_edges(previous, current, config)
    second = build_candidate_edges(list(reversed(previous)), list(reversed(current)), config)
    assert [(edge.source.node_id, edge.target.node_id) for edge in first] == [(1, 3), (2, 4)]
    assert {(edge.source.node_id, edge.target.node_id) for edge in first} == {
        (edge.source.node_id, edge.target.node_id) for edge in second
    }


def test_constrained_linking_models_division_without_invalid_lineage():
    frames = [
        [Detection(1, 0, 0, 0, 0)],
        [Detection(2, 1, 0, -2, 1), Detection(3, 1, 0, 2, 1)],
        [Detection(4, 2, 0, -2, 2), Detection(5, 2, 0, 2, 2)],
    ]
    edges = set(link_constrained(frames, LinkConfig(max_distance=12)))
    assert edges == {(1, 2), (1, 3), (2, 4), (3, 5)}
    assert validate_lineage([item for frame in frames for item in frame], edges) == []
    metrics = combine_metrics(
        [(d.z, d.y, d.x) for frame in frames for d in frame],
        [(d.z, d.y, d.x) for frame in frames for d in frame],
        edges,
        edges,
        tolerance=0,
    )
    assert metrics.edge_precision == metrics.edge_recall == metrics.division_f1 == 1


def test_lineage_validation_rejects_duplicate_parent_and_time_reversal():
    detections = [
        Detection(1, 0, 0, 0, 0),
        Detection(2, 0, 0, 1, 0),
        Detection(3, 1, 0, 0, 0),
    ]
    errors = validate_lineage(detections, {(1, 3), (2, 3), (3, 1)})
    assert "duplicate parent for 3" in errors
    assert "time reversal 3->1" in errors


def test_phase_correlation_recovers_integer_camera_drift():
    reference = np.zeros((8, 12, 12), dtype=float)
    reference[2:5, 4:7, 5:8] = 1
    moving = np.roll(reference, (2, -3, 1), axis=(0, 1, 2))
    assert estimate_phase_shift(reference, moving) == (-2.0, 3.0, -1.0)


def test_spatial_transform_round_trip_and_anisotropy():
    transform = SpatialTransform((2.0, 1.0, 0.5), (-1.0, 3.0, 2.0))
    point = (4.0, -2.0, 10.0)
    normalized = transform.forward(point)
    assert normalized == pytest.approx((6.0, 1.0, 6.0))
    assert transform.inverse(normalized) == pytest.approx(point, abs=1e-12)


def test_empty_frames_and_boundary_coordinates_are_explicit():
    empty = np.zeros((4, 5, 6), dtype=float)
    transforms = estimate_reference_transforms([empty, empty.copy()], (2.0, 1.0, 1.0), (2, 2, 2))
    assert [transform.alignment_shift for transform in transforms] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    assert transforms[0].forward((-1.0, 5.0, 7.0)) == (-2.0, 5.0, 7.0)


def test_build_rows_applies_drift_and_voxel_spacing():
    reference = np.zeros((8, 12, 12), dtype=float)
    reference[2:4, 5:7, 4:6] = 10
    moving = np.roll(reference, (1, 2, -2), axis=(0, 1, 2))
    rows = build_rows(
        "corrected",
        [reference, moving],
        threshold_percentile=99,
        min_voxels=4,
        max_link_distance=3,
        preprocessing={
            "voxel_spacing": [2.0, 1.0, 1.0],
            "max_shift_voxels": [2, 3, 3],
        },
    )
    nodes = [row for row in rows if row["row_type"] == "node"]
    assert [(row["z"], row["y"], row["x"]) for row in nodes] == [
        (5.0, 5.5, 4.5),
        (5.0, 5.5, 4.5),
    ]
    assert sum(row["row_type"] == "edge" for row in rows) == 1


def test_appearance_descriptor_clips_boundaries_and_falls_back_safely():
    volume = np.arange(27, dtype=float).reshape(3, 3, 3)
    descriptor = extract_appearance_descriptor(volume, (0.0, 0.0, 0.0), radius=2)
    assert descriptor is not None
    assert len(descriptor) == 5
    assert np.isfinite(descriptor).all()
    assert extract_appearance_descriptor(np.full((3, 3, 3), np.nan), (1, 1, 1)) is None
    assert extract_appearance_descriptor(volume, (-20, 1, 1)) is None


def test_appearance_and_motion_prevent_crossing_identity_switches():
    frames = [
        [Detection(1, 0, 0, 0, 0, (0,)), Detection(2, 0, 0, 0, 10, (2,))],
        [Detection(3, 1, 0, 0, 4, (0,)), Detection(4, 1, 0, 0, 6, (2,))],
        [Detection(5, 2, 0, 0, 8, (0,)), Detection(6, 2, 0, 0, 2, (2,))],
    ]
    expected = {(1, 3), (2, 4), (3, 5), (4, 6)}
    baseline = set(
        link_constrained(
            frames,
            LinkConfig(max_distance=12, k_neighbors=2, density_weight=0),
        )
    )
    candidate = set(
        link_constrained(
            frames,
            LinkConfig(
                max_distance=12,
                k_neighbors=2,
                density_weight=0,
                appearance_weight=0.2,
                motion_weight=0.25,
            ),
        )
    )
    assert count_identity_switches(baseline, expected) > 0
    assert candidate == expected
    assert count_identity_switches(candidate, expected) == 0


def test_motion_fallback_handles_missing_appearance_and_zero_velocity():
    frames = [
        [Detection(1, 0, 0, 0, 1, (0,))],
        [Detection(2, 1, 0, 0, 2, (0,))],
        [Detection(3, 2, 0, 0, 3, None)],
        [Detection(4, 3, 0, 0, 4, None)],
    ]
    config = LinkConfig(max_distance=4, appearance_weight=0.2, motion_weight=0.5)
    assert set(link_constrained(frames, config)) == {(1, 2), (2, 3), (3, 4)}
    stationary = [
        [Detection(5, 0, 0, 0, 0, None)],
        [Detection(6, 1, 0, 0, 0, None)],
        [Detection(7, 2, 0, 0, 0, None)],
    ]
    assert set(link_constrained(stationary, config)) == {(5, 6), (6, 7)}
