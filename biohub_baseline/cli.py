from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage

from .detect import detect_centroids
from .evaluate import combine_metrics, count_identity_switches, validate_lineage
from .experiment import deterministic_split, load_json, promotion_decision
from .ngff import open_ngff
from .submission import build_rows, validate_rows, write_submission
from .track import Detection, LinkConfig, link_constrained, link_nearest


def generate(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    rows = []
    for dataset_path in sorted(args.input.glob("*.zarr")):
        try:
            import zarr

            store = zarr.open(dataset_path, mode="r")
            array = store if hasattr(store, "shape") else store["0"]
        except ImportError:
            array = open_ngff(dataset_path)
        rows.extend(
            build_rows(
                dataset_path.stem,
                array,
                config["threshold_percentile"],
                config["min_voxels"],
                config["max_link_distance"],
                config.get("link_model"),
                config.get("detection_model"),
                config.get("preprocessing"),
            )
        )
    write_submission(rows, args.output)
    print(f"wrote {args.output} with {len(rows)} rows")


def validate(args: argparse.Namespace) -> None:
    import csv

    with args.submission.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    identifiers = [int(row["id"]) for row in rows]
    if identifiers != list(range(len(rows))):
        raise ValueError("id must be unique, contiguous, and start at zero")
    print(f"valid submission: {len(rows)} rows")


def split(args: argparse.Namespace) -> None:
    dataset_ids = [line.strip() for line in args.datasets.read_text().splitlines() if line.strip()]
    result = deterministic_split(dataset_ids, args.seed, args.screen_fraction)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def promote(args: argparse.Namespace) -> None:
    candidate, champion, gates = map(load_json, (args.candidate, args.champion, args.gates))
    decision = promotion_decision(candidate, champion, gates)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision))
    if not decision["promote"]:
        raise SystemExit(2)


def evaluate_fixture(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    frames = []
    for x in (4, 5, 6):
        frame = np.zeros((8, 12, 12), dtype=np.float32)
        frame[2:4, 5:7, x : x + 2] = 10
        frames.append(frame)
    rows = build_rows(
        "fixture",
        frames,
        config["threshold_percentile"],
        config["min_voxels"],
        config["max_link_distance"],
        config.get("link_model"),
        config.get("detection_model"),
        config.get("preprocessing"),
    )
    predicted_points = [
        (float(row["z"]), float(row["y"]), float(row["x"]))
        for row in rows
        if row["row_type"] == "node"
    ]
    predicted_edges = {
        (int(row["source_id"]), int(row["target_id"])) for row in rows if row["row_type"] == "edge"
    }
    spacing = np.asarray(config.get("preprocessing", {}).get("voxel_spacing", [1, 1, 1]))
    reference_point = tuple(np.asarray((2.5, 5.5, 4.5)) * spacing)
    expected = (
        [reference_point] * 3
        if config.get("preprocessing")
        else [
            (2.5, 5.5, 4.5),
            (2.5, 5.5, 5.5),
            (2.5, 5.5, 6.5),
        ]
    )
    metrics = combine_metrics(
        predicted_points,
        expected,
        predicted_edges,
        {(1, 2), (2, 3)},
        tolerance=0.01,
    ).as_dict()
    result = {
        "champion_id": config["champion_id"],
        "generated_at": "2026-07-26T00:00:00Z",
        "fixture": "tests deterministic synthetic moving-cell volumes",
        "screen": metrics,
        "confirm": metrics,
        "note": (
            "Synthetic baseline establishes pipeline reproducibility; real-data metrics "
            "replace these after labels are available."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def _lineage_case(offset: float) -> tuple[list[list[Detection]], set[tuple[int, int]]]:
    frames = [
        [Detection(1, 0, 0.0, 0.0, offset)],
        [
            Detection(2, 1, 0.0, -2.0, offset + 1.0),
            Detection(3, 1, 0.0, 2.0, offset + 1.0),
        ],
        [
            Detection(4, 2, 0.0, -2.0, offset + 2.0),
            Detection(5, 2, 0.0, 2.0, offset + 2.0),
        ],
    ]
    return frames, {(1, 2), (1, 3), (2, 4), (3, 5)}


def evaluate_lineage(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    link_config = LinkConfig(max_distance=config["max_link_distance"], **config["link_model"])
    stage_results: dict[str, dict[str, dict[str, float]]] = {}
    integrity: dict[str, list[str]] = {}
    for stage, offset in (("screen", 0.0), ("confirm", 0.5)):
        frames, expected_edges = _lineage_case(offset)
        detections = [detection for frame in frames for detection in frame]
        points = [(detection.z, detection.y, detection.x) for detection in detections]
        baseline_edges = set(link_nearest(frames, config["max_link_distance"]))
        candidate_edges = set(link_constrained(frames, link_config))
        stage_results[stage] = {
            "baseline": combine_metrics(
                points, points, baseline_edges, expected_edges, tolerance=0.0
            ).as_dict(),
            "candidate": combine_metrics(
                points, points, candidate_edges, expected_edges, tolerance=0.0
            ).as_dict(),
        }
        integrity[stage] = validate_lineage(detections, candidate_edges)
    champion = {stage: stage_results[stage]["baseline"] for stage in ("screen", "confirm")}
    candidate = {stage: stage_results[stage]["candidate"] for stage in ("screen", "confirm")}
    decision = promotion_decision(candidate, champion, load_json(args.gates))
    if any(integrity.values()):
        decision = {
            **decision,
            "promote": False,
            "reason": "lineage integrity failed",
        }
    result = {
        "experiment_id": "sot-1990-temporal-lineage-v1",
        "detection_stage": "fixed synthetic detections",
        "config": config["link_model"],
        "stages": stage_results,
        "lineage_errors": integrity,
        "decision": decision,
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-lineage "
            "--output artifacts/sot-1990-lineage-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _render_cells(
    shape: tuple[int, int, int],
    cells: list[tuple[tuple[float, float, float], float]],
    seed: int,
) -> np.ndarray:
    coordinates = np.indices(shape, dtype=float)
    volume = np.random.default_rng(seed).normal(1.0, 0.18, shape)
    for center, amplitude in cells:
        distance_squared = sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
        volume += amplitude * np.exp(-distance_squared / (2 * 1.15**2))
    return volume.astype(np.float32)


def _detection_cases() -> dict[str, list[tuple[np.ndarray, list[tuple[float, float, float]]]]]:
    return {
        "screen": [
            (
                _render_cells(
                    (12, 24, 24),
                    [((4.2, 7.4, 7.6), 8.0), ((7.1, 17.2, 16.8), 4.0)],
                    1989,
                ),
                [(4.2, 7.4, 7.6), (7.1, 17.2, 16.8)],
            ),
            (
                _render_cells(
                    (12, 24, 24),
                    [((5.0, 11.0, 10.0), 7.0), ((5.2, 11.3, 13.1), 6.0)],
                    1990,
                ),
                [(5.0, 11.0, 10.0), (5.2, 11.3, 13.1)],
            ),
        ],
        "confirm": [
            (
                _render_cells(
                    (12, 24, 24),
                    [
                        ((5.0, 12.0, 8.5), 7.0),
                        ((5.1, 9.8, 12.0), 5.5),
                        ((5.0, 14.2, 12.0), 5.5),
                    ],
                    1991,
                ),
                [(5.0, 12.0, 8.5), (5.1, 9.8, 12.0), (5.0, 14.2, 12.0)],
            ),
            (
                _render_cells(
                    (12, 24, 24),
                    [((3.8, 6.2, 17.3), 3.5), ((8.0, 17.0, 6.0), 8.0)],
                    1992,
                ),
                [(3.8, 6.2, 17.3), (8.0, 17.0, 6.0)],
            ),
        ],
    }


def evaluate_detection(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    detector_config = config["detection_model"]
    stage_results: dict[str, dict[str, dict[str, float]]] = {}
    case_results: dict[str, list[dict[str, object]]] = {}
    for stage, cases in _detection_cases().items():
        expected_all: list[tuple[float, float, float]] = []
        baseline_all: list[tuple[float, float, float]] = []
        candidate_all: list[tuple[float, float, float]] = []
        case_results[stage] = []
        for index, (volume, expected) in enumerate(cases):
            offset = np.asarray((index * 100.0, 0.0, 0.0))
            baseline = detect_centroids(
                volume, config["threshold_percentile"], config["min_voxels"]
            )
            candidate = detect_centroids(
                volume,
                config["threshold_percentile"],
                config["min_voxels"],
                detector_config,
            )
            expected_all.extend(tuple(np.asarray(point) + offset) for point in expected)
            baseline_all.extend(tuple(np.asarray(point) + offset) for point in baseline)
            candidate_all.extend(tuple(np.asarray(point) + offset) for point in candidate)
            case_results[stage].append(
                {
                    "case": index,
                    "expected": len(expected),
                    "baseline": len(baseline),
                    "candidate": len(candidate),
                }
            )
        baseline_metrics = combine_metrics(
            baseline_all, expected_all, set(), set(), tolerance=2.0
        ).as_dict()
        candidate_metrics = combine_metrics(
            candidate_all, expected_all, set(), set(), tolerance=2.0
        ).as_dict()
        # The tracking stage is deliberately fixed and excluded from the
        # detection-only promotion score; report it as unchanged evidence.
        for metric in ("edge_f1", "edge_precision", "edge_recall", "division_f1"):
            baseline_metrics[metric] = candidate_metrics[metric] = 1.0
        baseline_metrics["composite"] = round(0.7 * baseline_metrics["detection_f1"] + 0.3, 6)
        candidate_metrics["composite"] = round(0.7 * candidate_metrics["detection_f1"] + 0.3, 6)
        stage_results[stage] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
    champion = {stage: values["baseline"] for stage, values in stage_results.items()}
    candidate = {stage: values["candidate"] for stage, values in stage_results.items()}
    decision = promotion_decision(candidate, champion, load_json(args.gates))
    result = {
        "experiment_id": "sot-1989-adaptive-3d-detection-v1",
        "seed": 1989,
        "tracking_stage": "fixed temporal-lineage-v1 configuration",
        "representative_cases": ["sparse", "dense", "division-neighborhood", "noisy"],
        "config": detector_config,
        "cases": case_results,
        "stages": stage_results,
        "decision": decision,
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-detection "
            "--output artifacts/sot-1989-detection-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _drift_case(
    shifts: list[tuple[int, int, int]], seed: int
) -> tuple[list[np.ndarray], list[tuple[float, float, float]]]:
    shape = (16, 28, 28)
    center = (6.0, 13.0, 13.0)
    coordinates = np.indices(shape, dtype=float)
    distance_squared = sum((coordinates[axis] - center[axis]) ** 2 for axis in range(3))
    reference = 10 * np.exp(-distance_squared / (2 * 1.1**2))
    reference += np.random.default_rng(seed).normal(0.0, 0.01, shape)
    frames = [
        ndimage.shift(reference, shift, order=0, mode="constant", cval=0.0) for shift in shifts
    ]
    expected = [(center[0] * 2.0, center[1], center[2])] * len(frames)
    return [frame.astype(np.float32) for frame in frames], expected


def evaluate_preprocessing(args: argparse.Namespace) -> None:
    """Screen then confirm only the preprocessing candidate on fixed models."""
    config = load_json(args.config)
    candidate_config = config["preprocessing"]
    cases = {
        "screen": [
            _drift_case([(0, 0, 0), (2, -3, 2), (-2, 3, -2)], 2043),
            _drift_case([(0, 0, 0), (1, 4, -3), (-1, -4, 3)], 2044),
        ],
        "confirm": [
            _drift_case([(0, 0, 0), (3, -5, 4), (-3, 5, -4)], 2143),
            _drift_case([(0, 0, 0), (2, 5, 5), (-2, -5, -5)], 2144),
        ],
    }
    stage_results: dict[str, dict[str, dict[str, float]]] = {}
    series_results: dict[str, list[dict[str, object]]] = {}
    for stage, stage_cases in cases.items():
        expected_points: list[tuple[float, float, float]] = []
        baseline_points: list[tuple[float, float, float]] = []
        candidate_points: list[tuple[float, float, float]] = []
        expected_edges: set[tuple[int, int]] = set()
        baseline_edges: set[tuple[int, int]] = set()
        candidate_edges: set[tuple[int, int]] = set()
        series_results[stage] = []
        node_offset = 0
        for index, (frames, expected) in enumerate(stage_cases):
            common = (
                config["threshold_percentile"],
                config["min_voxels"],
                config["max_link_distance"],
                config.get("link_model"),
                config.get("detection_model"),
            )
            baseline_rows = build_rows(f"{stage}-{index}", frames, *common)
            candidate_rows = build_rows(
                f"{stage}-{index}", frames, *common, preprocessing=candidate_config
            )
            spatial_offset = np.asarray((index * 100.0, 0.0, 0.0))

            def points(
                rows: list[dict[str, object]],
                offset: np.ndarray = spatial_offset,
            ) -> list[tuple[float, float, float]]:
                return [
                    tuple(np.asarray((float(row["z"]), float(row["y"]), float(row["x"]))) + offset)
                    for row in rows
                    if row["row_type"] == "node"
                ]

            def edges(
                rows: list[dict[str, object]], offset: int = node_offset
            ) -> set[tuple[int, int]]:
                return {
                    (int(row["source_id"]) + offset, int(row["target_id"]) + offset)
                    for row in rows
                    if row["row_type"] == "edge"
                }

            baseline_points.extend(points(baseline_rows))
            candidate_points.extend(points(candidate_rows))
            expected_points.extend(tuple(np.asarray(point) + spatial_offset) for point in expected)
            expected_edges.update(
                (node_offset + source, node_offset + source + 1)
                for source in range(1, len(expected))
            )
            baseline_edges.update(edges(baseline_rows))
            candidate_edges.update(edges(candidate_rows))
            series_results[stage].append(
                {
                    "series": index,
                    "frames": len(frames),
                    "baseline_edges": sum(row["row_type"] == "edge" for row in baseline_rows),
                    "candidate_edges": sum(row["row_type"] == "edge" for row in candidate_rows),
                }
            )
            node_offset += len(expected)
        stage_results[stage] = {
            "baseline": combine_metrics(
                baseline_points,
                expected_points,
                baseline_edges,
                expected_edges,
                tolerance=1.5,
            ).as_dict(),
            "candidate": combine_metrics(
                candidate_points,
                expected_points,
                candidate_edges,
                expected_edges,
                tolerance=1.5,
            ).as_dict(),
        }
    champion = {stage: result["baseline"] for stage, result in stage_results.items()}
    candidate = {stage: result["candidate"] for stage, result in stage_results.items()}
    decision = promotion_decision(candidate, champion, load_json(args.gates))
    result = {
        "experiment_id": "sot-2043-phase-correlation-anisotropy-v1",
        "seed": 2043,
        "split": {"screen": [2043, 2044], "confirm": [2143, 2144]},
        "fixed_detection_model": config["detection_model"]["name"],
        "fixed_link_model": "temporal-lineage-v1",
        "candidate": candidate_config,
        "series": series_results,
        "stages": stage_results,
        "decision": decision,
        "coordinate_checks": {
            "round_trip_tolerance": 1e-9,
            "boundary_policy": "coordinates may be outside a frame after alignment",
            "empty_frame_policy": "zero alignment shift",
        },
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-preprocessing "
            "--output artifacts/sot-2043-preprocessing-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _identity_case(
    positions: list[tuple[float, float]],
    *,
    missing_appearance_at: set[tuple[int, int]] | None = None,
    y_offset: float = 0.0,
) -> tuple[list[list[Detection]], set[tuple[int, int]]]:
    """Create two labelled trajectories whose nearest coordinates become ambiguous."""
    missing = missing_appearance_at or set()
    frames: list[list[Detection]] = []
    expected: set[tuple[int, int]] = set()
    previous_ids: list[int] = []
    node_id = 1
    for time, (first_x, second_x) in enumerate(positions):
        identities = []
        for identity, (x, appearance) in enumerate(
            ((first_x, (0.1, 0.2, 0.3)), (second_x, (1.4, 0.7, 0.1)))
        ):
            descriptor = None if (time, identity) in missing else appearance
            identities.append(Detection(node_id, time, 0.0, y_offset + identity, x, descriptor))
            if previous_ids:
                expected.add((previous_ids[identity], node_id))
            node_id += 1
        frames.append(identities)
        previous_ids = [item.node_id for item in identities]
    return frames, expected


def _link_feature_cases() -> dict[str, list[tuple[str, list[list[Detection]], set[tuple[int, int]]]]]:
    screen_crossing = _identity_case([(0, 10), (4, 6), (8, 2)])
    screen_crowded = _identity_case([(1, 9), (4.2, 5.8), (8.5, 1.5)], y_offset=0.2)
    confirm_crossing = _identity_case([(0.5, 10.5), (4.5, 6.5), (8.5, 2.5)])
    confirm_missing = _identity_case(
        [(0, 12), (3, 9), (6, 6.5), (9, 3.5)],
        missing_appearance_at={(2, 0), (2, 1), (3, 0), (3, 1)},
    )
    confirm_zero = _identity_case([(2, 8), (2, 8), (2, 8)])
    return {
        "screen": [
            ("crossing", *screen_crossing),
            ("crowded", *screen_crowded),
        ],
        "confirm": [
            ("crossing", *confirm_crossing),
            ("temporary-missing-appearance", *confirm_missing),
            ("zero-motion", *confirm_zero),
        ],
    }


def _score_link_cases(
    cases: list[tuple[str, list[list[Detection]], set[tuple[int, int]]]],
    config: LinkConfig,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    predicted_all: set[tuple[int, int]] = set()
    expected_all: set[tuple[int, int]] = set()
    case_results: list[dict[str, object]] = []
    offset = 0
    for name, frames, expected in cases:
        detections = [item for frame in frames for item in frame]
        predicted = set(link_constrained(frames, config))
        shifted_predicted = {(source + offset, target + offset) for source, target in predicted}
        shifted_expected = {(source + offset, target + offset) for source, target in expected}
        predicted_all.update(shifted_predicted)
        expected_all.update(shifted_expected)
        errors = validate_lineage(detections, predicted)
        case_results.append(
            {
                "case": name,
                "edge_f1": combine_metrics([], [], predicted, expected, 0).edge_f1,
                "identity_switches": count_identity_switches(predicted, expected),
                "lineage_errors": errors,
            }
        )
        offset += len(detections)
    metrics = combine_metrics([], [], predicted_all, expected_all, 0).as_dict()
    metrics["detection_f1"] = 1.0
    metrics["composite"] = round(0.7 + 0.3 * metrics["edge_f1"], 6)
    return metrics, case_results


def evaluate_link_features(args: argparse.Namespace) -> None:
    """Screen appearance/motion weights, then confirm only the top candidate."""
    config = load_json(args.config)
    base_values = dict(config["link_model"])
    for key in ("appearance_weight", "motion_weight", "acceleration_weight"):
        base_values.pop(key, None)
    baseline_config = LinkConfig(max_distance=config["max_link_distance"], **base_values)
    candidates = [
        {"appearance_weight": 0.1, "motion_weight": 0.25, "acceleration_weight": 0.0},
        {"appearance_weight": 0.2, "motion_weight": 0.25, "acceleration_weight": 0.0},
        {"appearance_weight": 0.1, "motion_weight": 0.5, "acceleration_weight": 0.5},
    ]
    cases = _link_feature_cases()
    baseline_screen, baseline_screen_cases = _score_link_cases(cases["screen"], baseline_config)
    screen_results = []
    for weights in candidates:
        candidate_config = LinkConfig(
            max_distance=config["max_link_distance"], **base_values, **weights
        )
        metrics, case_results = _score_link_cases(cases["screen"], candidate_config)
        screen_results.append({"weights": weights, "metrics": metrics, "cases": case_results})
    top = max(
        screen_results,
        key=lambda item: (
            item["metrics"]["composite"],
            -sum(item["weights"].values()),
        ),
    )
    top_config = LinkConfig(
        max_distance=config["max_link_distance"], **base_values, **top["weights"]
    )
    baseline_confirm, baseline_confirm_cases = _score_link_cases(
        cases["confirm"], baseline_config
    )
    candidate_confirm, candidate_confirm_cases = _score_link_cases(cases["confirm"], top_config)
    champion = {"screen": baseline_screen, "confirm": baseline_confirm}
    candidate = {"screen": top["metrics"], "confirm": candidate_confirm}
    decision = promotion_decision(candidate, champion, load_json(args.gates))
    result = {
        "experiment_id": "sot-2044-appearance-motion-v1",
        "seed": 2044,
        "split": {
            "screen": ["crossing", "crowded"],
            "confirm": ["crossing", "temporary-missing-appearance", "zero-motion"],
        },
        "fixed_champion": config["champion_id"],
        "screen_candidates": screen_results,
        "top_candidate": top["weights"],
        "stages": {
            "screen": {
                "baseline": baseline_screen,
                "candidate": top["metrics"],
                "baseline_cases": baseline_screen_cases,
                "candidate_cases": top["cases"],
            },
            "confirm": {
                "baseline": baseline_confirm,
                "candidate": candidate_confirm,
                "baseline_cases": baseline_confirm_cases,
                "candidate_cases": candidate_confirm_cases,
            },
        },
        "fallback_checks": {
            "missing_appearance": "coordinate and motion costs remain active",
            "boundary_patch": "descriptor clips patch bounds",
            "zero_motion": "finite zero-velocity prediction",
        },
        "decision": decision,
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-link-features "
            "--output artifacts/sot-2044-appearance-motion-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _profile_link_cases(
    cases: list[tuple[str, list[list[Detection]], set[tuple[int, int]]]],
    config: LinkConfig,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, float]]:
    tracemalloc.start()
    started = perf_counter()
    metrics, series = _score_link_cases(cases, config)
    elapsed = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return metrics, series, {
        "runtime_seconds": round(elapsed, 6),
        "peak_memory_bytes": peak,
    }


def evaluate_calibration_ensemble(args: argparse.Namespace) -> None:
    """Screen rank/temperature candidates, then confirm only the best eligible candidate."""
    config = load_json(args.config)
    base_values = dict(config["link_model"])
    champion_config = LinkConfig(max_distance=config["max_link_distance"], **base_values)
    candidates = [
        {"calibration": "rank", "calibration_temperature": 0.75},
        {"calibration": "rank", "calibration_temperature": 1.0},
        {"calibration": "rank", "calibration_temperature": 1.5},
    ]
    cases = _link_feature_cases()
    champion_screen, champion_screen_series, champion_screen_resources = _profile_link_cases(
        cases["screen"], champion_config
    )
    screen_results = []
    for calibration in candidates:
        candidate_config = LinkConfig(
            max_distance=config["max_link_distance"], **base_values, **calibration
        )
        metrics, series, resources = _profile_link_cases(cases["screen"], candidate_config)
        screen_results.append(
            {
                "config": calibration,
                "metrics": metrics,
                "series": series,
                "resources": resources,
            }
        )
    top = max(
        screen_results,
        key=lambda item: (
            item["metrics"]["composite"],
            -item["config"]["calibration_temperature"],
        ),
    )
    gates = load_json(args.gates)
    screen_pass = (
        top["metrics"]["composite"]
        >= champion_screen["composite"] + gates["screen_min_delta"]
        and top["metrics"]["detection_f1"] >= gates["min_detection_f1"]
        and top["metrics"]["edge_f1"] >= gates["min_edge_f1"]
    )
    confirm_result: dict[str, object] = {
        "evaluated": False,
        "reason": "no candidate passed the screen gate",
    }
    decision = {
        "screen_pass": False,
        "confirm_evaluated": False,
        "promote": False,
        "reason": "screen gates failed",
    }
    regression_tolerance = 0.0
    series_regressions = [
        {
            "series": candidate["case"],
            "champion_edge_f1": champion["edge_f1"],
            "candidate_edge_f1": candidate["edge_f1"],
        }
        for champion, candidate in zip(champion_screen_series, top["series"])
        if candidate["edge_f1"] + regression_tolerance < champion["edge_f1"]
    ]
    if screen_pass:
        top_config = LinkConfig(
            max_distance=config["max_link_distance"], **base_values, **top["config"]
        )
        champion_confirm, champion_confirm_series, champion_confirm_resources = (
            _profile_link_cases(cases["confirm"], champion_config)
        )
        candidate_confirm, candidate_confirm_series, candidate_confirm_resources = (
            _profile_link_cases(cases["confirm"], top_config)
        )
        decision = promotion_decision(
            {"screen": top["metrics"], "confirm": candidate_confirm},
            {"screen": champion_screen, "confirm": champion_confirm},
            gates,
        )
        series_regressions = [
            {
                "series": candidate["case"],
                "champion_edge_f1": champion["edge_f1"],
                "candidate_edge_f1": candidate["edge_f1"],
            }
            for champion, candidate in zip(champion_confirm_series, candidate_confirm_series)
            if candidate["edge_f1"] + regression_tolerance < champion["edge_f1"]
        ]
        if series_regressions:
            decision = {**decision, "promote": False, "reason": "confirm series regression"}
        confirm_result = {
            "evaluated": True,
            "champion": champion_confirm,
            "candidate": candidate_confirm,
            "champion_series": champion_confirm_series,
            "candidate_series": candidate_confirm_series,
            "champion_resources": champion_confirm_resources,
            "candidate_resources": candidate_confirm_resources,
        }
    result = {
        "experiment_id": "sot-2045-calibration-ensemble-v1",
        "seed": 2045,
        "split": {
            "screen": ["crossing", "crowded"],
            "confirm": ["crossing", "temporary-missing-appearance", "zero-motion"],
            "disjoint": True,
        },
        "fixed_champion": config["champion_id"],
        "ensemble": {
            "components": ["coordinate", "density", "appearance", "motion"],
            "weights": {
                "coordinate": 1.0,
                "density": base_values["density_weight"],
                "appearance": base_values["appearance_weight"],
                "motion": base_values["motion_weight"],
            },
            "missing_component_policy": "renormalize over available components",
        },
        "screen_candidates": screen_results,
        "top_candidate": top["config"],
        "stages": {
            "screen": {
                "champion": champion_screen,
                "candidate": top["metrics"],
                "champion_series": champion_screen_series,
                "candidate_series": top["series"],
                "champion_resources": champion_screen_resources,
                "candidate_resources": top["resources"],
            },
            "confirm": confirm_result,
        },
        "series_regressions": series_regressions,
        "decision": decision,
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-calibration-ensemble "
            "--output artifacts/sot-2045-calibration-ensemble-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _touching_case(
    centers: list[tuple[float, float, float]],
    seed: int,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    coordinates = np.indices((14, 28, 28), dtype=float)
    volume = np.random.default_rng(seed).normal(0.0, 0.025, coordinates.shape[1:])
    for index, center in enumerate(centers):
        physical_distance = (
            ((coordinates[0] - center[0]) * 2.0) ** 2
            + (coordinates[1] - center[1]) ** 2
            + (coordinates[2] - center[2]) ** 2
        )
        volume += (7.0 - 0.3 * index) * np.exp(-physical_distance / (2 * 1.3**2))
    return volume.astype(np.float32), centers


def _touching_cases() -> dict[str, list[tuple[str, np.ndarray, list[tuple[float, float, float]]]]]:
    return {
        "screen": [
            ("two-x", *_touching_case([(5.0, 12.0, 10.0), (5.0, 12.0, 12.6)], 2169)),
            ("two-y", *_touching_case([(5.0, 10.0, 12.0), (5.0, 12.6, 12.0)], 2170)),
            ("isolated", *_touching_case([(5.2, 12.0, 12.0)], 2171)),
        ],
        "confirm": [
            (
                "three-cluster",
                *_touching_case(
                    [(5.0, 12.0, 9.8), (5.0, 10.1, 12.0), (5.0, 12.5, 12.2)],
                    2269,
                ),
            ),
            ("anisotropic-z", *_touching_case([(4.0, 12.0, 12.0), (5.4, 12.0, 12.0)], 2270)),
            ("confirm-isolated", *_touching_case([(5.0, 11.5, 12.5)], 2271)),
        ],
    }


def _profile_detector(
    cases: list[tuple[str, np.ndarray, list[tuple[float, float, float]]]],
    config: dict[str, object],
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, float | int]]:
    predicted_all: list[tuple[float, float, float]] = []
    expected_all: list[tuple[float, float, float]] = []
    case_results: list[dict[str, object]] = []
    over_split = 0
    under_split = 0
    tracemalloc.start()
    started = perf_counter()
    for index, (name, volume, expected) in enumerate(cases):
        predicted = detect_centroids(volume, 99.0, 4, config)
        offset = np.asarray((index * 100.0, 0.0, 0.0))
        predicted_all.extend(tuple(np.asarray(point) + offset) for point in predicted)
        expected_all.extend(tuple(np.asarray(point) + offset) for point in expected)
        over_split += max(0, len(predicted) - len(expected))
        under_split += max(0, len(expected) - len(predicted))
        case_results.append(
            {
                "case": name,
                "expected_instances": len(expected),
                "detected_instances": len(predicted),
                "over_split": max(0, len(predicted) - len(expected)),
                "under_split": max(0, len(expected) - len(predicted)),
            }
        )
    elapsed = perf_counter() - started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = combine_metrics(predicted_all, expected_all, set(), set(), tolerance=1.25).as_dict()
    for metric in ("edge_f1", "edge_precision", "edge_recall", "division_f1"):
        metrics[metric] = 1.0
    metrics["composite"] = round(0.7 * metrics["detection_f1"] + 0.3, 6)
    return metrics, case_results, {
        "runtime_seconds": round(elapsed, 6),
        "peak_memory_bytes": peak_memory,
        "over_split": over_split,
        "under_split": under_split,
    }


def evaluate_instance_separation(args: argparse.Namespace) -> None:
    """Small-N screen watershed settings, then confirm only the top eligible candidate."""
    champion = load_json(args.config)
    incumbent = dict(champion["detection_model"])
    base_candidate = {
        "name": "touching-watershed-v1",
        "threshold_percentile": 75.0,
        "local_sigma": 2.0,
        "local_offset": 0.5,
        "min_component_voxels": 8,
        "min_instance_voxels": 2,
        "max_component_voxels": 4096,
        "min_peak_distance": 0.5,
        "min_component_intensity_ratio": 0.1,
        "voxel_spacing": champion["preprocessing"]["voxel_spacing"],
    }
    candidates = [
        {**base_candidate, "marker_distance": marker_distance, "separation_confidence": confidence}
        for marker_distance, confidence in ((1.25, 0.3), (1.5, 0.4), (2.0, 0.5))
    ]
    cases = _touching_cases()
    incumbent_screen = _profile_detector(cases["screen"], incumbent)
    screen_results = []
    for candidate in candidates:
        metrics, case_results, resources = _profile_detector(cases["screen"], candidate)
        screen_results.append(
            {
                "config": candidate,
                "metrics": metrics,
                "cases": case_results,
                "resources": resources,
            }
        )
    top = max(
        screen_results,
        key=lambda item: (
            item["metrics"]["composite"],
            -item["resources"]["over_split"],
            -item["resources"]["under_split"],
            -item["resources"]["runtime_seconds"],
        ),
    )
    gates = load_json(args.gates)
    screen_pass = (
        top["metrics"]["composite"]
        >= incumbent_screen[0]["composite"] + gates["screen_min_delta"]
        and top["metrics"]["detection_f1"] >= gates["min_detection_f1"]
        and top["resources"]["over_split"] <= incumbent_screen[2]["over_split"]
    )
    incumbent_confirm = _profile_detector(cases["confirm"], incumbent)
    candidate_confirm = _profile_detector(cases["confirm"], top["config"])
    confirm: dict[str, object] = {
        "evaluated": True,
        "incumbent": {
            "metrics": incumbent_confirm[0],
            "cases": incumbent_confirm[1],
            "resources": incumbent_confirm[2],
        },
        "candidate": {
            "metrics": candidate_confirm[0],
            "cases": candidate_confirm[1],
            "resources": candidate_confirm[2],
        },
    }
    decision = {
        "screen_pass": False,
        "confirm_evaluated": True,
        "promote": False,
        "reason": "screen gates failed; confirm recorded for independent evidence",
    }
    if screen_pass:
        decision = promotion_decision(
            {"screen": top["metrics"], "confirm": candidate_confirm[0]},
            {"screen": incumbent_screen[0], "confirm": incumbent_confirm[0]},
            gates,
        )
        if candidate_confirm[2]["over_split"] > incumbent_confirm[2]["over_split"]:
            decision = {**decision, "promote": False, "reason": "confirm over-splitting regression"}
    result = {
        "experiment_id": "sot-2169-touching-watershed-v1",
        "seed": 2169,
        "split": {
            "screen": [name for name, _, _ in cases["screen"]],
            "confirm": [name for name, _, _ in cases["confirm"]],
            "disjoint": True,
        },
        "fixed_champion": champion["champion_id"],
        "incumbent_detection_model": incumbent,
        "screen": {
            "incumbent": {
                "metrics": incumbent_screen[0],
                "cases": incumbent_screen[1],
                "resources": incumbent_screen[2],
            },
            "candidates": screen_results,
            "top_candidate": top["config"],
        },
        "confirm": confirm,
        "decision": decision,
        "provenance": {
            "fixture": "deterministic anisotropic synthetic touching-nuclei volumes",
            "downstream_contract": "z/y/x centroids consumed unchanged by division/link evaluation",
        },
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-instance-separation "
            "--output artifacts/sot-2169-instance-separation-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def _division_case(
    name: str,
    *,
    source: tuple[float, float, float],
    daughters: tuple[tuple[float, float, float], tuple[float, float, float]],
    distractor: tuple[float, float, float],
    source_volume: float,
    daughter_volumes: tuple[float, float],
) -> tuple[str, list[list[Detection]], set[tuple[int, int]]]:
    frames = [
        [Detection(1, 0, *source, volume=source_volume)],
        [
            Detection(2, 1, *daughters[0], volume=daughter_volumes[0]),
            Detection(3, 1, *daughters[1], volume=daughter_volumes[1]),
            Detection(4, 1, *distractor, volume=source_volume),
        ],
    ]
    return name, frames, {(1, 2), (1, 3)}


def _division_cases() -> dict[
    str, list[tuple[str, list[list[Detection]], set[tuple[int, int]]]]
]:
    """Disjoint deterministic daughter-pair fixtures with closer non-daughter distractors."""
    return {
        "screen": [
            _division_case(
                "symmetric-x",
                source=(0.0, 0.0, 0.0),
                daughters=((0.0, -2.0, 1.0), (0.0, 2.0, 1.0)),
                distractor=(0.0, 0.0, 0.5),
                source_volume=10.0,
                daughter_volumes=(5.0, 5.0),
            ),
            _division_case(
                "oblique-balanced",
                source=(1.0, 3.0, 2.0),
                daughters=((1.0, 1.2, 3.2), (1.0, 4.9, 3.0)),
                distractor=(1.0, 3.1, 2.4),
                source_volume=12.0,
                daughter_volumes=(6.5, 5.5),
            ),
        ],
        "confirm": [
            _division_case(
                "anisotropic-z",
                source=(4.0, 2.0, 1.0),
                daughters=((2.5, 1.2, 1.8), (5.5, 2.8, 1.7)),
                distractor=(4.1, 2.0, 1.4),
                source_volume=14.0,
                daughter_volumes=(7.5, 6.5),
            ),
            _division_case(
                "unequal-daughters",
                source=(2.0, -1.0, 4.0),
                daughters=((2.0, -3.1, 5.0), (2.0, 0.8, 5.2)),
                distractor=(2.0, -0.9, 4.3),
                source_volume=11.0,
                daughter_volumes=(7.0, 4.0),
            ),
            _division_case(
                "translated-y",
                source=(0.5, 8.0, -2.0),
                daughters=((0.5, 5.8, -1.0), (0.5, 10.1, -0.8)),
                distractor=(0.5, 8.0, -1.6),
                source_volume=9.0,
                daughter_volumes=(4.0, 5.0),
            ),
        ],
    }


def _jaccard(predicted: set[tuple[int, int]], expected: set[tuple[int, int]]) -> float:
    union = predicted | expected
    return 1.0 if not union else len(predicted & expected) / len(union)


def _score_division_cases(
    cases: list[tuple[str, list[list[Detection]], set[tuple[int, int]]]],
    config: LinkConfig,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, float]]:
    started = perf_counter()
    predicted_all: set[tuple[int, int]] = set()
    expected_all: set[tuple[int, int]] = set()
    details: list[dict[str, object]] = []
    offset = 0
    for name, frames, expected in cases:
        predicted = set(link_constrained(frames, config))
        predicted_division = (
            predicted if len({target for source, target in predicted if source == 1}) == 2 else set()
        )
        shifted_predicted = {(source + offset, target + offset) for source, target in predicted}
        shifted_expected = {(source + offset, target + offset) for source, target in expected}
        predicted_all.update(shifted_predicted)
        expected_all.update(shifted_expected)
        details.append(
            {
                "case": name,
                "predicted_edges": sorted([list(edge) for edge in predicted]),
                "expected_edges": sorted([list(edge) for edge in expected]),
                "division_jaccard": round(_jaccard(predicted_division, expected), 6),
                "edge_jaccard": round(_jaccard(predicted, expected), 6),
                "false_division_edges": len(predicted_division - expected),
                "missed_division_edges": len(expected - predicted_division),
                "lineage_errors": validate_lineage(
                    [detection for frame in frames for detection in frame], predicted
                ),
            }
        )
        offset += sum(len(frame) for frame in frames)
    division_predicted = {
        edge
        for edge in predicted_all
        if sum(candidate[0] == edge[0] for candidate in predicted_all) == 2
    }
    metrics = {
        "division_jaccard": round(_jaccard(division_predicted, expected_all), 6),
        "edge_jaccard": round(_jaccard(predicted_all, expected_all), 6),
        "false_division_edges": len(division_predicted - expected_all),
        "missed_division_edges": len(expected_all - division_predicted),
        "runtime_seconds": round(perf_counter() - started, 6),
    }
    return metrics, details, {
        "detection_f1": 1.0,
        "edge_f1": round(
            combine_metrics([], [], predicted_all, expected_all, tolerance=0).edge_f1, 6
        ),
        "composite": round(
            0.6
            + 0.25 * combine_metrics([], [], predicted_all, expected_all, tolerance=0).edge_f1
            + 0.15 * _jaccard(division_predicted, expected_all),
            6,
        ),
    }


def evaluate_division_geometry(args: argparse.Namespace) -> None:
    """Small-N screen daughter geometry, then confirm only the top configuration."""
    champion = load_json(args.config)
    incumbent_values = dict(champion["link_model"])
    for key in (
        "division_time_window",
        "division_volume_weight",
        "division_balance_weight",
        "division_opposition_weight",
        "division_midpoint_weight",
        "division_max_volume_error",
    ):
        incumbent_values.pop(key, None)
    incumbent = LinkConfig(max_distance=champion["max_link_distance"], **incumbent_values)
    candidates = [
        {
            "division_time_window": 1,
            "division_volume_weight": weight,
            "division_balance_weight": weight,
            "division_opposition_weight": weight,
            "division_midpoint_weight": weight,
            "division_max_volume_error": volume_error,
        }
        for weight, volume_error in ((0.05, 0.25), (0.1, 0.35), (0.2, 0.5))
    ]
    cases = _division_cases()
    incumbent_screen = _score_division_cases(cases["screen"], incumbent)
    screen_results = []
    for candidate in candidates:
        candidate_config = LinkConfig(**{**incumbent.__dict__, **candidate})
        score = _score_division_cases(cases["screen"], candidate_config)
        screen_results.append(
            {"config": candidate, "diagnostics": score[0], "cases": score[1], "metrics": score[2]}
        )
    top = max(
        screen_results,
        key=lambda item: (
            item["diagnostics"]["division_jaccard"],
            item["diagnostics"]["edge_jaccard"],
            -item["diagnostics"]["false_division_edges"],
            -item["config"]["division_volume_weight"],
        ),
    )
    top_config = LinkConfig(**{**incumbent.__dict__, **top["config"]})
    incumbent_confirm = _score_division_cases(cases["confirm"], incumbent)
    candidate_confirm = _score_division_cases(cases["confirm"], top_config)
    gates = load_json(args.gates)
    decision = promotion_decision(
        {"screen": top["metrics"], "confirm": candidate_confirm[2]},
        {"screen": incumbent_screen[2], "confirm": incumbent_confirm[2]},
        gates,
    )
    if (
        candidate_confirm[0]["division_jaccard"]
        <= incumbent_confirm[0]["division_jaccard"]
        or candidate_confirm[0]["false_division_edges"]
        > incumbent_confirm[0]["false_division_edges"]
        or candidate_confirm[0]["missed_division_edges"]
        > incumbent_confirm[0]["missed_division_edges"]
    ):
        decision = {**decision, "promote": False, "reason": "division diagnostics did not improve"}
    result = {
        "experiment_id": "sot-2170-daughter-geometry-v1",
        "seed": 2170,
        "fixed_champion": "appearance-motion-link-v1",
        "split": {
            "screen": [case[0] for case in cases["screen"]],
            "confirm": [case[0] for case in cases["confirm"]],
            "disjoint": True,
        },
        "features": [
            "adjacent-frame time window",
            "daughter distance and opposition",
            "daughter-pair midpoint",
            "mother-to-daughter volume conservation",
            "daughter volume balance",
        ],
        "screen": {
            "incumbent": {
                "diagnostics": incumbent_screen[0],
                "cases": incumbent_screen[1],
                "metrics": incumbent_screen[2],
            },
            "candidates": screen_results,
            "top_candidate": top["config"],
        },
        "confirm": {
            "incumbent": {
                "diagnostics": incumbent_confirm[0],
                "cases": incumbent_confirm[1],
                "metrics": incumbent_confirm[2],
            },
            "candidate": {
                "diagnostics": candidate_confirm[0],
                "cases": candidate_confirm[1],
                "metrics": candidate_confirm[2],
            },
        },
        "stratification": {
            "touching_nuclei_separation": {
                "enabled": False,
                "reason": "SOT-2169 candidate was not promoted; active detector remains fixed",
            },
            "candidate_detector_contract": "volume-aware Detection is optional and centroid-compatible",
        },
        "decision": decision,
        "provenance": {
            "fixture": "deterministic daughter-pair geometry with disjoint screen/confirm cases",
            "incumbent_config": "config/champion.json at SOT-2169 terminal commit b0aed73",
        },
        "reproduce": (
            "python -m biohub_baseline.cli evaluate-division-geometry "
            "--output artifacts/sot-2170-division-geometry-experiment.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not decision["promote"]:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(required=True)
    generate_command = commands.add_parser("generate")
    generate_command.add_argument("--input", type=Path, required=True)
    generate_command.add_argument("--output", type=Path, default=Path("submission.csv"))
    generate_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    generate_command.set_defaults(function=generate)
    validate_command = commands.add_parser("validate")
    validate_command.add_argument("submission", type=Path)
    validate_command.set_defaults(function=validate)
    split_command = commands.add_parser("split")
    split_command.add_argument("--datasets", type=Path, required=True)
    split_command.add_argument("--output", type=Path, required=True)
    split_command.add_argument("--seed", type=int, default=1988)
    split_command.add_argument("--screen-fraction", type=float, default=0.4)
    split_command.set_defaults(function=split)
    promote_command = commands.add_parser("promote")
    promote_command.add_argument("--candidate", type=Path, required=True)
    promote_command.add_argument("--champion", type=Path, required=True)
    promote_command.add_argument("--gates", type=Path, default=Path("config/evaluation-gates.json"))
    promote_command.add_argument("--output", type=Path, default=Path("artifacts/promotion.json"))
    promote_command.set_defaults(function=promote)
    evaluate_command = commands.add_parser("evaluate-fixture")
    evaluate_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    evaluate_command.add_argument("--output", type=Path, required=True)
    evaluate_command.set_defaults(function=evaluate_fixture)
    lineage_command = commands.add_parser("evaluate-lineage")
    lineage_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    lineage_command.add_argument("--gates", type=Path, default=Path("config/evaluation-gates.json"))
    lineage_command.add_argument(
        "--output", type=Path, default=Path("artifacts/sot-1990-lineage-experiment.json")
    )
    lineage_command.set_defaults(function=evaluate_lineage)
    detection_command = commands.add_parser("evaluate-detection")
    detection_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    detection_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    detection_command.add_argument(
        "--output", type=Path, default=Path("artifacts/sot-1989-detection-experiment.json")
    )
    detection_command.set_defaults(function=evaluate_detection)
    preprocessing_command = commands.add_parser("evaluate-preprocessing")
    preprocessing_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    preprocessing_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    preprocessing_command.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sot-2043-preprocessing-experiment.json"),
    )
    preprocessing_command.set_defaults(function=evaluate_preprocessing)
    link_features_command = commands.add_parser("evaluate-link-features")
    link_features_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    link_features_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    link_features_command.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sot-2044-appearance-motion-experiment.json"),
    )
    link_features_command.set_defaults(function=evaluate_link_features)
    calibration_command = commands.add_parser("evaluate-calibration-ensemble")
    calibration_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    calibration_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    calibration_command.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sot-2045-calibration-ensemble-experiment.json"),
    )
    calibration_command.set_defaults(function=evaluate_calibration_ensemble)
    separation_command = commands.add_parser("evaluate-instance-separation")
    separation_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    separation_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    separation_command.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sot-2169-instance-separation-experiment.json"),
    )
    separation_command.set_defaults(function=evaluate_instance_separation)
    division_command = commands.add_parser("evaluate-division-geometry")
    division_command.add_argument("--config", type=Path, default=Path("config/champion.json"))
    division_command.add_argument(
        "--gates", type=Path, default=Path("config/evaluation-gates.json")
    )
    division_command.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sot-2170-division-geometry-experiment.json"),
    )
    division_command.set_defaults(function=evaluate_division_geometry)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
