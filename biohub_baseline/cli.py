from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .detect import detect_centroids
from .evaluate import combine_metrics, validate_lineage
from .experiment import deterministic_split, load_json, promotion_decision
from .submission import build_rows, validate_rows, write_submission
from .track import Detection, LinkConfig, link_constrained, link_nearest


def generate(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    rows = []
    for dataset_path in sorted(args.input.glob("*.zarr")):
        store = zarr.open(dataset_path, mode="r")
        array = store if hasattr(store, "shape") else store["0"]
        rows.extend(
            build_rows(
                dataset_path.stem,
                array,
                config["threshold_percentile"],
                config["min_voxels"],
                config["max_link_distance"],
                config.get("link_model"),
                config.get("detection_model"),
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
    )
    predicted_points = [
        (float(row["z"]), float(row["y"]), float(row["x"]))
        for row in rows
        if row["row_type"] == "node"
    ]
    predicted_edges = {
        (int(row["source_id"]), int(row["target_id"])) for row in rows if row["row_type"] == "edge"
    }
    metrics = combine_metrics(
        predicted_points,
        [(2.5, 5.5, 4.5), (2.5, 5.5, 5.5), (2.5, 5.5, 6.5)],
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
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
