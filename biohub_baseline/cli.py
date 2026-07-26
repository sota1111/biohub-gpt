from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .evaluate import combine_metrics
from .experiment import deterministic_split, load_json, promotion_decision
from .submission import build_rows, validate_rows, write_submission


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
    )
    predicted_points = [
        (float(row["z"]), float(row["y"]), float(row["x"]))
        for row in rows
        if row["row_type"] == "node"
    ]
    predicted_edges = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in rows
        if row["row_type"] == "edge"
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
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
