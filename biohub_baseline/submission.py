from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .detect import detect_centroids
from .track import Detection, LinkConfig, link_constrained, link_nearest

SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def build_rows(
    dataset: str,
    frames: Iterable[np.ndarray],
    threshold_percentile: float,
    min_voxels: int,
    max_link_distance: float,
    link_config: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    detections_by_time: list[list[Detection]] = []
    next_node_id = 1
    for time, frame in enumerate(frames):
        frame_detections = []
        for z, y, x in detect_centroids(frame, threshold_percentile, min_voxels):
            frame_detections.append(Detection(next_node_id, time, z, y, x))
            next_node_id += 1
        detections_by_time.append(frame_detections)

    rows: list[dict[str, object]] = []
    for frame_detections in detections_by_time:
        for detection in frame_detections:
            rows.append(
                {
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": detection.node_id,
                    "t": detection.t,
                    "z": round(detection.z, 4),
                    "y": round(detection.y, 4),
                    "x": round(detection.x, 4),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
    if link_config is None:
        edges = link_nearest(detections_by_time, max_link_distance)
    else:
        edges = link_constrained(
            detections_by_time,
            LinkConfig(max_distance=max_link_distance, **link_config),
        )
    for source, target in edges:
        rows.append(
            {
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source,
                "target_id": target,
            }
        )
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("submission has no rows")
    node_ids_by_dataset: dict[str, set[int]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        if row["row_type"] == "node":
            node_id = int(row["node_id"])
            if node_id <= 0:
                raise ValueError("node_id must be positive")
            nodes = node_ids_by_dataset.setdefault(dataset, set())
            if node_id in nodes:
                raise ValueError(f"duplicate node_id {node_id} in {dataset}")
            nodes.add(node_id)
        elif row["row_type"] != "edge":
            raise ValueError(f"invalid row_type: {row['row_type']}")
    for row in rows:
        if row["row_type"] != "edge":
            continue
        nodes = node_ids_by_dataset.get(str(row["dataset"]), set())
        source, target = int(row["source_id"]), int(row["target_id"])
        if source not in nodes or target not in nodes:
            raise ValueError(f"edge references missing node: {source}->{target}")
        if source == target:
            raise ValueError("self edges are not valid")


def write_submission(rows: list[dict[str, object]], output: Path) -> None:
    validate_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for identifier, row in enumerate(rows):
            writer.writerow({"id": identifier, **row})
