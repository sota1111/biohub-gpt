from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .detect import detect_centroids
from .preprocess import SpatialTransform, estimate_phase_shift
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


def extract_appearance_descriptor(
    frame: np.ndarray,
    point: tuple[float, float, float],
    radius: int = 2,
) -> tuple[float, ...] | None:
    """Return a bounded intensity/shape descriptor, or None when no patch is usable."""
    if frame.ndim != 3 or radius < 1 or not np.isfinite(point).all():
        return None
    center = np.rint(point).astype(int)
    starts = np.maximum(center - radius, 0)
    stops = np.minimum(center + radius + 1, frame.shape)
    if np.any(starts >= stops):
        return None
    patch = np.asarray(
        frame[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ],
        dtype=float,
    )
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return None
    mean = float(finite.mean())
    scale = float(finite.std())
    if scale <= 1e-12:
        scale = max(abs(mean), 1.0)
    weights = np.clip(np.nan_to_num(patch, nan=mean) - float(finite.min()), 0.0, None)
    mass = float(weights.sum())
    shape = np.zeros(3)
    if mass > 1e-12:
        coordinates = np.indices(patch.shape, dtype=float)
        centroid = np.asarray([(coordinates[axis] * weights).sum() / mass for axis in range(3)])
        shape = centroid / np.maximum(np.asarray(patch.shape) - 1, 1)
    return tuple(round(value, 6) for value in (mean / scale, scale / max(abs(mean), 1.0), *shape))


def estimate_detection_volume(
    frame: np.ndarray,
    point: tuple[float, float, float],
    threshold_percentile: float,
    radius: int = 3,
) -> float | None:
    """Estimate local above-threshold voxel mass for division volume conservation."""
    if frame.ndim != 3 or radius < 1 or not np.isfinite(point).all():
        return None
    finite = np.asarray(frame)[np.isfinite(frame)]
    if finite.size == 0:
        return None
    center = np.rint(point).astype(int)
    starts = np.maximum(center - radius, 0)
    stops = np.minimum(center + radius + 1, frame.shape)
    if np.any(starts >= stops):
        return None
    patch = np.asarray(
        frame[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ],
        dtype=float,
    )
    threshold = float(np.percentile(finite, threshold_percentile))
    volume = int(np.count_nonzero(np.isfinite(patch) & (patch >= threshold)))
    return float(volume) if volume > 0 else None


def build_rows(
    dataset: str,
    frames: Iterable[np.ndarray],
    threshold_percentile: float,
    min_voxels: int,
    max_link_distance: float,
    link_config: dict[str, object] | None = None,
    detection_model: dict[str, object] | None = None,
    preprocessing: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    frame_source = frames if hasattr(frames, "__getitem__") else list(frames)
    frame_count = int(frame_source.shape[0]) if hasattr(frame_source, "shape") else len(frame_source)
    transforms = [SpatialTransform((1.0, 1.0, 1.0)) for _ in range(frame_count)]
    if preprocessing is not None:
        spacing_values = preprocessing.get("voxel_spacing", [1.0, 1.0, 1.0])
        max_shift_values = preprocessing.get("max_shift_voxels")
        spacing = tuple(float(value) for value in spacing_values)
        max_shift = (
            tuple(int(value) for value in max_shift_values)
            if max_shift_values is not None
            else None
        )
        reference = None
        for time_index in range(frame_count):
            candidate = np.asarray(frame_source[time_index])
            if np.any(candidate):
                reference = candidate
                break
        if reference is None and frame_count:
            reference = np.asarray(frame_source[0])
        transforms = [
            SpatialTransform(
                spacing,
                estimate_phase_shift(reference, np.asarray(frame_source[time_index]), max_shift),
            )
            for time_index in range(frame_count)
        ]
    detections_by_time: list[list[Detection]] = []
    next_node_id = 1
    for time, transform in enumerate(transforms):
        frame = np.asarray(frame_source[time])
        frame_detections = []
        for point in detect_centroids(frame, threshold_percentile, min_voxels, detection_model):
            z, y, x = transform.forward(point)
            appearance = None
            if link_config is not None and float(link_config.get("appearance_weight", 0.0)) > 0:
                appearance = extract_appearance_descriptor(frame, point)
            volume = None
            if link_config is not None and (
                float(link_config.get("division_volume_weight", 0.0)) > 0
                or np.isfinite(float(link_config.get("division_max_volume_error", float("inf"))))
            ):
                volume = estimate_detection_volume(frame, point, threshold_percentile)
            frame_detections.append(
                Detection(next_node_id, time, z, y, x, appearance, volume)
            )
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
