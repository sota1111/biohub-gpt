from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .track import Detection


@dataclass(frozen=True)
class Metrics:
    detection_f1: float
    edge_f1: float
    edge_precision: float
    edge_recall: float
    division_f1: float
    composite: float

    def as_dict(self) -> dict[str, float]:
        return {
            "detection_f1": round(self.detection_f1, 6),
            "edge_f1": round(self.edge_f1, 6),
            "edge_precision": round(self.edge_precision, 6),
            "edge_recall": round(self.edge_recall, 6),
            "division_f1": round(self.division_f1, 6),
            "composite": round(self.composite, 6),
        }


def f1(true_positives: int, predicted: int, expected: int) -> float:
    denominator = predicted + expected
    return 1.0 if denominator == 0 else (2.0 * true_positives) / denominator


def match_points(
    predicted: list[tuple[float, float, float]],
    expected: list[tuple[float, float, float]],
    tolerance: float,
) -> int:
    if not predicted or not expected:
        return 0
    distances = np.linalg.norm(
        np.asarray(predicted)[:, None, :] - np.asarray(expected)[None, :, :], axis=2
    )
    predicted_indices, expected_indices = linear_sum_assignment(distances)
    return sum(
        int(distances[predicted_index, expected_index] <= tolerance)
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
    )


def combine_metrics(
    predicted_points: list[tuple[float, float, float]],
    expected_points: list[tuple[float, float, float]],
    predicted_edges: set[tuple[int, int]],
    expected_edges: set[tuple[int, int]],
    tolerance: float,
) -> Metrics:
    point_matches = match_points(predicted_points, expected_points, tolerance)
    detection_f1 = f1(point_matches, len(predicted_points), len(expected_points))
    edge_f1 = f1(len(predicted_edges & expected_edges), len(predicted_edges), len(expected_edges))
    edge_matches = len(predicted_edges & expected_edges)
    edge_precision = edge_matches / len(predicted_edges) if predicted_edges else 0.0
    edge_recall = edge_matches / len(expected_edges) if expected_edges else 0.0
    predicted_divisions = {
        source
        for source, _ in predicted_edges
        if sum(edge[0] == source for edge in predicted_edges) == 2
    }
    expected_divisions = {
        source
        for source, _ in expected_edges
        if sum(edge[0] == source for edge in expected_edges) == 2
    }
    division_f1 = f1(
        len(predicted_divisions & expected_divisions),
        len(predicted_divisions),
        len(expected_divisions),
    )
    composite = 0.6 * detection_f1 + 0.25 * edge_f1 + 0.15 * division_f1
    return Metrics(
        detection_f1,
        edge_f1,
        edge_precision,
        edge_recall,
        division_f1,
        composite,
    )


def validate_lineage(detections: list[Detection], edges: set[tuple[int, int]]) -> list[str]:
    """Return lineage errors: dangling/time-reversed edges, duplicate parents, cycles."""
    times = {detection.node_id: detection.t for detection in detections}
    errors: list[str] = []
    parents: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for source, target in edges:
        if source not in times or target not in times:
            errors.append(f"dangling edge {source}->{target}")
            continue
        if times[source] >= times[target]:
            errors.append(f"time reversal {source}->{target}")
        if target in parents:
            errors.append(f"duplicate parent for {target}")
        parents[target] = source
        children.setdefault(source, []).append(target)
        if len(children[source]) > 2:
            errors.append(f"more than two children for {source}")
    for start in times:
        seen: set[int] = set()
        node = start
        while node in parents:
            if node in seen:
                errors.append(f"cycle at {node}")
                break
            seen.add(node)
            node = parents[node]
    return sorted(set(errors))


def count_identity_switches(
    predicted_edges: set[tuple[int, int]], expected_edges: set[tuple[int, int]]
) -> int:
    """Count targets assigned to a different identity parent than the reference."""
    expected_parent = {target: source for source, target in expected_edges}
    return sum(
        target in expected_parent and expected_parent[target] != source
        for source, target in predicted_edges
    )
