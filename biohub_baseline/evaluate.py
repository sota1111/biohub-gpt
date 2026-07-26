from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Metrics:
    detection_f1: float
    edge_f1: float
    composite: float

    def as_dict(self) -> dict[str, float]:
        return {
            "detection_f1": round(self.detection_f1, 6),
            "edge_f1": round(self.edge_f1, 6),
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
        distances[predicted_index, expected_index] <= tolerance
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
    edge_f1 = f1(
        len(predicted_edges & expected_edges), len(predicted_edges), len(expected_edges)
    )
    return Metrics(detection_f1, edge_f1, 0.7 * detection_f1 + 0.3 * edge_f1)
