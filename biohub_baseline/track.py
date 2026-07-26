from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class Detection:
    node_id: int
    t: int
    z: float
    y: float
    x: float


def link_nearest(
    detections_by_time: list[list[Detection]], max_distance: float
) -> list[tuple[int, int]]:
    """Greedily link each detection to one unused nearest detection in the prior frame."""
    edges: list[tuple[int, int]] = []
    for previous, current in pairwise(detections_by_time):
        candidates: list[tuple[float, int, int]] = []
        for source_index, source in enumerate(previous):
            source_point = np.array((source.z, source.y, source.x))
            for target_index, target in enumerate(current):
                target_point = np.array((target.z, target.y, target.x))
                candidates.append(
                    (float(np.linalg.norm(source_point - target_point)), source_index, target_index)
                )
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for distance, source_index, target_index in sorted(candidates):
            if distance > max_distance:
                break
            if source_index in used_sources or target_index in used_targets:
                continue
            edges.append((previous[source_index].node_id, current[target_index].node_id))
            used_sources.add(source_index)
            used_targets.add(target_index)
    return edges
