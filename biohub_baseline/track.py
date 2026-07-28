from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Detection:
    node_id: int
    t: int
    z: float
    y: float
    x: float
    appearance: tuple[float, ...] | None = None


@dataclass(frozen=True)
class LinkConfig:
    max_distance: float
    k_neighbors: int = 3
    density_radius: float = 8.0
    density_weight: float = 0.15
    birth_cost: float = 1.0
    death_cost: float = 1.0
    division_cost: float = 0.25
    division_max_distance: float = 8.0
    division_min_separation: float = 1.0
    division_max_separation: float = 10.0
    appearance_weight: float = 0.0
    motion_weight: float = 0.0
    acceleration_weight: float = 0.0


@dataclass(frozen=True)
class CandidateEdge:
    source: Detection
    target: Detection
    distance: float
    cost: float
    appearance_distance: float | None = None
    motion_distance: float | None = None


def _point(detection: Detection) -> np.ndarray:
    return np.asarray((detection.z, detection.y, detection.x), dtype=float)


def _local_density(detections: list[Detection], radius: float) -> dict[int, int]:
    result: dict[int, int] = {}
    for detection in detections:
        result[detection.node_id] = sum(
            0 < float(np.linalg.norm(_point(detection) - _point(other))) <= radius
            for other in detections
        )
    return result


def build_candidate_edges(
    previous: list[Detection],
    current: list[Detection],
    config: LinkConfig,
    predictions: dict[int, np.ndarray] | None = None,
) -> list[CandidateEdge]:
    """Build a deterministic sparse mutual-kNN graph with density-aware costs."""
    if not previous or not current:
        return []
    distances = np.asarray(
        [
            [float(np.linalg.norm(_point(source) - _point(target))) for target in current]
            for source in previous
        ]
    )
    k = max(1, config.k_neighbors)
    source_neighbors = {
        (source_index, target_index)
        for source_index in range(len(previous))
        for target_index in np.argsort(distances[source_index], kind="stable")[:k]
    }
    target_neighbors = {
        (source_index, target_index)
        for target_index in range(len(current))
        for source_index in np.argsort(distances[:, target_index], kind="stable")[:k]
    }
    previous_density = _local_density(previous, config.density_radius)
    current_density = _local_density(current, config.density_radius)
    candidates = []
    for source_index, target_index in sorted(source_neighbors & target_neighbors):
        distance = float(distances[source_index, target_index])
        if distance > config.max_distance:
            continue
        source, target = previous[source_index], current[target_index]
        density_delta = abs(previous_density[source.node_id] - current_density[target.node_id])
        cost = distance / config.max_distance + config.density_weight * density_delta
        appearance_distance = None
        if source.appearance is not None and target.appearance is not None:
            source_descriptor = np.asarray(source.appearance)
            target_descriptor = np.asarray(target.appearance)
            if source_descriptor.shape == target_descriptor.shape:
                appearance_distance = float(np.linalg.norm(source_descriptor - target_descriptor))
                cost += config.appearance_weight * appearance_distance
        motion_distance = None
        if predictions is not None and source.node_id in predictions:
            motion_distance = float(np.linalg.norm(predictions[source.node_id] - _point(target)))
            cost += config.motion_weight * motion_distance / config.max_distance
        candidates.append(
            CandidateEdge(
                source,
                target,
                distance,
                cost,
                appearance_distance,
                motion_distance,
            )
        )
    return sorted(
        candidates, key=lambda edge: (edge.cost, edge.source.node_id, edge.target.node_id)
    )


def _valid_division(first: CandidateEdge, second: CandidateEdge, config: LinkConfig) -> bool:
    if first.source.node_id != second.source.node_id:
        return False
    if max(first.distance, second.distance) > config.division_max_distance:
        return False
    separation = float(np.linalg.norm(_point(first.target) - _point(second.target)))
    return config.division_min_separation <= separation <= config.division_max_separation


def link_constrained(
    detections_by_time: list[list[Detection]], config: LinkConfig
) -> list[tuple[int, int]]:
    """Select continuation/division edges under acyclic lineage constraints.

    Each target has at most one parent and each source has at most two children.
    A second child is accepted only when the configured division gate passes.
    Birth/death costs make the selection objective explicit and reproducible.
    """
    edges: list[tuple[int, int]] = []
    detections = {
        detection.node_id: detection
        for frame in detections_by_time
        for detection in frame
    }
    parent_by_target: dict[int, int] = {}
    for previous, current in pairwise(detections_by_time):
        predictions: dict[int, np.ndarray] = {}
        for source in previous:
            parent_id = parent_by_target.get(source.node_id)
            if parent_id is None:
                continue
            parent = detections[parent_id]
            velocity = _point(source) - _point(parent)
            prediction = _point(source) + velocity
            grandparent_id = parent_by_target.get(parent_id)
            if grandparent_id is not None and config.acceleration_weight > 0:
                old_velocity = _point(parent) - _point(detections[grandparent_id])
                prediction += config.acceleration_weight * (velocity - old_velocity)
            predictions[source.node_id] = prediction
        candidates = build_candidate_edges(previous, current, config, predictions)
        by_source: dict[int, list[CandidateEdge]] = {}
        for edge in candidates:
            by_source.setdefault(edge.source.node_id, []).append(edge)

        # One binary variable per source option (death, continuation, division).
        # Source equality and target upper-bound constraints provide a global
        # optimum without enumerating combinations.
        sources = [source.node_id for source in previous]
        grouped: dict[int, list[tuple[float, tuple[CandidateEdge, ...]]]] = {}
        for source_id in sources:
            source_options = [(config.death_cost, ())]
            source_edges = by_source.get(source_id, [])
            for edge in source_edges:
                source_options.append((edge.cost - config.birth_cost, (edge,)))
            for index, first in enumerate(source_edges):
                for second in source_edges[index + 1 :]:
                    if _valid_division(first, second, config):
                        source_options.append(
                            (
                                first.cost
                                + second.cost
                                + config.division_cost
                                - 2 * config.birth_cost,
                                (first, second),
                            )
                        )
            grouped[source_id] = source_options

        variables = [
            (source_id, option_cost, choice)
            for source_id in sources
            for option_cost, choice in grouped[source_id]
        ]
        target_ids = [target.node_id for target in current]
        constraint = lil_matrix((len(sources) + len(target_ids), len(variables)))
        source_rows = {source_id: index for index, source_id in enumerate(sources)}
        target_rows = {
            target_id: len(sources) + index for index, target_id in enumerate(target_ids)
        }
        for variable_index, (source_id, _, choice) in enumerate(variables):
            constraint[source_rows[source_id], variable_index] = 1
            for edge in choice:
                constraint[target_rows[edge.target.node_id], variable_index] = 1
        lower = np.concatenate((np.ones(len(sources)), np.zeros(len(target_ids))))
        upper = np.ones(len(sources) + len(target_ids))
        objective = np.asarray(
            [
                option_cost + variable_index * 1e-10
                for variable_index, (_, option_cost, _) in enumerate(variables)
            ]
        )
        result = milp(
            objective,
            integrality=np.ones(len(variables)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(constraint.tocsr(), lower, upper),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"lineage optimization failed: {result.message}")
        for selected, (_, _, choice) in zip(result.x > 0.5, variables):
            if selected:
                for edge in choice:
                    link = (edge.source.node_id, edge.target.node_id)
                    edges.append(link)
                    parent_by_target[link[1]] = link[0]
    return edges


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
