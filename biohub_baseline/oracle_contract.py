"""Oracle round-trip diagnostics for the pinned official graph evaluator."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .real_cv import _jaccard, _rows_to_graph, official_modules, repository_revision

Row = dict[str, object]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash directory content without depending on absolute paths or mtimes."""
    checksum = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        checksum.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
    return checksum.hexdigest()


def graph_to_rows(graph: Any) -> list[Row]:
    """Serialize a tracksdata graph to the competition node/edge row contract."""
    nodes = graph.node_attrs().sort("node_id").to_dicts()
    edges = graph.edge_attrs().sort("edge_id").to_dicts()
    rows: list[Row] = [
        {
            "row_type": "node",
            "node_id": int(node["node_id"]),
            "t": int(node["t"]),
            "z": float(node["z"]),
            "y": float(node["y"]),
            "x": float(node["x"]),
            "source_id": -1,
            "target_id": -1,
        }
        for node in nodes
    ]
    rows.extend(
        {
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": 0.0,
            "y": 0.0,
            "x": 0.0,
            "source_id": int(edge["source_id"]),
            "target_id": int(edge["target_id"]),
        }
        for edge in edges
    )
    return rows


def round_trip(rows: list[Row], td: Any) -> tuple[list[Row], Any]:
    """Exercise real JSON serialization and the production row-to-graph adapter."""
    serialized = canonical_json(rows)
    restored = json.loads(serialized)
    return restored, _rows_to_graph(restored, td)


def score(graph: Any, truth: Any, metrics: Any, scale: tuple[float, ...], max_distance: float) -> dict[str, Any]:
    try:
        result = metrics.evaluate(graph, truth, scale=scale, max_distance=max_distance)
    except TypeError as error:  # empty-graph behavior in the pinned evaluator
        return {
            "status": "evaluator_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    edge = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
    division = _jaccard(result.division_tp, result.division_fp, result.division_fn)
    return {
        "status": "ok",
        "edge_tp": result.edge_tp,
        "edge_fp": result.edge_fp,
        "edge_fn": result.edge_fn,
        "division_tp": result.division_tp,
        "division_fp": result.division_fp,
        "division_fn": result.division_fn,
        "edge_jaccard": round(edge, 8),
        "division_jaccard": round(division, 8),
        "score": round(edge + 0.1 * division, 8),
    }


def fixture_rows(kind: str) -> list[Row]:
    nodes = {
        "empty": [],
        "one_node": [(10, 0, 1.0, 2.0, 3.0)],
        "one_edge": [(10, 0, 1.0, 2.0, 3.0), (20, 1, 1.0, 2.0, 3.0)],
        "one_division": [
            (10, 0, 1.0, 2.0, 3.0),
            (20, 1, 1.0, 2.0, 3.0),
            (30, 2, 1.0, 1.0, 2.0),
            (40, 2, 1.0, 3.0, 4.0),
        ],
    }[kind]
    edges = {
        "empty": [], "one_node": [], "one_edge": [(10, 20)],
        "one_division": [(10, 20), (20, 30), (20, 40)],
    }[kind]
    rows: list[Row] = [
        {"row_type": "node", "node_id": node_id, "t": t, "z": z, "y": y, "x": x,
         "source_id": -1, "target_id": -1}
        for node_id, t, z, y, x in nodes
    ]
    rows.extend(
        {"row_type": "edge", "node_id": -1, "t": -1, "z": 0.0, "y": 0.0, "x": 0.0,
         "source_id": source, "target_id": target}
        for source, target in edges
    )
    return rows


def _mutate_nodes(rows: list[Row], operation: Callable[[Row], None]) -> list[Row]:
    changed = deepcopy(rows)
    for row in changed:
        if row["row_type"] == "node":
            operation(row)
    return changed


def _swap_zx(row: Row) -> None:
    row["z"], row["x"] = row["x"], row["z"]


def perturbations(rows: list[Row], scale: tuple[float, ...]) -> dict[str, tuple[list[Row], tuple[float, ...], float]]:
    node_ids = sorted(int(row["node_id"]) for row in rows if row["row_type"] == "node")
    id_map = {old: old + 9_000_000_000_000 for old in node_ids}
    remapped = deepcopy(rows)
    for row in remapped:
        for key in ("node_id", "source_id", "target_id"):
            value = int(row[key])
            if value in id_map:
                row[key] = id_map[value]
    reversed_edges = deepcopy(rows)
    for row in reversed_edges:
        if row["row_type"] == "edge":
            row["source_id"], row["target_id"] = row["target_id"], row["source_id"]
    no_divisions = deepcopy(rows)
    seen_sources: set[int] = set()
    kept: list[Row] = []
    for row in no_divisions:
        source = int(row["source_id"])
        if row["row_type"] == "edge" and source in seen_sources:
            continue
        if row["row_type"] == "edge":
            seen_sources.add(source)
        kept.append(row)
    z_offset = _mutate_nodes(rows, lambda row: row.__setitem__("z", float(row["z"]) + 5.0))
    tolerance_probe = _mutate_nodes(
        rows, lambda row: row.__setitem__("x", float(row["x"]) + 0.5)
    )
    return {
        "coordinate_axis_swap_zx": (_mutate_nodes(rows, _swap_zx), scale, 7.0),
        "voxel_spacing_isotropic": (z_offset, (1.0, 1.0, 1.0), 7.0),
        "frame_index_plus_one": (_mutate_nodes(rows, lambda row: row.__setitem__("t", int(row["t"]) + 1)), scale, 7.0),
        "node_id_remap": (remapped, scale, 7.0),
        "edge_direction_reverse": (reversed_edges, scale, 7.0),
        "division_flatten": (kept, scale, 7.0),
        "matching_tolerance_zero": (tolerance_probe, scale, 0.0),
    }


def run_oracle_contract(
    geff_path: Path, official_source: Path, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    metrics, td = official_modules(official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    rows = graph_to_rows(truth)
    restored, oracle = round_trip(rows, td)
    scale = tuple(float(item) for item in config["scale_um"])
    max_distance = float(config["matching_tolerance"])
    baseline = score(oracle, truth, metrics, scale, max_distance)
    diagnostics = {}
    for name, (candidate_rows, candidate_scale, tolerance) in perturbations(rows, scale).items():
        serialized, candidate = round_trip(candidate_rows, td)
        candidate_score = score(candidate, truth, metrics, candidate_scale, tolerance)
        diagnostics[name] = {
            "prediction_sha256": digest(serialized),
            "score_delta_from_oracle": round(candidate_score.get("score", 0.0) - baseline["score"], 8),
            "evaluator": candidate_score,
        }
    fixtures = {}
    for kind in ("empty", "one_node", "one_edge", "one_division"):
        fixture = fixture_rows(kind)
        if fixture:
            _, fixture_truth = round_trip(fixture, td)
            _, fixture_prediction = round_trip(fixture, td)
        else:
            fixture_truth = td.graph.InMemoryGraph()
            fixture_prediction = td.graph.InMemoryGraph()
            import polars as pl  # type: ignore

            for graph in (fixture_truth, fixture_prediction):
                for key in ("z", "y", "x"):
                    graph.add_node_attr_key(key, pl.Float64, -999999.0)
        fixtures[kind] = score(fixture_prediction, fixture_truth, metrics, (1.0, 1.0, 1.0), 7.0)
    division_rows = fixture_rows("one_division")
    flattened_rows = perturbations(division_rows, (1.0, 1.0, 1.0))["division_flatten"][0]
    _, division_truth = round_trip(division_rows, td)
    _, flattened_division = round_trip(flattened_rows, td)
    fixtures["one_division_flattened_prediction"] = score(
        flattened_division, division_truth, metrics, (1.0, 1.0, 1.0), 7.0
    )
    expected_upper = 1.1
    return {
        "schema_version": 1,
        "issue": "SOT-2302",
        "axis": "official-evaluator-perfect-oracle-contract",
        "result": "promoted" if np.isclose(baseline["score"], expected_upper) else "inconclusive",
        "dataset_id": config["screen_ids"][0],
        "split": {"screen": config["screen_ids"], "confirm": config["confirm_ids"], "disjoint": not bool(set(config["screen_ids"]) & set(config["confirm_ids"]))},
        "confirm_accessed": False,
        "production_comparison_blocked": not np.isclose(baseline["score"], expected_upper),
        "production_champion_changed": False,
        "kaggle_submission_executed": False,
        "official_evaluator": {"revision": repository_revision(official_source), "max_distance": max_distance, "scale_zyx_um": list(scale)},
        "hashes": {"input_geff_sha256": sha256_tree(geff_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "oracle_prediction_sha256": digest(restored)},
        "perfect_oracle": baseline,
        "fixtures": fixtures,
        "perturbations": diagnostics,
        "contract_conclusion": {
            "status": "adapter_contract_verified" if np.isclose(baseline["score"], expected_upper) else "adapter_contract_failure",
            "next_issue": "SOT-2303",
            "production_prediction_floor_is_adapter_caused": False if np.isclose(baseline["score"], expected_upper) else None,
            "required_contract": "rows use t plus z,y,x voxel coordinates; edges point parent-to-child; node ids are arbitrary but edge references must be consistent; evaluator uses scale in z,y,x order and max_distance=7.0",
            "empty_graph_behavior": "official evaluator raises TypeError; downstream adapter must guard empty predictions before evaluation",
        },
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
