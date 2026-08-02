"""Screen-only comparison of production graph serialization contracts."""

from __future__ import annotations

import hashlib
import resource
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment import load_json
from .oracle_contract import digest, sha256_tree
from .real_cv import _jaccard, _rows_to_graph, official_modules, repository_revision
from .submission import build_rows, validate_rows


def _matched_nodes(graph: Any, td: Any) -> int:
    import polars as pl  # type: ignore

    key = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    attrs = graph.node_attrs(attr_keys=[key])
    return attrs.filter(pl.col(key).is_not_null() & (pl.col(key) != -1)).height


def _evaluate(rows: list[dict[str, object]], truth: Any, metrics: Any, td: Any,
              scale: tuple[float, ...], tolerance: float) -> dict[str, Any]:
    validate_rows(rows)
    predicted = _rows_to_graph(rows, td)
    result = metrics.evaluate(predicted, truth, scale=scale, max_distance=tolerance)
    matched = _matched_nodes(predicted, td)
    edge = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
    division = _jaccard(result.division_tp, result.division_fp, result.division_fn)
    reference_errors = 0
    node_ids = {int(row["node_id"]) for row in rows if row["row_type"] == "node"}
    for row in rows:
        if row["row_type"] == "edge" and (
            int(row["source_id"]) not in node_ids or int(row["target_id"]) not in node_ids
        ):
            reference_errors += 1
    return {
        "pred_nodes": result.num_pred_nodes,
        "matched_nodes": matched,
        "node_precision": round(matched / result.num_pred_nodes, 8) if result.num_pred_nodes else 0.0,
        "node_recall": round(matched / truth.num_nodes(), 8) if truth.num_nodes() else 1.0,
        "edge_tp": result.edge_tp, "edge_fp": result.edge_fp, "edge_fn": result.edge_fn,
        "division_tp": result.division_tp, "division_fp": result.division_fp,
        "division_fn": result.division_fn,
        "edge_jaccard": round(edge, 8), "division_jaccard": round(division, 8),
        "score": round(edge + 0.1 * division, 8),
        "reference_integrity_errors": reference_errors,
        "rows_sha256": digest(rows),
    }


def run_screen(data_dir: Path, official_source: Path, champion_path: Path,
               screen_config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    screen_config = load_json(screen_config_path)
    champion = load_json(champion_path)
    if set(screen_config["screen_ids"]) & set(screen_config["confirm_ids"]):
        raise ValueError("screen and confirm IDs overlap")
    if len(screen_config["screen_ids"]) != 1:
        raise ValueError("exactly one immutable screen dataset is required")
    forbidden = [item for item in screen_config["confirm_ids"] if
                 (data_dir / f"{item}.geff").exists() or (data_dir / f"{item}.zarr").exists()]
    if forbidden:
        raise ValueError(f"confirm asset is visible to the screen process: {forbidden}")
    revision = repository_revision(official_source)
    if revision != screen_config["official_evaluator_revision"]:
        raise ValueError(f"official evaluator revision mismatch: {revision}")
    dataset_id = screen_config["screen_ids"][0]
    image_path, truth_path = data_dir / f"{dataset_id}.zarr", data_dir / f"{dataset_id}.geff"
    import zarr
    group = zarr.open(image_path, mode="r")
    frames = group if hasattr(group, "shape") else group["0"]
    metrics, td = official_modules(official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(truth_path)
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    common = (
        dataset_id, frames, champion["threshold_percentile"], champion["min_voxels"],
        champion["max_link_distance"], champion.get("link_model"),
        champion.get("detection_model"), champion.get("preprocessing"),
    )
    incumbent_rows = build_rows(*common)
    candidate_rows = build_rows(*common, graph_contract=screen_config["candidate"])
    scale = tuple(float(item) for item in screen_config["scale_um"])
    tolerance = float(screen_config["matching_tolerance"])
    incumbent = _evaluate(incumbent_rows, truth.copy(), metrics, td, scale, tolerance)
    candidate = _evaluate(candidate_rows, truth.copy(), metrics, td, scale, tolerance)
    delta = round(candidate["score"] - incumbent["score"], 8)
    gate = screen_config["gate"]
    passed = delta >= float(gate["minimum_score_delta"]) and candidate["reference_integrity_errors"] == 0
    candidate_config = deepcopy(champion)
    candidate_config["graph_contract"] = deepcopy(screen_config["candidate"])
    code_path = Path(__file__)
    return {
        "schema_version": 1, "issue": "SOT-2303", "axis": "official-voxel-zyx-serialization",
        "result": "promoted" if passed else "rejected", "dataset_id": dataset_id,
        "split": {"screen": screen_config["screen_ids"], "confirm": screen_config["confirm_ids"], "disjoint": True},
        "confirm_accessed": False, "production_champion_changed": False,
        "kaggle_submission_executed": False, "incumbent_detector_tracker_fixed": True,
        "official_evaluator": {"revision": revision, "scale_zyx_um": list(scale), "max_distance": tolerance},
        "incumbent": incumbent, "candidate": candidate, "score_delta": delta,
        "strata": {"all_screen": {"incumbent_score": incumbent["score"], "candidate_score": candidate["score"], "regression": delta < -float(gate["maximum_stratum_regression"])}},
        "gate": {**gate, "passed": passed},
        "hashes": {
            "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(screen_config_path.read_bytes()).hexdigest(),
            "input_zarr_sha256": sha256_tree(image_path), "input_geff_sha256": sha256_tree(truth_path),
            "candidate_config_sha256": digest(candidate_config), "candidate_output_sha256": digest(candidate_rows),
        },
        "candidate_config": candidate_config if passed else None,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
