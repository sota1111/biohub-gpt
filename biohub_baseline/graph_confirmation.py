"""One-shot held-out confirmation for a pre-frozen graph-contract candidate."""

from __future__ import annotations

import hashlib
import resource
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment import load_json
from .graph_contract import _evaluate
from .oracle_contract import digest, sha256_tree
from .real_cv import official_modules, repository_revision
from .submission import build_rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_candidate(root: Path, config_path: Path, champion_path: Path) -> dict[str, Any]:
    """Audit the screen winner and return its immutable pre-confirm identity."""
    config = load_json(config_path)
    screen_path = root / config["screen_artifact"]
    screen = load_json(screen_path)
    champion = load_json(champion_path)
    if screen["issue"] != "SOT-2303" or not screen["gate"]["passed"]:
        raise ValueError("SOT-2303 did not produce a passing screen winner")
    if screen["confirm_accessed"] or screen["production_champion_changed"]:
        raise ValueError("screen artifact violated the screen/confirm boundary")
    if screen["split"]["screen"] != config["screen_ids"] or screen["split"]["confirm"] != config["confirm_ids"]:
        raise ValueError("screen/confirm split changed after screening")
    if set(config["screen_ids"]) & set(config["confirm_ids"]):
        raise ValueError("screen and confirm IDs overlap")
    candidate = deepcopy(champion)
    candidate["graph_contract"] = deepcopy(screen["candidate_config"]["graph_contract"])
    expected = screen["hashes"]["candidate_config_sha256"]
    if digest(candidate) != expected or digest(screen["candidate_config"]) != expected:
        raise ValueError("candidate config no longer matches the frozen screen winner")
    return {
        "frozen_before_confirm": True,
        "screen_artifact": config["screen_artifact"],
        "screen_artifact_sha256": file_sha256(screen_path),
        "candidate_config_sha256": expected,
        "candidate_output_sha256": screen["hashes"]["candidate_output_sha256"],
        "screen_code_sha256": screen["hashes"]["code_sha256"],
        "screen_config_sha256": screen["hashes"]["config_sha256"],
        "screen_input_zarr_sha256": screen["hashes"]["input_zarr_sha256"],
        "screen_input_geff_sha256": screen["hashes"]["input_geff_sha256"],
        "candidate_config": candidate,
    }


def run_confirmation(data_dir: Path, official_source: Path, champion_path: Path,
                     config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.perf_counter()
    config = load_json(config_path)
    frozen = freeze_candidate(root, config_path, champion_path)
    revision = repository_revision(official_source)
    if revision != config["official_evaluator_revision"]:
        raise ValueError(f"official evaluator revision mismatch: {revision}")
    if len(config["confirm_ids"]) != 1:
        raise ValueError("exactly one held-out confirm dataset is required")
    dataset_id = config["confirm_ids"][0]
    image_path = data_dir / f"{dataset_id}.zarr"
    truth_path = data_dir / f"{dataset_id}.geff"
    import zarr
    group = zarr.open(image_path, mode="r")
    frames = group if hasattr(group, "shape") else group["0"]
    incumbent_config = load_json(champion_path)
    candidate_config = frozen.pop("candidate_config")
    common = (
        dataset_id, frames, incumbent_config["threshold_percentile"],
        incumbent_config["min_voxels"], incumbent_config["max_link_distance"],
        incumbent_config.get("link_model"), incumbent_config.get("detection_model"),
        incumbent_config.get("preprocessing"),
    )
    # This is the only candidate access to held-out data: one generation/evaluation per arm.
    incumbent_rows = build_rows(*common)
    candidate_rows = build_rows(*common, graph_contract=candidate_config["graph_contract"])
    metrics, td = official_modules(official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(truth_path)
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    scale = tuple(float(item) for item in config["scale_um"])
    tolerance = float(config["matching_tolerance"])
    incumbent = _evaluate(incumbent_rows, truth.copy(), metrics, td, scale, tolerance)
    candidate = _evaluate(candidate_rows, truth.copy(), metrics, td, scale, tolerance)
    delta = round(candidate["score"] - incumbent["score"], 8)
    runtime = round(time.perf_counter() - started, 6)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gate = config["gate"]
    strata = {
        "held_out_embryo": {
            "incumbent_score": incumbent["score"], "candidate_score": candidate["score"],
            "delta": delta, "regression": delta < -float(gate["maximum_stratum_regression"]),
        },
        "density": {"incumbent_pred_nodes": incumbent["pred_nodes"],
                    "candidate_pred_nodes": candidate["pred_nodes"],
                    "candidate_node_recall": candidate["node_recall"]},
        "development_stage": {"dataset": dataset_id, "full_time_series_evaluated": True},
        "division": {"incumbent_tp_fp_fn": [incumbent["division_tp"], incumbent["division_fp"], incumbent["division_fn"]],
                     "candidate_tp_fp_fn": [candidate["division_tp"], candidate["division_fp"], candidate["division_fn"]],
                     "regression": candidate["division_jaccard"] < incumbent["division_jaccard"]},
    }
    passed = (
        delta >= float(gate["minimum_score_delta"])
        and candidate["reference_integrity_errors"] == 0
        and not strata["held_out_embryo"]["regression"]
        and not strata["division"]["regression"]
        and runtime <= float(gate["maximum_runtime_seconds"])
        and rss <= int(gate["maximum_rss_kib"])
    )
    report = {
        "schema_version": 1, "issue": "SOT-2304",
        "axis": "official-voxel-zyx-serialization-held-out-confirm",
        "result": "promoted" if passed else "rejected",
        "screen_winner": frozen, "confirm_access": {"count": 1, "dataset_id": dataset_id},
        "split": {"screen": config["screen_ids"], "confirm": config["confirm_ids"], "disjoint": True},
        "official_evaluator": {"revision": revision, "scale_zyx_um": list(scale), "max_distance": tolerance},
        "incumbent": incumbent, "candidate": candidate, "score_delta": delta, "strata": strata,
        "gate": {**gate, "reference_integrity_errors": candidate["reference_integrity_errors"], "passed": passed},
        "hashes": {"confirm_config_sha256": file_sha256(config_path),
                   "confirm_input_zarr_sha256": sha256_tree(image_path),
                   "confirm_input_geff_sha256": sha256_tree(truth_path),
                   "incumbent_output_sha256": digest(incumbent_rows),
                   "candidate_output_sha256": digest(candidate_rows)},
        "runtime_seconds": runtime, "max_rss_kib": rss,
        "production_champion_changed": passed, "kaggle_submission_executed": False,
    }
    if passed:
        candidate_config["champion_id"] = "daughter-geometry-voxel-zyx-v1"
        candidate_config["cycle"] = 5
        candidate_config["graph_contract_confirmation"] = "artifacts/sot-2304-graph-confirmation.json"
        candidate_config["exec_compatibility_report"] = "artifacts/sot-2304-exec-compatibility.json"
        candidate_config["promotion_history"] = [*candidate_config["promotion_history"], {
            "issue": "SOT-2304", "candidate": "official-voxel-zyx-serialization",
            "promoted": True, "evidence": "artifacts/sot-2304-graph-confirmation.json",
        }]
        return report, candidate_config
    return report, None
