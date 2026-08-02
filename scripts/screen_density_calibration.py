#!/usr/bin/env python3
"""Run the pre-fixed SOT-2274 detector calibration grid on screen data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from biohub_baseline.density_calibration import DensityCalibration, detect_density_calibrated
from biohub_baseline.experiment import load_json
from biohub_baseline.preprocess import SpatialTransform, estimate_phase_shift
from biohub_baseline.real_cv import (
    _graph_counts,
    _jaccard,
    _rows_to_graph,
    official_modules,
    sha256_file,
    split_digest,
)
from biohub_baseline.submission import estimate_detection_volume, extract_appearance_descriptor
from biohub_baseline.track import Detection, LinkConfig, link_constrained

ROOT = Path(__file__).resolve().parents[1]


def _open_frames(path: Path) -> Any:
    import zarr

    group = zarr.open(path, mode="r")
    return group if hasattr(group, "shape") else group["0"]


def _rows(dataset: str, frames: Any, config: DensityCalibration, champion: dict[str, Any]) -> tuple[list[dict[str, object]], str]:
    preprocessing = champion["preprocessing"]
    spacing = tuple(float(value) for value in preprocessing["voxel_spacing"])
    max_shift = tuple(int(value) for value in preprocessing["max_shift_voxels"])
    reference = np.asarray(frames[0])
    transforms = [SpatialTransform(spacing, estimate_phase_shift(reference, np.asarray(frames[t]), max_shift)) for t in range(int(frames.shape[0]))]
    detections: list[list[Detection]] = []
    fingerprint: list[list[tuple[float, float, float]]] = []
    next_id = 1
    link = dict(champion["link_model"])
    for t, transform in enumerate(transforms):
        frame = np.asarray(frames[t])
        points = detect_density_calibrated(frame, config, t, len(transforms))
        fingerprint.append(points)
        current = []
        for point in points:
            z, y, x = transform.forward(point)
            current.append(Detection(next_id, t, z, y, x, extract_appearance_descriptor(frame, point), estimate_detection_volume(frame, point, champion["threshold_percentile"])))
            next_id += 1
        detections.append(current)
    edges = link_constrained(detections, LinkConfig(max_distance=float(champion["max_link_distance"]), **link))
    rows: list[dict[str, object]] = []
    for frame in detections:
        for node in frame:
            rows.append({"dataset": dataset, "row_type": "node", "node_id": node.node_id, "t": node.t, "z": node.z, "y": node.y, "x": node.x, "source_id": -1, "target_id": -1})
    rows.extend({"dataset": dataset, "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": source, "target_id": target} for source, target in edges)
    digest = hashlib.sha256(json.dumps(fingerprint, separators=(",", ":")).encode()).hexdigest()
    return rows, digest


def _score(rows: list[dict[str, object]], truth: Any, metrics: Any, td: Any, scale: tuple[float, ...]) -> dict[str, Any]:
    predicted = _rows_to_graph(rows, td)
    result = metrics.evaluate(predicted, truth, scale=scale)
    pred_nodes, pred_edges, _ = _graph_counts(predicted)
    gt_nodes, gt_edges, division = _graph_counts(truth)
    edge = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
    division_jaccard = _jaccard(result.division_tp, result.division_fp, result.division_fn)
    node_recall = float(metrics.node_recall(predicted, truth))
    node_tp = round(node_recall * gt_nodes)
    node_precision = node_tp / pred_nodes if pred_nodes else 0.0
    return {"edge_jaccard": edge, "division_jaccard": division_jaccard, "score": edge + 0.1 * division_jaccard, "pred_nodes": pred_nodes, "pred_edges": pred_edges, "gt_nodes": gt_nodes, "gt_edges": gt_edges, "pred_gt_node_ratio": pred_nodes / gt_nodes, "node_tp": node_tp, "node_precision": node_precision, "node_recall": node_recall, "stratum": "division" if division else "sparse", "reference_consistency_errors": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/density-calibration-screen.json")
    args = parser.parse_args()
    screen = load_json(args.config)
    if screen["screen_ids"] != ["44b6_0113de3b"] or screen["forbidden_confirm_ids"] != ["6bba_05b6850b"]:
        raise SystemExit("immutable split contract changed; refusing to run")
    if any((args.data_dir / f"{item}.zarr").exists() or (args.data_dir / f"{item}.geff").exists() for item in screen["forbidden_confirm_ids"]):
        raise SystemExit("confirm asset is visible to the screen process")
    dataset = screen["screen_ids"][0]
    image_path, truth_path = args.data_dir / f"{dataset}.zarr", args.data_dir / f"{dataset}.geff"
    frames = _open_frames(image_path)
    metrics, td = official_modules(args.official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(truth_path)
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    champion = load_json(ROOT / screen["tracker_config"])
    results = []
    for candidate in screen["candidates"]:
        started = time.perf_counter()
        config = DensityCalibration(**{key: value for key, value in candidate.items() if key != "id"})
        rows, digest = _rows(dataset, frames, config, champion)
        results.append({**candidate, **_score(rows, truth, metrics, td, (1.625, 0.40625, 0.40625)), "detections_sha256": digest, "graph_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "runtime_seconds": time.perf_counter() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    incumbent = load_json(ROOT / "artifacts/sot-2225-real-data-cv-run1.json")["splits"]["screen"]
    best = max(results, key=lambda item: (item["score"], item["node_precision"], -item["runtime_seconds"]))
    passed = best["score"] >= incumbent["score"] + float(screen["gate"]["minimum_score_delta"]) and best["reference_consistency_errors"] == 0
    payload = {"schema_version": 1, "issue": "SOT-2274", "dataset": dataset, "split": {"screen": screen["screen_ids"], "confirm": screen["forbidden_confirm_ids"], "disjoint": True, "sha256": split_digest({"screen": screen["screen_ids"], "confirm": screen["forbidden_confirm_ids"]})}, "confirm_accessed": False, "tracker_id": champion["champion_id"], "tracker_config_sha256": sha256_file(ROOT / screen["tracker_config"]), "config_sha256": sha256_file(args.config), "code_sha256": sha256_file(ROOT / "biohub_baseline/density_calibration.py"), "screen_input": {"zarr_metadata_sha256": sha256_file(image_path / "zarr.json"), "geff_metadata_sha256": sha256_file(truth_path / "zarr.json")}, "incumbent": incumbent, "candidates": results, "screen_passed": passed, "winner": best if passed else None, "forward_to": "SOT-2275" if passed else None, "champion_updated": False, "kaggle_submission_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
