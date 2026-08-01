#!/usr/bin/env python3
"""Run the SOT-2226 detector-only screen with the production tracker frozen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from biohub_baseline.experiment import load_json
from biohub_baseline.preprocess import SpatialTransform, estimate_phase_shift
from biohub_baseline.real_cv import _graph_counts, _jaccard, _rows_to_graph, official_modules
from biohub_baseline.submission import extract_appearance_descriptor
from biohub_baseline.track import Detection, LinkConfig, link_constrained
from biohub_baseline.two_seed import TwoSeedConfig, infer_two_seed_nodes

ROOT = Path(__file__).resolve().parents[1]


def _load_predictor(source_root: Path) -> Any:
    scripts = source_root / "repo" / "scripts"
    package = source_root / "repo" / "src"
    sys.path[:0] = [str(package), str(scripts)]
    spec = importlib.util.spec_from_file_location("sot2226_predictor", scripts / "predict_unet_transformer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned predictor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _open_frames(path: Path) -> Any:
    import zarr

    group = zarr.open(path, mode="r")
    return group if hasattr(group, "shape") else group["0"]


def _rows(dataset: str, frames: Any, nodes: list[list[tuple[float, float, float]]], champion: dict[str, Any]) -> list[dict[str, object]]:
    preprocessing = champion["preprocessing"]
    spacing = tuple(float(value) for value in preprocessing["voxel_spacing"])
    max_shift = tuple(int(value) for value in preprocessing["max_shift_voxels"])
    reference = np.asarray(frames[0])
    transforms = [SpatialTransform(spacing, estimate_phase_shift(reference, np.asarray(frames[t]), max_shift)) for t in range(len(nodes))]
    detections: list[list[Detection]] = []
    next_id = 1
    for t, points in enumerate(nodes):
        frame = np.asarray(frames[t])
        current = []
        for point in points:
            z, y, x = transforms[t].forward(point)
            current.append(Detection(next_id, t, z, y, x, extract_appearance_descriptor(frame, point), None))
            next_id += 1
        detections.append(current)
    link = dict(champion["link_model"])
    # Detector-only isolation: use the exact champion tracker, including its id/config.
    edges = link_constrained(detections, LinkConfig(max_distance=float(champion["max_link_distance"]), **link))
    rows: list[dict[str, object]] = []
    for frame in detections:
        for node in frame:
            rows.append({"dataset": dataset, "row_type": "node", "node_id": node.node_id, "t": node.t, "z": node.z, "y": node.y, "x": node.x, "source_id": -1, "target_id": -1})
    rows.extend({"dataset": dataset, "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": source, "target_id": target} for source, target in edges)
    return rows


def _score(rows: list[dict[str, object]], truth: Any, metrics: Any, td: Any, scale: tuple[float, ...]) -> dict[str, Any]:
    predicted = _rows_to_graph(rows, td)
    result = metrics.evaluate(predicted, truth, scale=scale)
    pred_nodes, pred_edges, _ = _graph_counts(predicted)
    edge = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
    division = _jaccard(result.division_tp, result.division_fp, result.division_fn)
    return {"edge_jaccard": edge, "division_jaccard": division, "score": edge + 0.1 * division, "pred_nodes": pred_nodes, "pred_edges": pred_edges, "edge_tp": result.edge_tp, "edge_fp": result.edge_fp, "edge_fn": result.edge_fn, "division_tp": result.division_tp, "division_fp": result.division_fp, "division_fn": result.division_fn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--secondary-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/two-seed-screen.json")
    args = parser.parse_args()
    screen = load_json(args.config)
    if screen["screen_ids"] != ["44b6_0113de3b"] or "6bba_05b6850b" not in screen["forbidden_confirm_ids"]:
        raise SystemExit("screen/confirm contract changed; refusing to run")
    if any((args.data_dir / f"{item}.zarr").exists() or (args.data_dir / f"{item}.geff").exists() for item in screen["forbidden_confirm_ids"]):
        raise SystemExit("confirm asset is visible to the screen process")

    import torch

    predictor = _load_predictor(args.primary_root)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    primary_path = args.primary_root / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary_path = args.secondary_root / "weights/unet_transformer/split_0/edge_predictor_best.pth"
    primary, _, downsample = predictor.load_model(primary_path, device)
    secondary, _, secondary_downsample = predictor.load_model(secondary_path, device)
    if tuple(downsample) != tuple(secondary_downsample):
        raise SystemExit("seed grids differ")

    class Adapter(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, value: Any) -> Any:
            return self.model.detect(value[0, 0])[None, None]

    dataset = screen["screen_ids"][0]
    frames = _open_frames(args.data_dir / f"{dataset}.zarr")
    metrics, td = official_modules(args.official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(args.data_dir / f"{dataset}.geff")
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    champion = load_json(ROOT / screen["tracker_config"])
    results = []
    for candidate in screen["candidates"]:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        config = TwoSeedConfig(patch_shape=tuple(candidate["patch_shape"]), overlap=tuple(candidate["overlap"]), blend_weight=float(candidate["blend_weight"]), threshold=float(candidate["threshold"]), min_instance_voxels=int(candidate["min_instance_voxels"]), voxel_spacing_um=(1.625, 1.625, 1.625))
        all_nodes = []
        for index in range(int(frames.shape[0])):
            raw = np.asarray(frames[index])[:: downsample[0], :: downsample[1], :: downsample[2]]
            _, points = infer_two_seed_nodes(raw, Adapter(primary), Adapter(secondary), config)
            all_nodes.append([(z * downsample[0], y * downsample[1], x * downsample[2]) for z, y, x in points])
        fingerprint = json.dumps(all_nodes, separators=(",", ":"))
        scored = _score(_rows(dataset, frames, all_nodes, champion), truth, metrics, td, tuple(load_json(ROOT / "config/real-data-cv.json")["scale_um"]))
        results.append({**candidate, **scored, "node_sha256": hashlib.sha256(fingerprint.encode()).hexdigest(), "runtime_seconds": time.perf_counter() - started, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1048576})
    incumbent = load_json(ROOT / "artifacts/sot-2225-real-data-cv-run1.json")["splits"]["screen"]
    best = max(results, key=lambda item: (item["score"], -item["runtime_seconds"]))
    passed = best["score"] >= incumbent["score"] + float(screen["gate"]["minimum_score_delta"])
    payload = {"schema_version": 1, "issue": "SOT-2226", "dataset": dataset, "confirm_accessed": False, "tracker_id": champion["champion_id"], "incumbent": incumbent, "candidates": results, "screen_passed": passed, "winner": best if passed else None, "forward_to": "SOT-2227" if passed else None, "champion_updated": False, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
