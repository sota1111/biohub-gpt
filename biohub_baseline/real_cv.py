"""Reproducible real-data CV utilities for the official Biohub evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import load_json
from .submission import build_rows


def embryo_id(dataset_id: str) -> str:
    """Return the biological-series id from ``{embryo}_{field}``."""
    if "_" not in dataset_id:
        raise ValueError(f"dataset id has no embryo prefix: {dataset_id}")
    return dataset_id.split("_", 1)[0]


def validate_fixed_split(
    dataset_ids: Iterable[str], screen_embryos: Iterable[str], confirm_embryos: Iterable[str]
) -> dict[str, list[str]]:
    """Materialize an embryo-disjoint split and reject omissions or leakage."""
    ids = sorted(set(dataset_ids))
    if not ids:
        raise ValueError("no dataset ids supplied")
    screen_families = set(screen_embryos)
    confirm_families = set(confirm_embryos)
    if not screen_families or not confirm_families:
        raise ValueError("screen and confirm must each contain at least one embryo")
    overlap = screen_families & confirm_families
    if overlap:
        raise ValueError(f"embryo leakage across splits: {sorted(overlap)}")
    known = screen_families | confirm_families
    unexpected = sorted({embryo_id(item) for item in ids} - known)
    if unexpected:
        raise ValueError(f"unassigned embryo ids: {unexpected}")
    result = {
        "screen": [item for item in ids if embryo_id(item) in screen_families],
        "confirm": [item for item in ids if embryo_id(item) in confirm_families],
    }
    if not result["screen"] or not result["confirm"]:
        raise ValueError("the available data does not populate both splits")
    return result


def split_digest(split: dict[str, list[str]]) -> str:
    payload = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def official_modules(source: Path) -> tuple[Any, Any]:
    """Import the pinned official scorer without its optional image stack."""
    source = source.resolve()
    package_root = source / "src"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    import tracksdata as td  # type: ignore
    from tracking_cellmot import metrics  # type: ignore

    return metrics, td


def _graph_counts(graph: Any) -> tuple[int, int, bool]:
    nodes = int(graph.num_nodes())
    edges = int(graph.num_edges())
    node_ids = graph.node_ids()
    divisions = bool(nodes and np.any(np.asarray(graph.out_degree(node_ids)) >= 2))
    return nodes, edges, divisions


def _rows_to_graph(rows: list[dict[str, object]], td: Any) -> Any:
    import polars as pl  # type: ignore

    frame = pl.DataFrame(rows)
    node_rows = frame.filter(pl.col("row_type") == "node")
    edge_rows = frame.filter(pl.col("row_type") == "edge")
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = graph.bulk_add_nodes(
        node_rows.select(
            pl.col("t").cast(pl.Int64),
            pl.col("z").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("x").cast(pl.Float64),
        ).to_dicts()
    )
    id_map = dict(zip(node_rows["node_id"].to_list(), assigned, strict=True))
    if edge_rows.height:
        graph.bulk_add_edges(
            [
                {"source_id": id_map[source], "target_id": id_map[target]}
                for source, target in zip(
                    edge_rows["source_id"].to_list(),
                    edge_rows["target_id"].to_list(),
                    strict=True,
                )
            ]
        )
    return graph


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator else 1.0


def _stage(density: float, division: bool) -> str:
    if division:
        return "division"
    if density < 50:
        return "sparse"
    if density < 120:
        return "medium"
    return "dense"


def evaluate_real_data(
    data_dir: Path,
    official_source: Path,
    config_path: Path,
    split_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the production champion against paired official image/GT assets."""
    metrics, td = official_modules(official_source)
    dataset_ids = sorted(split_config["screen_ids"] + split_config["confirm_ids"])
    missing = [
        item
        for item in dataset_ids
        if not (data_dir / f"{item}.zarr").exists()
        or not (data_dir / f"{item}.geff").exists()
    ]
    if missing:
        raise ValueError(f"fixed split assets are missing: {missing}")
    split = validate_fixed_split(
        dataset_ids,
        split_config["screen_embryos"],
        split_config["confirm_embryos"],
    )
    expected_split = {
        "screen": sorted(split_config["screen_ids"]),
        "confirm": sorted(split_config["confirm_ids"]),
    }
    if split != expected_split:
        raise ValueError(f"fixed split ids disagree with embryo assignment: {expected_split}")
    champion = load_json(config_path)
    records: list[dict[str, Any]] = []
    for split_name in ("screen", "confirm"):
        for dataset_id in split[split_name]:
            started = time.perf_counter()
            image_path = data_dir / f"{dataset_id}.zarr"
            gt_path = data_dir / f"{dataset_id}.geff"
            try:
                import zarr

                group = zarr.open(image_path, mode="r")
                frames = group if hasattr(group, "shape") else group["0"]
            except ImportError:
                from .ngff import open_ngff

                frames = open_ngff(image_path)
            rows = build_rows(
                dataset_id,
                frames,
                champion["threshold_percentile"],
                champion["min_voxels"],
                champion["max_link_distance"],
                champion.get("link_model"),
                champion.get("detection_model"),
                champion.get("preprocessing"),
                champion.get("graph_contract"),
            )
            predicted = _rows_to_graph(rows, td)
            loaded = td.graph.IndexedRXGraph.from_geff(gt_path)
            truth = loaded[0] if isinstance(loaded, tuple) else loaded
            result = metrics.evaluate(predicted, truth, scale=tuple(split_config["scale_um"]))
            gt_nodes, gt_edges, has_division = _graph_counts(truth)
            pred_nodes, pred_edges, _ = _graph_counts(predicted)
            timepoints = max(1, int(frames.shape[0]))
            density = gt_nodes / timepoints
            edge_jaccard = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
            division_jaccard = _jaccard(
                result.division_tp, result.division_fp, result.division_fn
            )
            records.append(
                {
                    "split": split_name,
                    "dataset_id": dataset_id,
                    "embryo_id": embryo_id(dataset_id),
                    "stratum": _stage(density, has_division),
                    "timepoints": timepoints,
                    "gt_nodes": gt_nodes,
                    "gt_edges": gt_edges,
                    "pred_nodes": pred_nodes,
                    "pred_edges": pred_edges,
                    "nodes_per_frame": round(density, 6),
                    "has_division": has_division,
                    "edge_tp": result.edge_tp,
                    "edge_fp": result.edge_fp,
                    "edge_fn": result.edge_fn,
                    "division_tp": result.division_tp,
                    "division_fp": result.division_fp,
                    "division_fn": result.division_fn,
                    "edge_jaccard": round(edge_jaccard, 8),
                    "division_jaccard": round(division_jaccard, 8),
                    "score": round(edge_jaccard + 0.1 * division_jaccard, 8),
                    "runtime_seconds": round(time.perf_counter() - started, 6),
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        totals = defaultdict(int)
        for row in selected:
            for key in (
                "edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                "gt_nodes", "gt_edges", "pred_nodes", "pred_edges",
            ):
                totals[key] += row[key]
        edge = _jaccard(totals["edge_tp"], totals["edge_fp"], totals["edge_fn"])
        division = _jaccard(
            totals["division_tp"], totals["division_fp"], totals["division_fn"]
        )
        return {
            "datasets": len(selected),
            **totals,
            "edge_jaccard": round(edge, 8),
            "division_jaccard": round(division, 8),
            "score": round(edge + 0.1 * division, 8),
            "runtime_seconds": round(sum(row["runtime_seconds"] for row in selected), 6),
            "max_rss_kib": max(row["max_rss_kib"] for row in selected),
        }

    by_split = {
        name: aggregate([row for row in records if row["split"] == name])
        for name in ("screen", "confirm")
    }
    strata = {
        name: aggregate([row for row in records if row["stratum"] == name])
        for name in sorted({row["stratum"] for row in records})
    }
    return {
        "schema_version": 1,
        "issue": "SOT-2225",
        "champion_id": champion["champion_id"],
        "production_champion_changed": False,
        "split": {
            **split,
            "screen_embryos": sorted(split_config["screen_embryos"]),
            "confirm_embryos": sorted(split_config["confirm_embryos"]),
            "disjoint": True,
            "sha256": split_digest(split),
        },
        "official_evaluator": {
            "repository": split_config["official_evaluator"]["repository"],
            "revision": repository_revision(official_source),
            "expected_revision": split_config["official_evaluator"]["revision"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "splits": by_split,
        "strata": strata,
        "datasets": records,
    }


def sanity_gate(official_source: Path, geff_path: Path, scale: tuple[float, ...]) -> dict[str, Any]:
    metrics, td = official_modules(official_source)
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    truth = loaded[0] if isinstance(loaded, tuple) else loaded
    result = metrics.evaluate(truth.copy(), truth, scale=scale)
    edge = _jaccard(result.edge_tp, result.edge_fp, result.edge_fn)
    division = _jaccard(result.division_tp, result.division_fp, result.division_fn)
    score = edge + 0.1 * division
    if not np.isclose(score, 1.1):
        raise RuntimeError(f"official evaluator self-score was {score}, expected 1.1")
    return {"edge_jaccard": edge, "division_jaccard": division, "score": score}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
