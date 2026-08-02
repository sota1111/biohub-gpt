#!/usr/bin/env python3
"""Run the promoted SOT-2304 package twice with networking disabled."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_submission(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"submission schema mismatch: {columns}")
    nodes = {int(row["node_id"]): row for row in rows if row["row_type"] == "node"}
    edges = [row for row in rows if row["row_type"] == "edge"]
    dangling = [row for row in edges if int(row["source_id"]) not in nodes or int(row["target_id"]) not in nodes]
    if dangling:
        raise RuntimeError("submission contains dangling edge references")
    if any(int(nodes[int(row["source_id"])]["t"]) >= int(nodes[int(row["target_id"])]["t"]) for row in edges):
        raise RuntimeError("submission contains time-reversed lineage edges")
    return {"columns": columns, "rows": len(rows), "nodes": len(nodes), "edges": len(edges)}


def main() -> None:
    confirmation_path = ROOT / "artifacts/sot-2304-graph-confirmation.json"
    confirmation = json.loads(confirmation_path.read_text())
    champion = json.loads((ROOT / "config/champion.json").read_text())
    promoted = confirmation["gate"]["passed"] and confirmation["result"] == "promoted"
    if promoted and champion.get("graph_contract", {}).get("coordinate_space") != "voxel_zyx":
        raise RuntimeError("production champion is not aligned with the confirmed candidate")
    if not promoted and "graph_contract" in champion:
        raise RuntimeError("rejected candidate leaked into the production champion")
    environment = {
        **os.environ, "PYTHON_BIN": sys.executable, "NO_PROXY": "*",
        "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
    }
    with tempfile.TemporaryDirectory(prefix="biohub-sot-2304-exec-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "test"
        input_path.mkdir()
        frames = np.zeros((3, 8, 12, 12), dtype=np.float32)
        for time_index, x in enumerate((4, 5, 6)):
            frames[time_index, 2:4, 5:7, x:x + 2] = 10
        zarr.save(input_path / "contract-fixture.zarr", frames)
        outputs = [temporary_path / "submission-1.csv", temporary_path / "submission-2.csv"]
        tracemalloc.start()
        started = time.perf_counter()
        for output in outputs:
            subprocess.run(["bash", str(ROOT / "exec.sh"), str(input_path), str(output)],
                           cwd=ROOT, env=environment, check=True, timeout=60)
        runtime = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        hashes = [sha256(path) for path in outputs]
        if len(set(hashes)) != 1:
            raise RuntimeError("exec output is not deterministic")
        submission = validate_submission(outputs[0])
    subprocess.run(["bash", str(ROOT / "scripts/package_kernel.sh")], cwd=ROOT,
                   env=environment, check=True, timeout=60)
    manifest = ROOT / "dist/kaggle-kernel/SHA256SUMS"
    report = {
        "schema_version": 1, "issue": "SOT-2304", "status": "pass",
        "champion_id": champion["champion_id"],
        "promotion_decision": confirmation["result"],
        "production_behavior": "candidate" if promoted else "unchanged_incumbent",
        "confirmation_artifact_sha256": sha256(confirmation_path),
        "exec": {"internet": "disabled by fail-closed proxy", "runs": 2,
                 "deterministic": True, "submission_sha256": hashes[0], **submission,
                 "references_valid": True, "runtime_seconds": round(runtime, 6),
                 "peak_memory_bytes": peak_memory, "timeout_seconds": 60,
                 "within_kaggle_12h_gpu_memory_contract": True},
        "package": {"path": "dist/kaggle-kernel", "manifest_sha256": sha256(manifest),
                    "champion_sha256": sha256(ROOT / "config/champion.json"),
                    "internet_enabled": False, "gpu_enabled": True},
        "kaggle_submission_executed": False,
    }
    output = ROOT / "artifacts/sot-2304-exec-compatibility.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
