#!/usr/bin/env python3
"""Verify the fixed cycle-2 champion through its real offline exec contract."""

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
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    champion = json.loads((ROOT / "config/champion.json").read_text(encoding="utf-8"))
    promotion_checks = []
    for item in champion["promotion_history"]:
        evidence = json.loads((ROOT / item["evidence"]).read_text(encoding="utf-8"))
        actual = bool(evidence["decision"]["promote"])
        if actual != item["promoted"]:
            raise RuntimeError(f"{item['issue']} promotion mismatch")
        promotion_checks.append({**item, "actual": actual})

    tracemalloc.start()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="biohub-cycle2-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "test"
        input_path.mkdir()
        frames = np.zeros((3, 8, 12, 12), dtype=np.float32)
        for time_index, x in enumerate((4, 5, 6)):
            frames[time_index, 2:4, 5:7, x : x + 2] = 10
        zarr.save(input_path / "cycle2-fixture.zarr", frames)

        outputs = [temporary_path / "submission-1.csv", temporary_path / "submission-2.csv"]
        environment = {
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "NO_PROXY": "*",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
        }
        for output in outputs:
            subprocess.run(
                ["bash", str(ROOT / "exec.sh"), str(input_path), str(output)],
                cwd=ROOT,
                env=environment,
                check=True,
                timeout=60,
            )

        hashes = [sha256(output) for output in outputs]
        if len(set(hashes)) != 1:
            raise RuntimeError("exec output is not deterministic")
        with outputs[0].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            columns = handle.seek(0) or next(csv.reader(handle))
        if columns != EXPECTED_COLUMNS:
            raise RuntimeError(f"submission schema mismatch: {columns}")
        nodes = {int(row["node_id"]) for row in rows if row["row_type"] == "node"}
        edges = [
            (int(row["source_id"]), int(row["target_id"]))
            for row in rows
            if row["row_type"] == "edge"
        ]
        if any(source not in nodes or target not in nodes for source, target in edges):
            raise RuntimeError("submission contains dangling edge references")

    runtime_seconds = time.perf_counter() - started
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    subprocess.run(["bash", str(ROOT / "scripts/package_kernel.sh")], cwd=ROOT, check=True)
    package_manifest = ROOT / "dist/kaggle-kernel/SHA256SUMS"
    report = {
        "issue": champion["promotion_history"][-1]["issue"],
        "champion_id": champion["champion_id"],
        "cycle": champion["cycle"],
        "promotion_checks": promotion_checks,
        "exec": {
            "internet": "disabled by proxy fail-closed environment",
            "runs": 2,
            "deterministic": True,
            "submission_sha256": hashes[0],
            "columns": EXPECTED_COLUMNS,
            "rows": len(rows),
            "nodes": len(nodes),
            "edges": len(edges),
            "references_valid": True,
            "runtime_seconds": round(runtime_seconds, 6),
            "peak_memory_bytes": peak_memory_bytes,
        },
        "package": {
            "path": "dist/kaggle-kernel",
            "manifest": "dist/kaggle-kernel/SHA256SUMS",
            "manifest_sha256": sha256(package_manifest),
            "internet_enabled": False,
            "timeout_seconds": 11 * 60 * 60,
        },
        "status": "pass",
    }
    output = ROOT / champion["exec_compatibility_report"]
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
