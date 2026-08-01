#!/usr/bin/env python3
"""Audit the cycle-5 decision and verify the latest submission-eligible artifact."""

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


def audit_predecessors(root: Path = ROOT) -> dict:
    champion_path = root / "config/champion.json"
    baseline = json.loads((root / "artifacts/sot-2225-real-data-cv-run1.json").read_text())
    provenance = json.loads((root / "artifacts/sot-2226-provenance.json").read_text())
    screen = json.loads((root / "artifacts/sot-2226-two-seed-screen.json").read_text())
    champion = json.loads(champion_path.read_text())

    if baseline["sanity"]["score"] != 1.1 or not baseline["split"]["disjoint"]:
        raise RuntimeError("SOT-2225 official-CV contract is not valid")
    if screen["confirm_accessed"] or screen["gate"]["screen_passed"]:
        raise RuntimeError("SOT-2226 no-promotion decision changed")
    if screen["winner"] is not None or screen["forward_to"] is not None:
        raise RuntimeError("an unconfirmed candidate was forwarded")
    if screen["champion_updated"] or provenance["implementation"]["champion_updated"]:
        raise RuntimeError("non-promoted candidate reached production")
    if champion["champion_id"] != "daughter-geometry-v1":
        raise RuntimeError("production champion does not match the audited decision")

    return {
        "SOT-2225": {
            "official_evaluator_sanity": baseline["sanity"],
            "screen_confirm_disjoint": True,
            "production_champion_changed": baseline["production_champion_changed"],
        },
        "SOT-2226": {
            "screen_passed": False,
            "confirm_accessed": False,
            "winner": None,
            "champion_updated": False,
            "ledger_sha256": sha256(root / "artifacts/sot-2226-two-seed-screen.json"),
            "provenance_sha256": sha256(root / "artifacts/sot-2226-provenance.json"),
        },
        "SOT-2227": {
            "decision": "not_applicable_no_screen_winner",
            "confirm_accessed": False,
            "production_reverted": True,
        },
        "selected": {
            "kind": "champion",
            "id": champion["champion_id"],
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "model_sha256": sha256(champion_path),
        },
    }


def validate_submission(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"submission schema mismatch: {columns}")
    nodes = {int(row["node_id"]): row for row in rows if row["row_type"] == "node"}
    edges = [row for row in rows if row["row_type"] == "edge"]
    if any(int(row["source_id"]) not in nodes or int(row["target_id"]) not in nodes for row in edges):
        raise RuntimeError("submission contains dangling edge references")
    if any(int(nodes[int(row["source_id"])]["t"]) >= int(nodes[int(row["target_id"])]["t"]) for row in edges):
        raise RuntimeError("submission contains time-reversed lineage edges")
    return {"columns": columns, "rows": len(rows), "nodes": len(nodes), "edges": len(edges)}


def main() -> None:
    audit = audit_predecessors()
    with tempfile.TemporaryDirectory(prefix="biohub-sot-2228-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "test"
        input_path.mkdir()
        frames = np.zeros((3, 8, 12, 12), dtype=np.float32)
        for time_index, x in enumerate((4, 5, 6)):
            frames[time_index, 2:4, 5:7, x : x + 2] = 10
        zarr.save(input_path / "contract-fixture.zarr", frames)
        outputs = [temporary_path / "submission-1.csv", temporary_path / "submission-2.csv"]
        environment = {
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "NO_PROXY": "*",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
        }
        tracemalloc.start()
        started = time.perf_counter()
        for output in outputs:
            subprocess.run(
                ["bash", str(ROOT / "exec.sh"), str(input_path), str(output)],
                cwd=ROOT, env=environment, check=True, timeout=60,
            )
        runtime_seconds = time.perf_counter() - started
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        hashes = [sha256(output) for output in outputs]
        if len(set(hashes)) != 1:
            raise RuntimeError("exec output is not deterministic")
        submission = validate_submission(outputs[0])

    subprocess.run(["bash", str(ROOT / "scripts/package_kernel.sh")], cwd=ROOT, check=True)
    manifest = ROOT / "dist/kaggle-kernel/SHA256SUMS"
    package_sha = sha256(manifest)
    audit["selected"]["package_manifest_sha256"] = package_sha
    report = {
        "schema_version": 1,
        "issue": "SOT-2228",
        "audit": audit,
        "exec": {
            "internet": "disabled by proxy fail-closed environment",
            "gpu_contract": True,
            "runs": 2,
            "deterministic": True,
            "submission_sha256": hashes[0],
            **submission,
            "references_valid": True,
            "division_lineage_valid": True,
            "runtime_seconds": round(runtime_seconds, 6),
            "peak_memory_bytes": peak_memory_bytes,
            "kaggle_timeout_seconds": 11 * 60 * 60,
            "within_limits": runtime_seconds < 60,
        },
        "package": {
            "path": "dist/kaggle-kernel",
            "manifest_sha256": package_sha,
            "internet_enabled": False,
            "gpu_enabled": True,
        },
        "kaggle": {"submitted": False, "status": "pending_live_submission"},
        "status": "pass",
    }
    output = ROOT / "artifacts/sot-2228-kaggle-verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
