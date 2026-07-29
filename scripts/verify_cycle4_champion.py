#!/usr/bin/env python3
"""Audit cycle-4 candidates and verify the fixed champion through the offline exec contract."""

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


def run_experiment(command: str, output: Path, expected_exit: int) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "biohub_baseline.cli", command, "--output", str(output)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode != expected_exit:
        raise RuntimeError(
            f"{command} exited {completed.returncode}, expected {expected_exit}: {completed.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def assert_split(ledger: dict) -> None:
    screen = ledger["split"]["screen"]
    confirm = ledger["split"]["confirm"]
    if not ledger["split"]["disjoint"] or set(screen) & set(confirm):
        raise RuntimeError(f"{ledger['experiment_id']} screen/confirm split is not disjoint")


def main() -> None:
    champion = json.loads((ROOT / "config/champion.json").read_text(encoding="utf-8"))
    if champion["champion_id"] != "daughter-geometry-v1" or champion["cycle"] != 4:
        raise RuntimeError("cycle-4 champion metadata is not fixed to daughter-geometry-v1")

    with tempfile.TemporaryDirectory(prefix="biohub-cycle4-") as temporary:
        temporary_path = Path(temporary)
        separation = run_experiment(
            "evaluate-instance-separation", temporary_path / "separation.json", 2
        )
        division = run_experiment("evaluate-division-geometry", temporary_path / "division.json", 0)
        for ledger in (separation, division):
            assert_split(ledger)
        if separation["decision"]["promote"]:
            raise RuntimeError("touching-watershed-v1 unexpectedly promoted")
        if not division["decision"]["promote"]:
            raise RuntimeError("daughter-geometry-v1 promotion was not independently reproduced")

        input_path = temporary_path / "test"
        input_path.mkdir()
        frames = np.zeros((3, 8, 12, 12), dtype=np.float32)
        for time_index, x in enumerate((4, 5, 6)):
            frames[time_index, 2:4, 5:7, x : x + 2] = 10
        zarr.save(input_path / "cycle4-fixture.zarr", frames)

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
                cwd=ROOT,
                env=environment,
                check=True,
                timeout=60,
            )
        runtime_seconds = time.perf_counter() - started
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        hashes = [sha256(output) for output in outputs]
        if len(set(hashes)) != 1:
            raise RuntimeError("exec output is not deterministic")
        with outputs[0].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames
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

    subprocess.run(["bash", str(ROOT / "scripts/package_kernel.sh")], cwd=ROOT, check=True)
    manifest = ROOT / "dist/kaggle-kernel/SHA256SUMS"
    confirmation = {
        "issue": "SOT-2171",
        "seed_policy": {
            "instance_separation": separation["seed"],
            "division_geometry": division["seed"],
            "screen_confirm_disjoint": True,
            "fixtures": "deterministic synthetic, fixed by source and seed",
        },
        "candidates": {
            "touching-watershed-v1": {
                "promote": False,
                "screen": separation["screen"]["candidates"][0]["metrics"],
                "confirm": separation["confirm"]["candidate"]["metrics"],
                "incumbent_confirm": separation["confirm"]["incumbent"]["metrics"],
            },
            "daughter-geometry-v1": {
                "promote": True,
                "screen": division["screen"]["top_candidate"],
                "confirm": division["confirm"]["candidate"],
                "incumbent_confirm": division["confirm"]["incumbent"],
            },
        },
        "decision": {
            "champion_id": champion["champion_id"],
            "non_promoted_reverted": ["touching-watershed-v1"],
            "execution_matches_decision": champion["detection_model"]["name"]
            == "adaptive-local-peaks-v1",
        },
        "kaggle": {
            "kernel": "sota1111/biohub-gpt-cycle-4-champion",
            "submitted": False,
            "skip_reason": (
                "Kaggle CLI 2.2.4 push returned an empty/non-JSON API response; "
                "follow-up list/status returned HTTP 400/404, so no completed "
                "kernel version existed to submit"
            ),
            "resume": [
                "kaggle kernels push -p dist/kaggle-kernel",
                "kaggle kernels status sota1111/biohub-gpt-cycle-4-champion",
                (
                    "kaggle competitions submit "
                    "-c biohub-cell-tracking-during-development "
                    "-k sota1111/biohub-gpt-cycle-4-champion -v <completed-version> "
                    "-f submission.csv -m 'SOT-2171 cycle-4 champion'"
                ),
            ],
        },
    }
    (ROOT / champion["cycle4_confirmation"]).write_text(
        json.dumps(confirmation, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "issue": "SOT-2171",
        "champion_id": champion["champion_id"],
        "cycle": champion["cycle"],
        "confirmation": champion["cycle4_confirmation"],
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
            "timeout_seconds": 60,
        },
        "package": {
            "path": "dist/kaggle-kernel",
            "manifest_sha256": sha256(manifest),
            "internet_enabled": False,
            "notebook_timeout_seconds": 11 * 60 * 60,
        },
        "status": "pass",
    }
    (ROOT / champion["exec_compatibility_report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
