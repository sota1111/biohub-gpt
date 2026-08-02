#!/usr/bin/env python3
"""Compare contract-aligned serialization on the immutable screen dataset only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from biohub_baseline.graph_contract import run_screen
from biohub_baseline.oracle_contract import digest
from biohub_baseline.real_cv import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--champion", type=Path, default=ROOT / "config/champion.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/sot-2303-graph-contract.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sot-2303-graph-contract.json")
    args = parser.parse_args()
    first = run_screen(args.data_dir, args.official_source, args.champion, args.config)
    second = run_screen(args.data_dir, args.official_source, args.champion, args.config)
    ignored = {"runtime_seconds", "max_rss_kib"}
    stable_first = {key: value for key, value in first.items() if key not in ignored}
    stable_second = {key: value for key, value in second.items() if key not in ignored}
    first["determinism"] = {"runs": 2, "identical": stable_first == stable_second,
                            "stable_payload_sha256": digest(stable_first)}
    if not first["determinism"]["identical"]:
        raise SystemExit("screen output is not deterministic")
    first["hashes"]["runner_sha256"] = subprocess.check_output(
        ["sha256sum", __file__], text=True
    ).split()[0]
    write_json_atomic(args.output, first)
    print(json.dumps(first, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
