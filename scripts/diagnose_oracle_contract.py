#!/usr/bin/env python3
"""Re-anchor the official evaluator contract using screen-only oracle round trips."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from biohub_baseline.oracle_contract import digest, run_oracle_contract
from biohub_baseline.real_cv import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/oracle-contract.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sot-2302-oracle-contract.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.official_source, text=True
    ).strip()
    if revision != config["official_evaluator_revision"]:
        raise SystemExit(
            f"official evaluator revision mismatch: expected "
            f"{config['official_evaluator_revision']}, got {revision}"
        )
    if set(config["screen_ids"]) & set(config["confirm_ids"]):
        raise SystemExit("screen and confirm IDs overlap")
    if len(config["screen_ids"]) != 1:
        raise SystemExit("this diagnostic requires exactly one immutable screen field")
    forbidden = [
        item for item in config["confirm_ids"]
        if (args.data_dir / f"{item}.geff").exists() or (args.data_dir / f"{item}.zarr").exists()
    ]
    if forbidden:
        raise SystemExit(f"confirm asset is visible to the screen process: {forbidden}")
    screen = config["screen_ids"][0]
    geff = args.data_dir / f"{screen}.geff"
    if not geff.exists():
        raise SystemExit(f"screen GT is missing: {geff}")
    first = run_oracle_contract(geff, args.official_source, args.config, config)
    second = run_oracle_contract(geff, args.official_source, args.config, config)
    ignored = {"runtime_seconds", "max_rss_kib"}
    stable_first = {key: value for key, value in first.items() if key not in ignored}
    stable_second = {key: value for key, value in second.items() if key not in ignored}
    first["determinism"] = {
        "runs": 2,
        "stable_payload_sha256": digest(stable_first),
        "identical": stable_first == stable_second,
    }
    if not first["determinism"]["identical"]:
        raise SystemExit("oracle/evaluator output was not deterministic")
    write_json_atomic(args.output, first)
    print(json.dumps(first, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
