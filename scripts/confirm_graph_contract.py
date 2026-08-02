#!/usr/bin/env python3
"""Confirm the frozen SOT-2303 winner once on the held-out embryo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_baseline.graph_confirmation import run_confirmation
from biohub_baseline.real_cv import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--champion", type=Path, default=ROOT / "config/champion.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/sot-2304-confirm.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sot-2304-graph-confirmation.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing a second confirm evaluation; output already exists: {args.output}")
    report, promoted = run_confirmation(args.data_dir, args.official_source, args.champion, args.config, ROOT)
    write_json_atomic(args.output, report)
    if promoted is not None:
        write_json_atomic(args.champion, promoted)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
