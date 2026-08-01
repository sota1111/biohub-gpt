#!/usr/bin/env python3
"""Run the immutable embryo-held-out CV with the pinned official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_baseline.real_cv import evaluate_real_data, sanity_gate, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, default=ROOT / "config/real-data-cv.json")
    parser.add_argument("--champion", type=Path, default=ROOT / "config/champion.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sot-2225-real-data-cv.json")
    args = parser.parse_args()
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    expected = split_config["official_evaluator"]["revision"]
    result = evaluate_real_data(
        args.data_dir, args.official_source, args.champion, split_config
    )
    if result["official_evaluator"]["revision"] != expected:
        raise SystemExit(
            "official evaluator revision mismatch: "
            f"expected {expected}, got {result['official_evaluator']['revision']}"
        )
    first = next(iter(result["split"]["screen"] + result["split"]["confirm"]))
    result["sanity"] = sanity_gate(
        args.official_source,
        args.data_dir / f"{first}.geff",
        tuple(split_config["scale_um"]),
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
