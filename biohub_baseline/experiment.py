from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def deterministic_split(
    dataset_ids: list[str], seed: int, screen_fraction: float
) -> dict[str, list[str]]:
    ordered = sorted(
        dataset_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    screen_count = max(1, min(len(ordered) - 1, round(len(ordered) * screen_fraction)))
    return {"screen": ordered[:screen_count], "confirm": ordered[screen_count:]}


def promotion_decision(
    candidate: dict[str, Any], champion: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    screen_pass = (
        candidate["screen"]["composite"]
        >= champion["screen"]["composite"] + gates["screen_min_delta"]
        and candidate["screen"]["detection_f1"] >= gates["min_detection_f1"]
        and candidate["screen"]["edge_f1"] >= gates["min_edge_f1"]
    )
    confirm_pass = screen_pass and (
        candidate["confirm"]["composite"]
        >= champion["confirm"]["composite"] + gates["confirm_min_delta"]
        and candidate["confirm"]["detection_f1"] >= gates["min_detection_f1"]
        and candidate["confirm"]["edge_f1"] >= gates["min_edge_f1"]
    )
    return {
        "screen_pass": screen_pass,
        "confirm_evaluated": screen_pass,
        "promote": confirm_pass,
        "reason": (
            "confirm gates passed"
            if confirm_pass
            else "screen gates failed"
            if not screen_pass
            else "confirm gates failed"
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
