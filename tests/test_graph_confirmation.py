import json
from pathlib import Path

import pytest

from biohub_baseline.graph_confirmation import freeze_candidate

ROOT = Path(__file__).resolve().parents[1]


def test_screen_winner_is_frozen_before_confirm() -> None:
    frozen = freeze_candidate(ROOT, ROOT / "config/sot-2304-confirm.json", ROOT / "config/champion.json")
    assert frozen["frozen_before_confirm"] is True
    assert frozen["candidate_config_sha256"] == "ff3b976481bd6fd4b487fa5a1171428d424e2cbb8bdc691f905c17a052d00b60"
    assert frozen["candidate_output_sha256"] == "c0119d8eba7e03e9b3c29804e309330225770950791cfbdd0cf333fccaf11fd3"


def test_overlap_is_rejected(tmp_path: Path) -> None:
    config = json.loads((ROOT / "config/sot-2304-confirm.json").read_text())
    config["confirm_ids"] = config["screen_ids"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="split changed|overlap"):
        freeze_candidate(ROOT, path, ROOT / "config/champion.json")
