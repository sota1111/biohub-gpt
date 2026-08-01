import csv
from pathlib import Path

import pytest

from scripts.verify_latest_artifact import EXPECTED_COLUMNS, audit_predecessors, validate_submission


def test_predecessor_audit_selects_incumbent() -> None:
    audit = audit_predecessors()
    assert audit["selected"]["kind"] == "champion"
    assert audit["selected"]["id"] == "daughter-geometry-v1"
    assert audit["SOT-2226"]["screen_passed"] is False
    assert audit["SOT-2227"]["confirm_accessed"] is False


def test_submission_validation_rejects_dangling_edge(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerow({"id": 0, "dataset": "x", "row_type": "edge", "source_id": 1, "target_id": 2})
    with pytest.raises(RuntimeError, match="dangling"):
        validate_submission(output)
