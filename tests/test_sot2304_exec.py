import csv
from pathlib import Path

import pytest

from scripts.verify_sot2304_exec import EXPECTED_COLUMNS, validate_submission


def test_exec_validator_rejects_dangling_edges(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerow({"id": 0, "dataset": "x", "row_type": "edge", "source_id": 1, "target_id": 2})
    with pytest.raises(RuntimeError, match="dangling"):
        validate_submission(output)
