"""Kaggle entry point. The repository is uploaded as this kernel's source bundle."""

import subprocess
from pathlib import Path

subprocess.run(
    [
        "bash",
        str(Path(__file__).resolve().parent / "exec.sh"),
        "/kaggle/input/biohub-cell-tracking-during-development/test",
        "/kaggle/working/submission.csv",
    ],
    check=True,
    timeout=11 * 60 * 60,
)
