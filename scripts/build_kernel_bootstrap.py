#!/usr/bin/env python3
"""Embed the offline runtime into the one source file uploaded by Kaggle CLI."""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dist/kaggle-kernel/notebook.py"


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in [
            Path("exec.sh"),
            Path("config/champion.json"),
            *sorted(Path("biohub_baseline").glob("*.py")),
        ]:
            archive.add(ROOT / relative, arcname=relative)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    template = (ROOT / "kaggle/notebook.py").read_text(encoding="utf-8")
    if 'PAYLOAD = ""' not in template:
        raise RuntimeError("kernel payload placeholder is missing")
    DESTINATION.write_text(
        template.replace('PAYLOAD = ""', f'PAYLOAD = "{payload}"'),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
