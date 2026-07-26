#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR="${1:-/kaggle/input/biohub-cell-tracking-during-development/test}"
OUTPUT_FILE="${2:-/kaggle/working/submission.csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
python -m biohub_baseline.cli generate --input "$INPUT_DIR" --output "$OUTPUT_FILE"
python -m biohub_baseline.cli validate "$OUTPUT_FILE"
