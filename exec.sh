#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR="${1:-/kaggle/input/biohub-cell-tracking-during-development/test}"
OUTPUT_FILE="${2:-/kaggle/working/submission.csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python
GENERATE_ARGS=(--input "$INPUT_DIR" --output "$OUTPUT_FILE")
if [[ -n "${GRAPH_COORDINATE_SPACE:-}" ]]; then
  GENERATE_ARGS+=(--graph-coordinate-space "$GRAPH_COORDINATE_SPACE")
fi
"$PYTHON_BIN" -m biohub_baseline.cli generate "${GENERATE_ARGS[@]}"
"$PYTHON_BIN" -m biohub_baseline.cli validate "$OUTPUT_FILE"
