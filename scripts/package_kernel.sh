#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="$REPO_ROOT/dist/kaggle-kernel"
mkdir -p "$DESTINATION/config" "$DESTINATION/biohub_baseline"
cp "$REPO_ROOT/kaggle/kernel-metadata.json" "$DESTINATION/kernel-metadata.json"
cp "$REPO_ROOT/kaggle/notebook.py" "$DESTINATION/notebook.py"
cp "$REPO_ROOT/exec.sh" "$DESTINATION/exec.sh"
cp "$REPO_ROOT/config/champion.json" "$DESTINATION/config/champion.json"
cp "$REPO_ROOT"/biohub_baseline/*.py "$DESTINATION/biohub_baseline/"
echo "Offline kernel package ready: $DESTINATION"
