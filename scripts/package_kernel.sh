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
PACKAGE_PYTHON="${PYTHON_BIN:-python3}"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PACKAGE_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
"$PACKAGE_PYTHON" "$REPO_ROOT/scripts/build_kernel_bootstrap.py"
(
  cd "$DESTINATION"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
echo "Offline kernel package ready: $DESTINATION"
