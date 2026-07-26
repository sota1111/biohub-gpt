#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m pytest -q
python -m biohub_baseline.cli validate tests/fixtures/valid_submission.csv
metrics_file="$(mktemp)"
trap 'rm -f "$metrics_file"' EXIT
python -m biohub_baseline.cli evaluate-fixture --output "$metrics_file"
cmp "$metrics_file" artifacts/champion-metrics.json
python -m biohub_baseline.cli evaluate-lineage \
  --output artifacts/sot-1990-lineage-experiment.json
echo "champion config: config/champion.json"
echo "champion metrics: artifacts/champion-metrics.json"
