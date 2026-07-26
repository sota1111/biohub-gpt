#!/usr/bin/env bash
set -euo pipefail
DESTINATION="${1:-data/kaggle}"
COMPETITION="biohub-cell-tracking-during-development"
mkdir -p "$DESTINATION"
kaggle competitions download -c "$COMPETITION" -p "$DESTINATION"
echo "Downloaded $COMPETITION to $DESTINATION"
