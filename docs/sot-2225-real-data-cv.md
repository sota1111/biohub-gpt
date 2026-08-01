# SOT-2225 real-data CV contract

This issue freezes an embryo-held-out evaluation contract for the current
`daughter-geometry-v1` champion. It does not promote a candidate or change the
production configuration.

## Immutable split

The immutable screen field is `44b6_0113de3b`; the independent confirm field is
`6bba_05b6850b`. They come from different biological embryos. The selected
image and truth directories are pinned by manifest digests in
`artifacts/sot-2225-provenance.json`; raw data is not redistributed.
`scripts/evaluate_real_cv.py` rejects missing assets, unknown embryos, an empty
side, an ID/config disagreement, or an embryo assigned to both sides, and
records the exact sorted field IDs plus their SHA-256 digest in its output.

## Official evaluator and sanity gate

The runner accepts an explicit checkout of
`royerlab/kaggle-cell-tracking-competition` pinned to revision
`075fc5f5a52d11077f9dc2b074644618f26939e2`. Before accepting a ledger it
self-scores an official ground-truth GEFF graph. The required result is Edge
Jaccard `1.0`, Division Jaccard `1.0`, total `1.1`.

```bash
python scripts/evaluate_real_cv.py \
  --data-dir /path/to/official/train \
  --official-source /path/to/kaggle-cell-tracking-competition
```

The committed ledgers from two fixed-seed runs store split-level and field-level Edge/Division Jaccard, combined
score, GT/predicted node and edge counts, density/development/division strata,
runtime, and peak RSS. Run the command twice on the same immutable inputs; the
split digest and every non-resource aggregate must match.

## Provenance finding

`artifacts/sot-2225-provenance.json` prevents a misleading lineage claim: the
completed Kaggle submission scoring `0.509` is submission `55130948` from
`biohub-claude` champion `detect-link-v1`, while this repository's current
champion is `daughter-geometry-v1`. Both artifacts are traceable, but they are
not the same package.

## Incumbent result

On both fixed-seed runs, screen scored `0.10000000` (Edge `0.0`, Division
`1.0`) and confirm scored `0.14788419` (Edge `0.04788419`, Division `1.0`).
The screen prediction produced 24,780 nodes against 52 GT nodes; confirm
produced 9,515 against 861. Excess detections and the resulting poor edge
precision/recall are therefore the incumbent bottleneck. Runtime and peak RSS
remain in the JSON ledgers; resource measurements are intentionally excluded
from the determinism assertion.

## Public candidate audit

The public version-32 two-seed notebook is Apache-2.0 and its four attached
datasets advertise CC0-1.0. It is not adopted: authenticated retrieval of the
exact v32 source returned HTTP 403 and the attached mutable dataset references
did not expose content hashes, so exact bytes cannot yet be reproduced.
