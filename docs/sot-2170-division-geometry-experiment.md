# SOT-2170 daughter-pair division geometry

## Decision

`daughter-geometry-v1` was promoted on top of the fixed
`appearance-motion-link-v1` champion. SOT-2169's touching-nuclei detector was
not promoted, so the active `adaptive-local-peaks-v1` detector remains fixed.

Division candidates now enforce an adjacent-frame time window and explicitly
score daughter distance/opposition, daughter-pair midpoint, mother-to-daughter
volume conservation, and daughter volume balance. Local above-threshold voxel
mass supplies the optional volume feature in the normal generation and
Kaggle-exec path; detections without usable volume retain the geometry-only
fallback.

## Fixed split screen and independent confirm

Three parameter sets were screened on two deterministic cases. The least
complex tied winner (`0.05` for each soft geometry weight and `0.25` maximum
relative volume error) alone was carried to three disjoint confirm cases.

| Split | Model | Division Jaccard | Edge Jaccard | False division edges | Missed division edges |
| --- | --- | ---: | ---: | ---: | ---: |
| screen | incumbent | 0.333333 | 0.333333 | 2 | 2 |
| screen | candidate | 1.000000 | 1.000000 | 0 | 0 |
| confirm | incumbent | 0.333333 | 0.333333 | 3 | 3 |
| confirm | candidate | 1.000000 | 1.000000 | 0 | 0 |

Runtime is recorded per stage in the experiment ledger. Both configurations
completed in milliseconds on these small-N fixtures, with no lineage integrity
errors.

## Provenance and interaction

The ledger at `artifacts/sot-2170-division-geometry-experiment.json` records
each candidate, each case's predicted/expected edges, diagnostic counts,
runtime, fixed champion, split membership, and promotion decision. The
touching-nuclei stratum is recorded as disabled because SOT-2169 did not
promote; the centroid contract remains compatible with its optional candidate.

Reproduce:

```bash
python -m biohub_baseline.cli evaluate-division-geometry \
  --output artifacts/sot-2170-division-geometry-experiment.json
```

The promoted configuration is stored in `config/champion.json`, which is read
unchanged by local generation and the packaged `exec.sh`/Kaggle notebook path.
