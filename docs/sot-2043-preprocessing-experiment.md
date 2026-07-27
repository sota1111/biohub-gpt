# SOT-2043 drift and anisotropy experiment

## Decision

The `phase-correlation-anisotropy-v1` candidate is promoted. It is the only
variable in this experiment; detection remains `adaptive-local-peaks-v1` and
tracking remains `temporal-lineage-v1`.

## Reproducible setup

- Seed family: `2043`
- Screen series: `2043`, `2044`
- Confirm series: `2143`, `2144`
- Voxel spacing: z/y/x = `2.0/1.0/1.0`
- Maximum phase-correlation shift: z/y/x = `4/8/8` voxels
- Gates: `config/evaluation-gates.json`

Run:

```bash
python -m biohub_baseline.cli evaluate-preprocessing \
  --output artifacts/sot-2043-preprocessing-experiment.json
```

The machine-readable artifact records overall and per-series results. Screen
improved composite from `0.400000` to `1.000000`; confirm improved it from
`0.316667` to `1.000000`. Confirm edge F1 improved from `0.666667` to
`1.000000`.

## Coordinate behavior

Phase correlation estimates the alignment shift of each frame against the
first non-empty reference. Coordinates are then shifted and multiplied by
z/y/x voxel spacing. The inverse mapping is defined and tested to absolute
tolerance `1e-9`. Empty/constant frames use a zero shift. Corrected coordinates
are intentionally allowed outside the original frame bounds because the output
is a common physical reference, not an array index.

## Promotion consistency

`config/champion.json`, `artifacts/champion-metrics.json`, `exec.sh`, and the
Kaggle notebook entry point all use the same preprocessing configuration.
`scripts/reproduce_baseline.sh` regenerates the experiment and champion
artifacts without network access.
