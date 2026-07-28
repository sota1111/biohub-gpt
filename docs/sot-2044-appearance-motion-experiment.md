# SOT-2044 appearance / motion promotion

## Decision

Promote `appearance-motion-link-v1`.

The predecessor champion (`phase-correlation-anisotropy-v1`) was held fixed
apart from the link candidate cost. Three lightweight weight combinations were
screened; only the lowest-complexity top candidate was evaluated on confirm.

## Fixed split and result

| Stage | Cases | Baseline edge F1 | Candidate edge F1 | Baseline switches | Candidate switches |
| --- | --- | ---: | ---: | ---: | ---: |
| screen | crossing, crowded | 0.500000 | 1.000000 | 4 | 0 |
| confirm | crossing, temporary appearance loss, zero motion | 0.714286 | 1.000000 | 4 | 0 |

The candidate passed the configured screen and confirm deltas. All predicted
lineages passed parent-count, time-order, and cycle validation.

## Promoted cost

- appearance descriptor weight: `0.1`
- constant-velocity residual weight: `0.25`
- acceleration correction: `0.0`

The screen included an acceleration candidate. It tied on metrics, so the
lower-weight constant-velocity candidate won the deterministic complexity
tie-break and was promoted.

## Safety and compatibility

- Image-boundary patches are clipped to valid array bounds.
- Empty, non-finite, or out-of-frame patches return no descriptor and retain
  the existing coordinate/density cost.
- Missing history retains the coordinate/appearance cost.
- Zero velocity produces a finite stationary prediction.
- `exec.sh` uses `config/champion.json`, so local and Kaggle-offline generation
  share the promoted settings.

## Reproduce

```bash
python -m biohub_baseline.cli evaluate-link-features \
  --output artifacts/sot-2044-appearance-motion-experiment.json
./scripts/reproduce_baseline.sh
./scripts/package_kernel.sh
```

Machine-readable evidence is in
`artifacts/sot-2044-appearance-motion-experiment.json`.
