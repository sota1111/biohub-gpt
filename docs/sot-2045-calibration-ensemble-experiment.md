# SOT-2045 calibration and ensemble experiment

## Decision

Do not promote the rank-calibrated ensemble. Keep
`appearance-motion-link-v1` as the champion and leave its execution
configuration unchanged.

The candidate implementation combines coordinate, density, appearance, and
motion costs after deterministic rank calibration. Missing appearance or
motion components are excluded and the remaining weights are renormalized.
Temperatures `0.75`, `1.0`, and `1.5` were screened with seed `2045`.

## Fixed split and result

| Stage | Series | Champion edge F1 | Best candidate edge F1 | Decision |
| --- | --- | ---: | ---: | --- |
| screen | crossing, crowded | 1.000000 | 0.500000 | fail |
| confirm | crossing, temporary appearance loss, zero motion | not run | not run | screen gate prevented evaluation |

Both screen series regressed from edge F1 `1.0` to `0.5`, with two identity
switches per series. The configured screen delta therefore failed. In
accordance with the fixed screen-to-confirm protocol, the independent confirm
split was not inspected or used to tune the candidates.

The deterministic tie-break selected the lowest screened temperature
(`temperature=0.75`). Runtime and peak traced memory for every candidate and
the incumbent are retained in the machine-readable artifact. Resource use
does not override the accuracy and per-series regression gates.

## Revert and compatibility

- `config/champion.json` remains `appearance-motion-link-v1`.
- No rank calibration is enabled by default; `LinkConfig.calibration="raw"`
  preserves the incumbent behavior.
- `exec.sh` and the Kaggle package continue to consume the unchanged champion
  configuration, so the executable artifact remains consistent with the
  promoted champion.
- The candidate code and experiment CLI remain available for reproducible
  future evaluation without changing production behavior.

## Reproduce

```bash
python -m biohub_baseline.cli evaluate-calibration-ensemble \
  --output artifacts/sot-2045-calibration-ensemble-experiment.json
```

Exit code `2` is the expected machine-readable non-promotion result. Full
settings, per-series metrics, runtime, memory, split, and decision are recorded
in `artifacts/sot-2045-calibration-ensemble-experiment.json`.
