# SOT-2274 density-aware detector calibration

The production `daughter-geometry-v1` tracker and lineage configuration were
held fixed while three pre-registered detector configurations varied robust
confidence, minimum connected-component size, and anisotropy-aware physical
NMS. The implementation also raises those filters when provisional candidate
density grows and applies a fixed late-stage relaxation. No production
configuration was changed.

## Immutable screen protocol

Only screen field `44b6_0113de3b` was mounted. The independent confirm field
`6bba_05b6850b` was explicitly forbidden and the runner aborts if either of its
assets is visible. The split digest is
`f26b52e2a6174a21984e9d3fe089a2a2ab7291d5740217bfd2270d90e9efd320`.
Both fixed runs record identical detection and graph hashes for every candidate.
Input metadata, detector code, grid configuration, and tracker configuration
hashes are recorded in the machine-readable ledgers.

## Screen result

| configuration | score | Edge Jaccard | Division Jaccard | nodes / GT | node precision | node recall | runtime (s), run 1 | peak RSS (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 0.100000 | 0.000000 | 1.000000 | 476.54 | n/a | n/a | 372.45 | 607,520 |
| balanced | 0.100000 | 0.000000 | 1.000000 | 130.40 | 0.000737 | 0.096154 | 304.60 | 623,024 |
| precision | 0.100000 | 0.000000 | 1.000000 | 22.94 | 0.000000 | 0.000000 | 114.80 | 714,424 |
| high-precision | 0.100000 | 0.000000 | 1.000000 | 4.40 | 0.000000 | 0.000000 | 74.82 | 714,424 |

The screen field is the sparse/no-division stratum (52 GT nodes, 50 GT edges).
All candidates retained the incumbent score in that stratum and reported zero
reference-consistency errors, but none reached the pre-fixed `+0.005` score
gate. The balanced setting reduced predicted nodes from 24,780 to 6,781 while
retaining five matched nodes; stricter settings reduced the node ratio further
but lost all matched nodes. Runtime remained well inside the 12-hour Kaggle
budget; observed peak RSS was below 0.7 GiB, so GPU memory is not required by
this CPU detector stage.

## Decision

No candidate is forwarded. The champion remains unchanged, confirm was not
accessed, and no Kaggle submission was executed. Full candidate values and the
non-promotion reason are preserved in
`artifacts/sot-2274-density-calibration-run1.json` and
`artifacts/sot-2274-density-calibration-run2.json`.
