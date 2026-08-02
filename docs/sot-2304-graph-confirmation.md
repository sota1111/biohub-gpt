# SOT-2304 held-out graph-contract confirmation

The SOT-2303 screen winner was frozen before confirm access with candidate
config SHA-256 `ff3b976481bd6fd4b487fa5a1171428d424e2cbb8bdc691f905c17a052d00b60`
and screen output SHA-256
`c0119d8eba7e03e9b3c29804e309330225770950791cfbdd0cf333fccaf11fd3`.
The complete screen artifact, code, config, input, and output hashes are copied
into `artifacts/sot-2304-graph-confirmation.json`. Screen
`44b6_0113de3b` and confirm `6bba_05b6850b` are disjoint.

## Confirm decision

The held-out embryo was accessed once. Incumbent and candidate used the same
detector/tracker, official evaluator revision
`075fc5f5a52d11077f9dc2b074644618f26939e2`, physical scale, matching
tolerance, and resource envelope.

| Metric | Incumbent | Candidate |
| --- | ---: | ---: |
| Official score | 0.14788419 | 0.21206744 |
| Edge Jaccard | 0.04788419 | 0.21206744 |
| Node recall | 0.10917538 | 0.58420441 |
| Edge TP / FP / FN | 43 / 53 / 802 | 239 / 282 / 606 |
| Division TP / FP / FN | 0 / 0 / 0 | 0 / 1 / 0 |
| Reference errors | 0 | 0 |

The overall delta was `+0.06418325`, runtime was `271.891890s`, and peak RSS
was `470376 KiB`. However, the candidate added one false-positive division,
violating the predeclared zero-regression stratum gate. The candidate is
therefore **not promoted**. `config/champion.json`, local behavior, and
`exec.sh` remain on the verified `daughter-geometry-v1` incumbent.

The offline compatibility artifact records two internet-disabled `exec.sh`
runs, deterministic submission/schema/reference checks, runtime/memory, and
the packaged kernel manifest fingerprint. No Kaggle submission was made.
