# SOT-2303 official graph output contract screen

The SOT-2302 oracle diagnosis fixed the official contract: prediction rows use
drift-corrected voxel coordinates in Z/Y/X order, and the evaluator applies the
physical spacing `(1.625, 0.40625, 0.40625)`. Production tracking intentionally
continues to use physical coordinates. This candidate changes only the final
serialization boundary from physical Z/Y/X to voxel Z/Y/X; detector, tracker,
time indexing, node IDs, and parent-to-child edges are unchanged.

## Screen result

Only immutable screen dataset `44b6_0113de3b` was mounted. Confirm dataset
`6bba_05b6850b` was not visible or accessed. The pinned official evaluator at
revision `075fc5f5a52d11077f9dc2b074644618f26939e2` produced:

| Metric | Incumbent | voxel Z/Y/X candidate |
| --- | ---: | ---: |
| Edge Jaccard | 0.00000000 | 0.19318182 |
| Division Jaccard | 1.00000000 | 1.00000000 |
| Official score | 0.10000000 | 0.29318182 |
| Node precision | 0.00016142 | 0.00193705 |
| Node recall | 0.07692308 | 0.92307692 |
| Edge TP / FP / FN | 0 / 5 / 50 | 17 / 38 / 33 |
| Division TP / FP / FN | 0 / 0 / 0 | 0 / 0 / 0 |
| Reference integrity errors | 0 | 0 |

The score delta is `+0.19318182`, above the predeclared `+0.005` gate, with no
stratum regression in the available screen stratum. The high predicted-node
count is an incumbent detector property and was deliberately not tuned in this
contract-only issue.

Two full generation/evaluation runs were identical. Runtime was `758.122203s`
for the complete two-run comparison and peak RSS was `677336 KiB`.

## Reproduction and exact candidate

Run the screen comparison with a directory containing only the screen Zarr and
GEFF assets:

```bash
PYTHONPATH=. python scripts/screen_graph_contract.py \
  --data-dir /path/to/screen-only \
  --official-source /path/to/kaggle-cell-tracking-competition
```

Local generation and the Kaggle `exec.sh` path both call the same
`biohub_baseline.cli generate` implementation. Exercise the candidate without
changing `config/champion.json` using:

```bash
GRAPH_COORDINATE_SPACE=voxel_zyx ./exec.sh /path/to/test /path/to/submission.csv
```

Exact code/config/input/output hashes, metric counts, determinism evidence, and
the complete candidate config are recorded in
`artifacts/sot-2303-graph-contract.json`. Candidate output SHA-256 is
`c0119d8eba7e03e9b3c29804e309330225770950791cfbdd0cf333fccaf11fd3`;
candidate config SHA-256 is
`ff3b976481bd6fd4b487fa5a1171428d424e2cbb8bdc691f905c17a052d00b60`.
The production champion was not updated, confirm was not accessed, and no
Kaggle submission was made.
