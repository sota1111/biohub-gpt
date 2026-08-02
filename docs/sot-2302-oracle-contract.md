# SOT-2302 official evaluator oracle contract

The immutable screen field `44b6_0113de3b` was used to round-trip the official
train GT graph through the production node/edge row serializer and adapter. The
independent confirm field `6bba_05b6850b` was not mounted or accessed. No model,
production champion, or Kaggle submission was changed.

## Result

The perfect oracle scores the expected upper bound: Edge Jaccard `1.0`,
Division Jaccard `1.0`, combined score `1.1`. Two complete runs produced the
same non-resource payload hash
`bc6b7db7a3f1fc8cf262867772b936ced7688dbd53fdedb22417228c2a87a406`.
The row adapter is therefore not the cause of the incumbent's `0.100000` floor.

The floor is exactly reproduced by swapping Z and X or reversing edge direction:
both yield Edge Jaccard `0.0` and combined score `0.1`. A one-frame time offset
yields `0.98461538`; a half-voxel offset with zero matching tolerance yields
`0.1`; consistent node-ID remapping remains at `1.1`. The screen field contains
no division, so division representation is isolated with the known division
fixture: flattening one daughter edge changes Division Jaccard from `1.0` to
`0.0` and score from `1.1` to `0.66666667`.

## Known fixtures and evaluator boundary

One-node, one-edge, and one-division oracle fixtures each score `1.1`. The
pinned official evaluator raises `TypeError: reduce() of empty iterable with no
initial value` for an empty predicted/GT graph. This behavior is recorded rather
than hidden: adapters must guard empty predictions before calling the evaluator.

## Reproduction and provenance

Run `scripts/diagnose_oracle_contract.py` twice indirectly through its built-in
determinism check, with only the screen GEFF visible:

```bash
python scripts/diagnose_oracle_contract.py \
  --data-dir /path/to/screen-only/train \
  --official-source /path/to/kaggle-cell-tracking-competition
```

Machine-readable evidence is in `artifacts/sot-2302-oracle-contract.json`. It
pins evaluator revision `075fc5f5a52d11077f9dc2b074644618f26939e2`, input,
config, serialized prediction, and determinism hashes, plus runtime and peak RSS.

## Contract handed to SOT-2303

Rows must use `t` and voxel coordinates ordered `z,y,x`; edges point from parent
to child; node identifiers may change only when all edge references change with
them; evaluation uses physical spacing `(1.625, 0.40625, 0.40625)` in Z,Y,X
order and matching tolerance `7.0`. SOT-2303 should inspect the production
serializer for coordinate order and edge direction first.
