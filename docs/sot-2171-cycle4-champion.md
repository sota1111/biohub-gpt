# SOT-2171 cycle-4 champion confirmation

## Final decision

The cycle-4 champion is `daughter-geometry-v1`. The fixed-seed ledgers were
regenerated from source and their disjoint screen/confirm provenance audited.

- `touching-watershed-v1` remains non-promoted: its confirm detection F1 and
  under-splitting equal the incumbent while runtime and memory are higher.
- `daughter-geometry-v1` remains promoted: on the independent confirm cases,
  division and edge Jaccard are both `1.0` versus `0.333333` for the incumbent,
  with no false or missed division edges.
- The execution configuration retains `adaptive-local-peaks-v1` and enables
  only the promoted daughter-geometry settings.

Machine-readable combined evidence is in
`artifacts/sot-2171-cycle4-confirmation.json`.

## Offline exec gate

Run:

```bash
python scripts/verify_cycle4_champion.py
pytest
ruff check .
```

The verifier regenerates both candidate ledgers, checks disjoint provenance
and promotion decisions, executes the real internet-disabled entry point
twice, and validates determinism, schema, node/edge references, runtime,
memory, timeout, and the packaged artifact hash. Results are in
`artifacts/sot-2171-exec-compatibility.json`.

## Kaggle

The packaged kernel is `sota1111/biohub-gpt-cycle-4-champion`. Push it from
`dist/kaggle-kernel`, wait for completion, and submit its output:

The live push was attempted on 2026-07-29 with Kaggle CLI 2.2.4, but the API
returned `Expecting value: line 1 column 1` and did not create the kernel.
Follow-up `kernels list --mine` returned HTTP 400 and `kernels status` returned
HTTP 404. Submission is therefore skipped because no completed kernel version
exists; the local package and offline exec gate passed, so no code failure is
being hidden. Re-authenticate the Kaggle CLI if necessary, then resume with:

```bash
kaggle kernels push -p dist/kaggle-kernel
kaggle kernels status sota1111/biohub-gpt-cycle-4-champion
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -k sota1111/biohub-gpt-cycle-4-champion -v <completed-version> \
  -f submission.csv -m "SOT-2171 cycle-4 champion"
kaggle competitions submissions \
  -c biohub-cell-tracking-during-development
```

If the kernel is not complete during this run, that status is the explicit
skip reason; resume with the status and submit commands above.
