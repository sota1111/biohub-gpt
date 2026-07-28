# SOT-2046 cycle-2 champion

## Champion decision

The cycle-2 champion is uniquely fixed at `appearance-motion-link-v1`.

- SOT-2043 promoted `phase-correlation-anisotropy-v1`.
- SOT-2044 promoted `appearance-motion-link-v1` on top of that preprocessing.
- SOT-2045 rejected the rank-calibrated ensemble at screen, so it did not
  change execution behavior.

`config/champion.json` links all three machine-readable decisions and is the
single configuration consumed by local `exec.sh` and the Kaggle kernel.

## Exec compatibility

Run from a clean checkout:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python scripts/verify_cycle2_champion.py
pytest
ruff check .
```

The gate executes the real entry point twice against a deterministic Zarr
fixture with network proxies failed closed. It validates the exact submission
columns, contiguous IDs, node/edge references, deterministic SHA-256 output,
runtime, peak traced memory, the 11-hour notebook timeout, and the offline
kernel package manifest. Results are in
`artifacts/sot-2046-exec-compatibility.json`.

## Kaggle proof

Kernel `sota1111/biohub-gpt-cycle-2-champion` version **10** was pushed on
2026-07-28 and entered `KernelWorkerStatus.RUNNING`. Kaggle had not completed
the full hidden-test execution within this worker session, so the competition
submission step was intentionally skipped: a Code competition only accepts a
completed kernel version, and submitting a still-running version would not be
valid evidence.

The live attempts exposed and fixed four exec-only incompatibilities before
version 10: Kaggle's one-source-file packaging, the offline image lacking
`zarr`, the competition input mount location, and bounded/streaming NGFF frame
iteration. Version 10 includes all fixes and is the version to resume.

After Kaggle reports `COMPLETE`, submit the exact kernel output:

```bash
kaggle kernels status sota1111/biohub-gpt-cycle-2-champion
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -k sota1111/biohub-gpt-cycle-2-champion -v 10 \
  -f submission.csv -m "SOT-2046 cycle-2 champion"
kaggle competitions submissions \
  -c biohub-cell-tracking-during-development
```

If version 10 reaches `ERROR`, download its log with:

```bash
kaggle kernels output sota1111/biohub-gpt-cycle-2-champion -p /tmp/biohub-v10
```

Then fix the reported runtime error, rerun
`python scripts/verify_cycle2_champion.py`, push the next version, and substitute
that completed version in the submit command. The prior accepted biohub-gpt
submission remains ref `55053037` (`COMPLETE`, public score `0.000`); it predates
this cycle-2 package and is not claimed as version-10 proof.
