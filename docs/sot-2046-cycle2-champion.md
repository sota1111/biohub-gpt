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

Pending execution. After packaging, push and submit the exact kernel version:

```bash
./scripts/package_kernel.sh
kaggle kernels push -p dist/kaggle-kernel
kaggle kernels status sota1111/biohub-gpt-cli-baseline
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -k sota1111/biohub-gpt-cli-baseline -v <VERSION> \
  -f submission.csv -m "SOT-2046 cycle-2 champion"
kaggle competitions submissions \
  -c biohub-cell-tracking-during-development
```

The final kernel version, submission reference, status, score, and package
commit are recorded here after Kaggle accepts the run.
