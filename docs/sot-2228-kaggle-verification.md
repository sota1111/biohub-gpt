# SOT-2228 latest verified artifact

## Selection decision

The cycle-5 screen and confirm chain did not produce an eligible candidate.
`artifacts/sot-2226-two-seed-screen.json` records `screen_passed: false`,
`confirm_accessed: false`, and no winner or forward target. SOT-2227 therefore
kept the production champion unchanged. The submission target is the current
`daughter-geometry-v1` champion in `config/champion.json`, not an unconfirmed
neural candidate.

The machine-readable audit is
`artifacts/sot-2228-kaggle-verification.json`. It binds the SOT-2225 baseline,
SOT-2226 screen and provenance ledgers, champion metadata, generated
submission, and offline package through SHA-256 values.

## Offline execution gate

Run:

```bash
python scripts/verify_latest_artifact.py
```

The verifier fails closed on an invalid predecessor decision, runs the real
`exec.sh` entrypoint twice with network proxies pointed at a closed local port,
and checks deterministic bytes, the exact CSV schema, node/edge references,
lineage time direction, runtime, and package settings. The Kaggle package has
internet disabled, GPU enabled, and an 11-hour application timeout within the
competition's 12-hour limit.

## Kaggle evidence

Kernel: `sota1111/biohub-gpt-sot-2228-verified-artifact`, version 1. Kaggle
accepted the push and returned the public kernel URL. At the final observation
(`2026-08-01T15:42:49Z`) the asynchronous GPU run was still
`KernelWorkerStatus.RUNNING`. A competition submission cannot be created until
that version completes, so submission ref/status, public score, and observed
rank are explicitly `null` in the ledger rather than inferred from another
lineage. The kernel URL, concrete external wait reason, and exact resume
commands are recorded there.

To reproduce the external steps:

```bash
bash scripts/package_kernel.sh
kaggle kernels push -p dist/kaggle-kernel
kaggle kernels status sota1111/biohub-gpt-sot-2228-verified-artifact
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -k sota1111/biohub-gpt-sot-2228-verified-artifact \
  -v 1 -f submission.csv -m "SOT-2228 latest verified artifact"
kaggle competitions submissions -c biohub-cell-tracking-during-development
```
