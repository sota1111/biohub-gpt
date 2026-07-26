# biohub-gpt

Reproducible baseline for the Kaggle competition
`biohub-cell-tracking-during-development`. It detects bright 3-D connected
components per frame, then associates detections with a sparse mutual-kNN
temporal graph and constrained continuation/division optimization.

## Reproduce the champion

Python 3.10+ is required. From a clean checkout:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
./scripts/reproduce_baseline.sh
```

The immutable starting point is `baseline-v1`. Its parameters and reproduction
command are in `config/champion.json`; recorded metrics are in
`artifacts/champion-metrics.json`. The first metrics use deterministic synthetic
volumes because Kaggle provides no public labels for the test series. They test
the evaluator and establish an auditable baseline; labelled train/validation
metrics should replace them without changing the gate.

Download the competition inputs with `./scripts/fetch_data.sh`; the verified
schema and node/edge invariants are recorded in `docs/data-contract.md`.

## Generate a submission

```bash
./exec.sh /path/to/test submission.csv
```

The input directory contains one Zarr array per dataset, shaped `t,z,y,x`.
`exec.sh` produces and validates the required columns:
`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`. Validation rejects
duplicate nodes, dangling/self edges, invalid row types, and non-contiguous IDs.

## Screen → confirm

`config/evaluation-gates.json` fixes seed `1988`, a deterministic 40% screen
split, metric thresholds, and required deltas over the current champion. A
candidate is evaluated on confirm only after its screen composite improves by
at least `0.01`; it is promoted only when confirm improves by at least `0.005`
and both detection/edge floors pass.

```bash
biohub-baseline split --datasets datasets.txt --output split.json --seed 1988
biohub-baseline promote \
  --candidate artifacts/candidate.json \
  --champion artifacts/champion-metrics.json \
  --output artifacts/promotion.json
```

The promotion command exits `0` only for promotion and `2` otherwise, so CI or
an improvement runner can make the decision mechanically.

## Temporal lineage model

`config/champion.json` records every candidate-graph and lineage cost parameter:
maximum motion, mutual-kNN size, density radius/weight, birth/death/division
costs, and division distance/separation gates. Each target receives at most one
parent, each source at most two children, and second-child links must pass the
division gate. Because edges only connect adjacent frames, the result cannot
contain cycles or time-reversed links.

The fixed-detection screen/confirm comparison is reproducible with:

```bash
python -m biohub_baseline.cli evaluate-lineage \
  --output artifacts/sot-1990-lineage-experiment.json
```

The artifact records baseline/candidate edge precision, recall, F1, division
F1, composite score, integrity errors, gate decision, and the exact parameters.

## Kaggle offline package

`kaggle/kernel-metadata.json` disables internet and `kaggle/notebook.py` invokes
the same `exec.sh`, with an 11-hour timeout below Kaggle's 12-hour constraint.
Bundle the repository files with the kernel before `kaggle kernels push`.
`./scripts/package_kernel.sh` builds the complete upload directory at
`dist/kaggle-kernel`.
