# biohub-gpt

Reproducible baseline for the Kaggle competition
`biohub-cell-tracking-during-development`. It detects bright 3-D connected
components per frame, then associates detections with a sparse mutual-kNN
temporal graph and constrained continuation/division optimization.

The champion detector uses a locally adaptive residual threshold, 3-D peak
suppression, intensity-weighted sub-voxel refinement, and distance-based
duplicate suppression. Its fixed-tracker screen/confirm comparison covers
sparse, dense, division-neighborhood, and noisy synthetic volumes:

```bash
python -m biohub_baseline.cli evaluate-detection \
  --output artifacts/sot-1989-detection-experiment.json
```

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

## Appearance and motion links

The champion link cost also combines a clipped local intensity/shape descriptor
with a short-term constant-velocity prediction. Missing or unusable patches
fall back to coordinate, density, and motion terms; a source without sufficient
history falls back to coordinate, density, and appearance terms. The small
weight screen includes an acceleration candidate, while the promoted setting
keeps acceleration disabled because the simpler constant-velocity candidate
tied for the best screen result.

The fixed screen/confirm comparison covers crowded crossings, temporary missing
appearance, and zero motion:

```bash
python -m biohub_baseline.cli evaluate-link-features \
  --output artifacts/sot-2044-appearance-motion-experiment.json
```

The recorded confirm result improves edge F1 from `0.714286` to `1.0` and
reduces identity switches from four to zero across the stratified cases.

## Kaggle offline package

`kaggle/kernel-metadata.json` disables internet and `kaggle/notebook.py` invokes
the same `exec.sh`, with an 11-hour timeout below Kaggle's 12-hour constraint.
Bundle the repository files with the kernel before `kaggle kernels push`.
`./scripts/package_kernel.sh` builds the complete upload directory at
`dist/kaggle-kernel` and records a SHA-256 manifest. The cycle-2 champion gate
audits every predecessor promotion decision, runs the real `exec.sh` twice with
network access failed closed, validates schema and graph references, and records
determinism/runtime/memory/package hashes:

```bash
python scripts/verify_cycle2_champion.py
```

The machine-readable result is
`artifacts/sot-2046-exec-compatibility.json`; the submission result and exact
rerun commands are recorded in `docs/sot-2046-cycle2-champion.md`.

The latest cycle audit and Kaggle package contract can be reproduced with:

```bash
python scripts/verify_latest_artifact.py
```

It rejects unconfirmed candidates, selects the current champion when the latest
screen has no winner, runs the real entrypoint twice with internet failed closed,
and records source/model/package hashes in
`artifacts/sot-2228-kaggle-verification.json`.
