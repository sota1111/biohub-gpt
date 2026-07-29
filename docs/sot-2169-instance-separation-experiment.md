# SOT-2169 touching-nuclei instance separation

## Decision

`touching-watershed-v1` was **not promoted**. The active champion remains
`appearance-motion-link-v1`, including its `adaptive-local-peaks-v1` detection
model. `config/champion.json` is intentionally unchanged, so local and Kaggle
execution behavior is unchanged.

The candidate remains available as an explicitly selected detection model for
future experiments. It combines an anisotropic Euclidean distance transform,
local-maxima markers, marker-controlled watershed, component-size/intensity
constraints, and a relative marker-strength separation confidence.

## Fixed split and small-N screen

The incumbent was frozen from `config/champion.json`. Three parameter sets were
screened on `two-x`, `two-y`, and `isolated`; only the deterministic top-ranked
configuration was evaluated on the disjoint `three-cluster`, `anisotropic-z`,
and `confirm-isolated` split.

The top screen configuration used marker distance `1.25`, separation confidence
`0.3`, physical voxel spacing `[2, 1, 1]`, and an eight-voxel component floor.

| Split | Model | Detection F1 | Composite | Over-split | Under-split | Runtime (s) | Peak memory (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen | incumbent | 0.750000 | 0.825000 | 0 | 2 | 0.003719 | 490251 |
| screen | candidate | 0.750000 | 0.825000 | 0 | 2 | 0.028159 | 917491 |
| confirm | incumbent | 0.444444 | 0.611111 | 0 | 3 | 0.003454 | 486862 |
| confirm | candidate | 0.444444 | 0.611111 | 0 | 3 | 0.028188 | 913782 |

Runtime and memory are diagnostic measurements from one local run and may vary
slightly. Accuracy and split membership are deterministic. The candidate did
not clear the `+0.01` screen composite gate, did not reduce under-splitting, and
used materially more compute, so promotion was rejected.

## Provenance and downstream use

The complete configurations, per-case counts, metrics, resource measurements,
split membership, and decision are stored in
`artifacts/sot-2169-instance-separation-experiment.json`. The candidate returns
the same z/y/x centroid contract consumed by existing linking and division
evaluation, so the artifact can seed later detector or division experiments
without changing submission schema.

Reproduce the ledger:

```bash
python -m biohub_baseline.cli evaluate-instance-separation \
  --output artifacts/sot-2169-instance-separation-experiment.json
```

Exit status `2` is expected for this recorded non-promotion decision.
