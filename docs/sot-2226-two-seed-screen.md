# SOT-2226 two-seed 3-D logit-blend screen

Exact Kaggle notebook v32 and both independent model seeds were pinned by checksum. The repository implementation runs both detection heads on the same normalized, overlap-weighted 3-D patch grid, affine-aligns secondary logits, blends them, and converts the result into anisotropy-aware node candidates. Tracking remains fixed at `daughter-geometry-v1`.

Only screen embryo `44b6_0113de3b` was made visible to the runner. The script fails closed if confirm embryo `6bba_05b6850b` is present. Two independent executions produced identical node hashes and graph metrics for every candidate.

| Blend weight | Nodes | Edges | Edge Jaccard | Division Jaccard | Score | Runtime | Peak GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 24,780 | 15,855 | 0.000 | 1.000 | 0.100 | 372.45 s | CPU baseline |
| 0.350 | 1,133 | 487 | 0.000 | 1.000 | 0.100 | 41.25 s | 294.33 MiB |
| 0.475 | 928 | 401 | 0.000 | 1.000 | 0.100 | 40.42 s | 294.33 MiB |
| 0.600 | 833 | 379 | 0.000 | 1.000 | 0.100 | 40.77 s | 294.33 MiB |

The fixed screen contains one sparse, non-division series, so medium/dense/development-stage and division strata are unavailable rather than inferred from confirm data. All candidates tie the incumbent and miss the pre-fixed `+0.01` score gate. No winner is emitted, nothing is forwarded to SOT-2227, and `config/champion.json` is unchanged. The three-candidate run took 122.44 seconds at 294.33 MiB peak GPU memory, comfortably below Kaggle's 12-hour and 12-GiB limits.

Provenance is recorded in `artifacts/sot-2226-provenance.json`; machine-readable results are in `artifacts/sot-2226-two-seed-screen.json`.
