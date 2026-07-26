# Biohub data and submission contract

Verified with the Kaggle API on 2026-07-26 for competition
`biohub-cell-tracking-during-development`.

- Test inputs are Zarr stores named `<dataset>.zarr`; array `0` has axes
  `t,z,y,x`.
- `sample_submission.csv` has the exact ordered columns
  `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- A node row has positive `node_id`, time and coordinates; its source/target are
  `-1`.
- An edge row has source/target node IDs from the same dataset; all node fields
  are `-1`.
- Submission IDs are unique contiguous integers starting at zero.
- The competition accepts notebook submissions, so `enable_internet=false` and
  the executable must write `/kaggle/working/submission.csv`.

Fetch the competition archive with `./scripts/fetch_data.sh`. Generate through
`./exec.sh <test-dir> <output.csv>` and validate independently with
`biohub-baseline validate <output.csv>`.
