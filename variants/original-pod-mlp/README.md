# Original POD–MLP

The original 40-case, 4-mode common-grid model is already part of this branch's
inherited `main` tree. The core training source is `src/pod_naca0012.py`,
inference is `src/predict.py`, the POD basis and MLP are in
`models/baseline-v1/`, and the processed snapshots are in
`data/processed/baseline-v1/`. The corrected seven-case validation errors
are in `results/summary.json` and `results/baseline-v1/validation_field_errors.csv`.
The same repository contains the separate CL/CD companion MLP.
