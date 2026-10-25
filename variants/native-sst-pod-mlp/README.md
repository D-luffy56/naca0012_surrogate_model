# Native SST POD–MLP

The training pipeline is `src/build_sst_snapshots.py` followed by
`src/train_sst_pod.py`. The 40/7 processed native snapshots are in
`outputs/native_sst/`. The 16-mode basis, saved 64–64 ReLU SST checkpoint,
validation predictions, loss history, per-case errors, and residual
diagnostics are in `outputs/sst_pod_model/`. Fresh-CFD results are in
`results/native_sst/historical_fresh_comparison.json`.

The SST and wall losses approximate the equations using graph derivatives
and CFD wall targets. They do not reproduce OpenFOAM's finite-volume SST
operators exactly. Training from the native snapshots also needs the raw
mesh export for graph-derivative construction. The fresh seven cases failed
the stricter force-history stability criterion.
