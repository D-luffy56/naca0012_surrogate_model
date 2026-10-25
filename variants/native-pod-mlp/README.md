# Native six-field POD–MLP

`src/build_sst_snapshots.py` prepares the 40/7 native-mesh dataset. The
`src/fit_native_regressor.py` `mlp64` option fits a 64–64 ReLU MLP on the
saved 16-mode POD basis. `outputs/native_sst/` contains the common source
snapshots; `outputs/native_pod_mlp/` contains the basis, refitted estimator,
original-validation and fresh-CFD predictions, and measured errors.

The historical baseline predictions were saved without the sklearn
estimator. This branch's estimator is a new refit, and its metrics must be
read from its own summary. The five-fold development comparison is in
`outputs/native_pod_regressor_cv/`. A full re-extraction from CFD still
requires the raw OpenFOAM archive.
