# Native six-field POD–GPR

`src/fit_native_regressor.py` with `--kind gpr` fits a Matérn 5/2 Gaussian
process to the 16 POD coefficients from 40 native-mesh CFD cases. The saved
regressor, basis, original-validation predictions, fresh-CFD predictions,
and errors are in `outputs/native_pod_gpr/`. The common six-field snapshot
source is `outputs/native_sst/`.

Five-fold development results are in `outputs/native_pod_regressor_cv/`.
Additional case plots and analyses are in `results/gpr_field_validation/`
and `results/gpr_validation/`. The GPR has no SST physics loss. All seven
fresh cases met residual control but failed the stricter force-stability
criterion. The fresh set has now been inspected and is no longer untouched.
