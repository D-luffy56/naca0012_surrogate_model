# Selected settings: 32–32 tanh, L-BFGS, L2 = 1e-4

The saved SST checkpoint and its 16-mode basis, predictions, loss history,
and validation summary are in `outputs/sst_pod_tanh32_lbfgs_l2_1e-4/`.
Training source is `src/train_sst_pod.py`, with `--hidden-units 32 32
--activation tanh --optimizer lbfgs --l2 0.0001`. The companion data-only
32–32 estimator had not been saved; `src/fit_native_regressor.py` refits it
and its new artifact and predictions are in `outputs/native_pod_tanh_data/`.
Use each artifact's own summary for measured errors. The SST loss and wall
condition are approximations; native-mesh retraining needs the raw CFD mesh.
