# Clean POD–PINN study: NACA0012

This directory is a clean rebuild from the 48 source CFD arrays. No previous model,
checkpoint, or derived POD basis was reused.

## Locked model specification

- Inputs: Reynolds number and angle of attack
- Outputs: 25 POD coefficients for nondimensional `u`, `w`, `p`, and `nut`
- Architectures: `2-32-32-25`, `2-64-64-25`, `2-128-128-25`
- Activation: `tanh`
- Optimizer: L-BFGS only, 500 fixed iterations
- L2 coefficient: `1e-4`
- Data, continuity, momentum, and boundary multipliers: all `1.0`

The POD basis is computed on the common native unstructured mesh with cell-volume
weights. Delaunay interpolation supplies one common structured grid for finite-
difference physics derivatives and visualization. It is not used as the POD truth.

## Reproduce

Run from the parent directory that also contains
`NACA0012_PINN_Study/data/native`:

```bash
python POD_PINN_Clean/preprocess_pod.py
python POD_PINN_Clean/validate_delaunay_operator.py
python POD_PINN_Clean/build_pod_pinn_operator.py
python POD_PINN_Clean/train_fixed_pod_pinn.py
python POD_PINN_Clean/make_results.py
```

The final comparison is in `artifacts/results`, and the best checkpoint is
`artifacts/models/pod_pinn_2x64_rank25.pt`.

## Self-adaptive extension

An additional 64×64 model learns positive global weights for the data,
continuity, momentum, and boundary losses using bounded homoscedastic uncertainty
weighting. Its checkpoint is
`artifacts/models/pod_pinn_2x64_rank25_self_adaptive.pt` and its full comparison is
in `SELF_ADAPTIVE_REPORT.md` and `artifacts/self_adaptive_results`.

The adaptive model reduces mean held-out field error from 1.426% to 1.180%, while
trading some `nut` accuracy for improved `w`, pressure, and pressure-derived force
metrics. The physics/boundary weights reach the imposed stability ceiling, which is
reported explicitly rather than treated as an unconstrained optimum.

## Extended analysis

`make_extended_analysis.py` adds case-wise validation heatmaps, error dependence on
Reynolds number and AoA, surface pressure distributions, pressure-force parity,
POD spectra and leading modes, POD-coefficient parity, three held-out contour sets,
wake profiles, spatial physics-residual maps, and adaptive weighted-loss
contributions. Outputs are stored in `artifacts/extended_analysis`.

## Important limitations

- The structured physics grid excludes a 0.02-chord band around the airfoil because
  a uniform grid does not resolve the wall-function layer.
- The force comparison is pressure-only; viscous drag is not inferred from the
  reduced boundary representation.
- All 48 CFD cases report solver convergence, but only 10 pass the conservative
  force-tail stability flag in the supplied manifest. Treat force metrics accordingly.

