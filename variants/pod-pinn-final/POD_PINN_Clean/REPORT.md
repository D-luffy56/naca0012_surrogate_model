# POD–PINN engineering report

## Outcome

The fixed 64×64 architecture is the best of the three prescribed networks. On the
eight untouched validation cases, its native-mesh relative L2 errors are:

| Field | 32×32 | 64×64 | 128×128 |
|---|---:|---:|---:|
| `u` | 0.138% | **0.111%** | 0.105% |
| `w` | 4.803% | **2.087%** | 2.919% |
| `p` | 4.269% | **2.002%** | 2.852% |
| `nut` | 2.220% | **1.505%** | 1.586% |
| Mean | 2.858% | **1.426%** | 1.865% |

The 64×64 model also gives a pressure-only lift-coefficient MAE of 0.0100 and a
pressure-only drag-coefficient MAE of 0.00303. These are not total force coefficients.

## Data and POD

- 40 training and 8 validation cases, all with the same 37,800-cell mesh hash.
- Validation Reynolds numbers and angles were never included in model fitting.
- Cell-volume-weighted, four-field POD was fit using training snapshots only.
- Fixed rank: 25. Captured training fluctuation energy: 99.9999998%.
- Validation oracle projection floor: 0.093% (`u`), 0.260% (`w`), 0.546% (`p`),
  and 0.964% (`nut`).

This projection floor shows that the POD basis is not the dominant validation error;
the remaining error primarily comes from mapping `(Re, AoA)` to POD coefficients.

## Delaunay and physics operator

The native mesh remains the POD/data representation. A 251×201 structured grid over
`x/c = [-0.5, 2]`, `z/c = [-0.5, 0.5]` is built through Delaunay interpolation for
physics derivatives and plots. A 0.02-chord wall band and unsafe finite-difference
stencils are excluded.

Operator checks passed:

- Manufactured first/second derivative relative RMS errors: 0.0016%–0.0151%.
- Across all 48 reference CFD cases, normalized residual imbalance has maxima of
  6.16% for continuity, 5.52% for x-momentum, and 3.03% for z-momentum.
- Delaunay grid round-trip relative L2 maxima: 0.694% (`u`), 2.10% (`w`),
  1.82% (`p`), and 0.684% (`nut`).

These checks are why the interpolated grid is used only as a physics operator, not as
the field-data truth.

## Model and loss

Each network maps normalized `(Re, AoA)` to 25 standardized POD coefficients. The
four locked losses are:

1. POD coefficient data MSE
2. continuity residual MSE
3. RANS momentum residual MSE with `1/Re + nut` effective viscosity
4. boundary-field MSE through a training-only affine boundary reconstruction

All four multipliers are 1.0. L2 regularization is `1e-4`. L-BFGS is the only
optimizer. The three models use identical preprocessing, collocation points, loss
definitions, iteration count, and evaluation cases.

## Engineering interpretation

The 128×128 network fits training data slightly more closely but generalizes worse
than 64×64 for cross-flow velocity and pressure. The 64×64 network is therefore the
recommended checkpoint. Its validation physics residuals remain close to the
finite-difference residual floor observed in the source CFD fields.

The result should be used as an interpolating reduced-order surrogate within the
sampled Reynolds/AoA envelope. It is not evidence of reliable extrapolation, direct
wall-shear prediction, or total drag prediction.

