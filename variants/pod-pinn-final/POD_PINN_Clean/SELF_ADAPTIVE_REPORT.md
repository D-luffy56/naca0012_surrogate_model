# Self-adaptive POD–PINN extension

## What changed

The original best architecture, `2-64-64-25`, was retrained from scratch with
self-adaptive global weights for the four existing loss components. The POD basis,
training/validation split, Delaunay grid, collocation points, activation, L2
coefficient, network initialization seed, and 500-iteration L-BFGS configuration were
kept unchanged.

For component losses (L_i\in\{L_d,L_c,L_m,L_b\}), trainable log variances (s_i)
were introduced:

\[
L_{adaptive}=\sum_i\left[e^{-s_i}L_i+s_i\right]+10^{-4}\lVert\theta\rVert_2^2,
\qquad \lambda_i=e^{-s_i}.
\]

All four weights began at 1. To prevent numerical collapse under L-BFGS, the learned
log variances were smoothly bounded to ([-5,5]), giving weights in
([e^{-5},e^5]).

## Learned weights

| Component | Initial | Final |
|---|---:|---:|
| Data | 1.000 | 1.421 |
| Continuity | 1.000 | 148.396 |
| Momentum | 1.000 | 148.372 |
| Boundary | 1.000 | 148.083 |

The three physics-related weights reached the imposed upper stability bound. This is
important: the model clearly preferred stronger physics/boundary enforcement, but the
reported values are a bounded adaptive solution rather than an unconstrained optimum.

## Held-out validation comparison

| Metric | Fixed 64×64 | Self-adaptive 64×64 | Better |
|---|---:|---:|---|
| `u` relative L2 | 0.111% | 0.117% | Fixed |
| `w` relative L2 | 2.087% | **1.204%** | Adaptive |
| `p` relative L2 | 2.002% | **1.430%** | Adaptive |
| `nut` relative L2 | **1.505%** | 1.969% | Fixed |
| Mean field error | 1.426% | **1.180%** | Adaptive |
| Continuity RMS | 0.03273 | **0.03040** | Adaptive |
| x-momentum RMS | **0.02572** | 0.02862 | Fixed |
| z-momentum RMS | 0.02331 | **0.02159** | Adaptive |
| Pressure-only Cl MAE | 0.01005 | **0.00848** | Adaptive |
| Pressure-only Cd MAE | 0.00303 | **0.00082** | Adaptive |

The adaptive model reduces the mean validation field error by approximately 17.3%.
Its main gains are in cross-flow velocity, pressure, continuity, z-momentum, and
pressure-derived force coefficients. It trades away some `nut` accuracy and slightly
worsens the x-momentum residual.

## Interpretation

This is a useful ablation, not evidence that adaptive weighting is universally better.
The result shows that equal numerical multipliers were not the best balance for this
particular reduced-order problem. The saturation of three weights also demonstrates
why the training history must be shown alongside the final accuracy table.

Use `artifacts/models/pod_pinn_2x64_rank25_self_adaptive.pt` for the adaptive model.
The fixed 64×64 checkpoint remains the cleaner baseline; both should be presented.

