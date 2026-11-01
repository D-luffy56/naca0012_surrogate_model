# Combined engineering report: fixed and self-adaptive POD–PINNs

## 1. Objective

The study constructs a reduced-order aerodynamic surrogate for NACA0012. Reynolds
number and angle of attack are mapped to 25 POD coefficients, which reconstruct the
nondimensional velocity, pressure, and eddy-viscosity fields. Physics is enforced on
the reconstructed fields during training rather than applied as a post-processing
correction.

Two approaches are compared:

1. **Fixed weighting:** data, continuity, momentum, and boundary multipliers remain
   `1:1:1:1`.
2. **Self-adaptive weighting:** a 64×64 network jointly learns positive global weights
   for the same four loss components.

## 2. Data and reduced representation

The dataset contains 40 training and 8 held-out validation cases on one common
37,800-cell OpenFOAM mesh. Four nondimensional fields are retained: streamwise
velocity `u`, cross-flow velocity `w`, kinematic pressure `p`, and turbulent eddy
viscosity `nut`.

A cell-volume-weighted POD is fitted only to training snapshots. Rank 25 captures
99.9999998% of the training fluctuation energy. Projection of held-out CFD snapshots
onto this training basis gives oracle errors of 0.093% (`u`), 0.260% (`w`), 0.546%
(`p`), and 0.964% (`nut`). The POD basis therefore has sufficient capacity; most
remaining error comes from predicting coefficients at unseen parameters.

## 3. Physics operator

The native mesh remains the data truth. Delaunay interpolation creates a 251×201 grid
over `x/c=[-0.5,2]` and `z/c=[-0.5,0.5]` for finite-difference derivatives and
visualization. Unsafe stencils and a 0.02-chord wall band are excluded.

Manufactured first/second derivative errors remain below 0.016%. Across all 48 source
cases, maximum normalized discrete residual imbalances are approximately 6.16%
(continuity), 5.52% (x-momentum), and 3.03% (z-momentum). Delaunay round-trip errors
remain below 2.11% for every retained field.

## 4. Fixed-weight approach

All networks use two `tanh` hidden layers, L-BFGS only, L2=`1e-4`, rank 25, and 500
fixed optimizer iterations.

| Architecture | u | w | p | nut | Mean |
|---|---:|---:|---:|---:|---:|
| 32×32 | 0.138% | 4.803% | 4.269% | 2.220% | 2.858% |
| **64×64** | 0.111% | **2.087%** | **2.002%** | **1.505%** | **1.426%** |
| 128×128 | **0.105%** | 2.919% | 2.852% | 1.586% | 1.865% |

The 64×64 model is the best fixed-weight architecture. The 128×128 network fits the
training set slightly more closely but generalizes worse for cross-flow velocity and
pressure.

## 5. Self-adaptive approach

For component losses `data`, `continuity`, `momentum`, and `boundary`, trainable log
variances are used:

\[
L_{adaptive}=\sum_i\left[e^{-s_i}L_i+s_i\right]+10^{-4}\lVert\theta\rVert_2^2,
\qquad \lambda_i=e^{-s_i}.
\]

All weights begin at one. Log variances are smoothly bounded to `[-5,5]` to prevent
numerical collapse under L-BFGS.

| Component | Initial weight | Learned weight |
|---|---:|---:|
| Data | 1.000 | 1.421 |
| Continuity | 1.000 | 148.396 |
| Momentum | 1.000 | 148.372 |
| Boundary | 1.000 | 148.083 |

Continuity, momentum, and boundary weights reach the imposed stability ceiling. The
adaptive model therefore indicates a strong preference for physics enforcement, but
the final values must be described as a bounded solution rather than an unconstrained
optimum.

## 6. Direct comparison

| Metric | Fixed 64×64 | Self-adaptive 64×64 |
|---|---:|---:|
| u relative L2 | **0.111%** | 0.117% |
| w relative L2 | 2.087% | **1.204%** |
| p relative L2 | 2.002% | **1.430%** |
| nut relative L2 | **1.505%** | 1.969% |
| Mean field error | 1.426% | **1.180%** |
| Continuity RMS | 0.03273 | **0.03040** |
| x-momentum RMS | **0.02572** | 0.02862 |
| z-momentum RMS | 0.02331 | **0.02159** |
| Pressure-only Cl MAE | 0.01005 | **0.00848** |
| Pressure-only Cd MAE | 0.00303 | **0.00082** |

The adaptive model reduces mean validation field error by 17.3%. Its strongest gains
are in cross-flow velocity, pressure, continuity, z-momentum, and pressure-derived
forces. It slightly degrades streamwise velocity, eddy viscosity, and x-momentum.

## 7. Engineering conclusion

The fixed 64×64 network is the simpler and more interpretable baseline. The
self-adaptive 64×64 model provides the lowest overall held-out field error and better
pressure-derived force estimates, making it the preferred accuracy model within the
sampled Reynolds/AoA envelope.

The work should be described as a validated hybrid POD–PINN engineering surrogate,
not as a new fundamental PINN methodology. It does not establish extrapolation
accuracy, wall-shear accuracy, or total-drag accuracy. The adaptive-weight saturation
and the CFD force-stability flags must remain visible in any presentation.

## 8. Extended diagnostic analysis

The case-wise error heatmap shows that the adaptive improvement is not uniform. It
substantially improves the difficult fixed-model case V03 and the cross-flow velocity
of the high-incidence cases V06 and V08. Conversely, V01 becomes worse in `w`, `p`,
and `nut`, and `nut` error increases for several cases. Averaging every case and field
equally gives 1.699% for the fixed model and 1.414% for the adaptive model. This
case-wise mean differs from the global volume-weighted aggregate because every case
receives equal weight in the heatmap statistic.

Surface-pressure plots at low, medium, and high incidence show that both models
reproduce the suction peak and downstream pressure recovery closely. The adaptive
model provides the better pressure-force parity overall, consistent with its lower
pressure-only lift and drag MAE. These curves also reveal local leading-edge
differences that are not visible in a single global pressure error.

Wake profiles at `x/c = 0.5, 1.0, 1.5, 2.0` show that agreement is strongest near the
airfoil and that model differences grow downstream as small coefficient errors alter
the wake location and deficit. Neither weighting strategy dominates every velocity
component at every station; the adaptive gain is an aggregate improvement rather
than uniform pointwise superiority.

Spatial residual maps localize the largest continuity and momentum imbalances near
the leading edge and in the wake. Repeating arc patterns are associated with the
Delaunay-interpolation/finite-difference representation, which is why the report uses
the reference-CFD residual floor and excludes the first 0.02 chord near the wall. The
adaptive model reduces continuity and z-momentum RMS overall but produces a stronger
x-momentum wake residual in V06, matching the scalar residual comparison.

The POD spectrum confirms that most fluctuation energy is concentrated in the first
few modes, while the retained rank 25 provides a negligible truncation floor. Leading
coefficient parity plots demonstrate that field accuracy is controlled mainly by the
first energetic coefficients; large standardized errors in very weak high-order modes
should therefore not be interpreted as equally important physical errors.

