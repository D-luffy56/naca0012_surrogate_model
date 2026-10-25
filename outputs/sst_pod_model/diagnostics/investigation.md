# Why SST physics loss did not improve the POD surrogate

This diagnostic reuses the *same* graph residual implementation as training.
It evaluates the 40 training cases and all seven held-out CFD cases. Scores
below are normalized equation-residual mean squares, **not field L2 errors**;
lower means closer to satisfying this approximate numerical physics check.

## 1. Check the checker on CFD

| Seven held-out CFD cases | Physics score |
| --- | ---: |
| CFD solutions | 0.369 |
| Native POD–MLP, data only | 5.110 |
| Native POD–MLP, SST loss | 5.197 |

The CFD score is not zero. The graph derivative and approximate wall distance
do not reproduce OpenFOAM's finite-volume operators and wall functions. This
sets a numerical reference floor for interpreting model scores. The SST-trained
model scores slightly worse than the data-only model on unseen cases.

The turbulent-viscosity closure term accounts for about 85% of the data-only
model's normalized validation score: its normalized squared component is
26.0, versus 0.63–2.24 for each other equation. The CFD closure component is
0.076. V02 and V04, both at 7°, have the highest model closure residuals.

## 2. Did it learn the training equations?

| 40 training cases | Physics score |
| --- | ---: |
| CFD solutions | 0.8494 |
| Native POD–MLP, data only | 0.8565 |
| Native POD–MLP, SST loss | 0.8560 |

The training physics score improved by only about 0.06%. That small gain did
not transfer to the seven unseen cases. The SST loss also slightly worsened
held-out u, w, p field errors (see `../validation_report.md`).

## 3. Where did field errors change?

The table shows relative L2 errors on the *native mesh* within each region.
The near-wall band is within 0.05 chord of the airfoil; the wake is x/c from
1 to 3 with |z/c| <= 0.5. The SST result is slightly worse in each row.

| Region and field | Data only | SST loss |
| --- | ---: | ---: |
| Near-wall u | 0.625% | 0.629% |
| Near-wall w | 1.690% | 1.702% |
| Near-wall p | 3.618% | 3.635% |
| Wake u | 1.100% | 1.102% |
| Wake w | 2.511% | 2.533% |
| Wake p | 1.218% | 1.224% |

The PDE loss is evaluated at 33,692 mesh points and excludes 4,918 points
where the native graph stencil is unsuitable, including the closest wall
region. The wall loss does use the separate exported boundary values.

## 4. Is the POD basis the bottleneck?

Projecting validation CFD fields directly onto the saved 16 modes gives the
best reconstruction available in this fixed basis *if the coefficients were
known*. Compare that with the data-only MLP's predictions:

| Field | POD projection floor | Data-only MLP error |
| --- | ---: | ---: |
| u | 0.460% | 0.618% |
| w | 0.151% | 1.902% |
| p | 0.155% | 3.426% |

The 16-mode basis has substantial room to improve w and p. The larger error
for those fields comes mainly from predicting the coefficients for unseen
conditions, rather than from an absolute inability of this basis to express
their validation fields. The u error is closer to its projection floor.

## 5. Were training signals balanced?

At the last epoch, the weighted scalar losses were data 0.0000387, wall
0.0001876, and physics 0.0000853. Scalar loss sizes alone do not tell us how
strongly network weights are updated. On a representative four-case batch,
the corresponding gradient norms were data 0.0457, wall 0.00147, and physics
0.00598. The data gradient was about 7.6 times the physics gradient. Their
cosine similarity was +0.20, so this sample does not show a strongly opposing
physics signal; it shows a relatively weak one.

## Next experiment

First validate a more faithful SST residual against CFD, especially wall
distance, finite-volume derivatives, and the omega wall function. Then test a
network that computes turbulent viscosity from predicted k and omega through
the SST closure instead of predicting all three independently. Compare it
with a data-only control using the same native mesh and POD representation.
Choose settings using training-only splits; the seven cases in this report
have now been inspected and should not be treated as a fresh untouched test
set for future tuning.

Machine-readable values: `summary.json` and `validation_residuals_by_case.csv`
in this directory.
