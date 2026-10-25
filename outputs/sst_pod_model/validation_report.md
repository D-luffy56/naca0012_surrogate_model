# Native-mesh SST POD validation

Training cases: 40; validation cases: 7; native mesh points: 38610; airfoil-wall points: 400.

The baseline and SST-trained models use the same six-field POD basis and network size.
Relative L2 errors below are aggregated over the seven validation cases.

| Field | Data-only baseline | SST-regularized | Change |
| --- | ---: | ---: | ---: |
| u | 0.618% | 0.620% | +0.001% |
| w | 1.902% | 1.917% | +0.015% |
| p | 3.426% | 3.443% | +0.017% |
| k | 5.885% | 5.900% | +0.015% |
| omega | 1.130% | 1.134% | +0.004% |
| nut | 4.526% | 4.531% | +0.005% |

## Comparison on the original baseline grid

All three predictions below use the same held-out CFD values and the original 192 × 128 sampling grid.

| Field | Original 3-field baseline | Native 6-field data-only | Native SST-regularized |
| --- | ---: | ---: | ---: |
| u | 0.843% | 0.274% | 0.274% |
| w | 1.622% | 2.104% | 2.119% |
| p | 3.725% | 1.876% | 1.885% |

## Airfoil wall errors

Wall RMSE divided by each field's training standard deviation, against the separate CFD boundary exports:

| Field | Data-only baseline | SST-regularized |
| --- | ---: | ---: |
| u | 0 | 0 |
| w | 0 | 0 |
| p | 0.1005 | 0.1009 |
| k | 0.06193 | 0.06212 |
| omega | 0.09312 | 0.09325 |
| nut | 5.447e-05 | 5.443e-05 |

## Per-case field errors

| Case | Re | AoA | Baseline u | SST u | Baseline w | SST w | Baseline p | SST p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V01 | 2500000 | 2° | 1.019% | 1.021% | 0.613% | 0.579% | 4.190% | 4.293% |
| V02 | 2500000 | 7° | 0.224% | 0.227% | 0.577% | 0.625% | 2.258% | 2.312% |
| V03 | 3500000 | 2° | 0.951% | 0.953% | 2.300% | 2.318% | 5.285% | 5.339% |
| V04 | 3500000 | 7° | 0.541% | 0.543% | 3.162% | 3.181% | 5.861% | 5.871% |
| V05 | 4500000 | 5° | 0.319% | 0.316% | 0.921% | 0.887% | 3.615% | 3.583% |
| V06 | 4500000 | 11° | 0.401% | 0.406% | 1.998% | 2.023% | 1.933% | 1.957% |
| V07 | 5500000 | 5° | 0.354% | 0.350% | 0.805% | 0.775% | 3.054% | 3.024% |

## Interpretation

Least-squares graph derivatives and CFD wall-value loss do not reproduce OpenFOAM finite-volume operators or nonlinear omega wall functions exactly.

The original baseline and native-mesh models use different POD bases and state definitions. The common-grid comparison holds the evaluation points fixed, but differences may reflect the representation as well as training.
