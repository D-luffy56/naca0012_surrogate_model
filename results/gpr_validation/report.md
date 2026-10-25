# POD-GPR validation for the AE646 digital twin

## Model and protocol

The model maps Reynolds number and angle of attack to 16 POD coefficients using a Matérn 5/2 Gaussian process. The coefficients reconstruct six native-mesh fields: u, w, kinematic pressure, k, omega, and eddy viscosity. It was fitted on 40 steady incompressible k-omega SST RANS CFD cases. The POD basis and scalers use only those 40 cases.

The original seven validation cases and seven later fresh CFD cases are reported separately. Relative L2 is the norm of prediction minus CFD divided by the norm of CFD, pooled over the named cases and mesh points. These are field errors, not classification accuracy. Pressure is OpenFOAM kinematic pressure.

## Aggregate field error

| Case group and model | u | w | pressure |
| --- | ---: | ---: | ---: |
| Original validation (7), POD-GPR | 0.622% | 2.285% | 3.013% |
| Fresh CFD (7), POD-GPR | 0.751% | 1.786% | 2.115% |
| Fresh CFD (7), data-only POD-MLP 64-64 | 1.094% | 1.355% | 2.602% |

## GPR error for each fresh CFD case

| Case | Re | AoA | u | w | pressure |
| --- | ---: | ---: | ---: | ---: | ---: |
| N01 | 2200000 | -2° | 1.106% | 3.446% | 8.737% |
| N02 | 2800000 | 9° | 0.159% | 0.463% | 0.379% |
| N03 | 3300000 | 5° | 0.150% | 0.674% | 1.020% |
| N04 | 3900000 | 13° | 1.221% | 2.490% | 2.134% |
| N05 | 4400000 | 1° | 1.027% | 4.508% | 5.698% |
| N06 | 5200000 | 7° | 0.075% | 0.112% | 0.167% |
| N07 | 5800000 | 11° | 0.388% | 0.932% | 0.856% |

## GPR error for each original validation case

| Case | Re | AoA | u | w | pressure |
| --- | ---: | ---: | ---: | ---: | ---: |
| V01 | 2500000 | 2° | 1.133% | 7.703% | 11.065% |
| V02 | 2500000 | 7° | 0.076% | 0.118% | 0.159% |
| V03 | 3500000 | 2° | 1.094% | 6.819% | 9.591% |
| V04 | 3500000 | 7° | 0.072% | 0.078% | 0.160% |
| V05 | 4500000 | 5° | 0.145% | 0.642% | 0.961% |
| V06 | 4500000 | 11° | 0.412% | 1.091% | 1.009% |
| V07 | 5500000 | 5° | 0.163% | 0.887% | 1.397% |

## Interpretation

On the seven fresh CFD cases pooled together, GPR had lower u and pressure error; the data-only MLP had lower w error. Individual cases can have different winners. The GPR has no physics loss; this comparison measures prediction against CFD, not equation residuals.

The 40-case five-fold cross-validation found the MLP better overall than GPR when each fold trained on 32 cases. The full-40 fresh-test ranking therefore should not be presented as a universal GPR advantage.

## Limits

- Fresh CFD cases met solver residual control, but all failed the stricter last-50-force-sample stability criterion.
- The fresh comparison's data-only MLP was refitted and did not exactly reproduce the older saved baseline predictions.
- The seven fresh cases have already been inspected and must not be reused as an untouched selection set.
- GPR is data-only; it has no SST physics loss or wall condition enforcement.

## Presentation assets

- `fresh_case_field_errors.png`: per-case u, w, and pressure error comparison.
- `original_validation_by_case.csv`: GPR errors for seven original validation cases, including wall and wake regions.
- `fresh_comparison_by_case.csv`: GPR and data-only MLP errors for seven fresh CFD cases and three regions.
- `summary.json`: machine-readable configuration and aggregate results.
