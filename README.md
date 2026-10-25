# NACA 0012 POD–MLP Aerodynamic Surrogate

The project's six model families, variant branches, accuracy tables, and
comparison plot are collected in [MODEL_VARIANTS.md](MODEL_VARIANTS.md).

A reproducible AE646 scientific machine-learning project that converts a fixed-geometry family of OpenFOAM CFD solutions into a reduced-order surrogate:

```text
(Reynolds number, angle of attack) -> POD coefficients -> u, w, p flow field
```

This repository implements a data-driven POD–MLP workflow. It does **not** contain a physics-informed neural network (PINN).

## Verified repository results

The values below are recorded in the committed result files. The processed
snapshot arrays and trained baseline artifacts are also available through Git LFS.

| Item | Committed result |
| --- | --- |
| Training CFD cases | 40 |
| Independent validation cases | 7 |
| Recorded CFD setup | OpenFOAM `simpleFoam`; steady incompressible RANS; SIMPLE; k–omega SST |
| Reynolds-number range | 2.0M–6.0M |
| Training angle-of-attack range | -4° to 14° |
| Shared POD grid | 192 x 128 points; 24,468 fluid locations |
| State per snapshot | `u`, `w`, `p` (73,404 scalar values) |
| POD modes retained | 4 |
| Scaled POD energy retained | 99.9533% |
| Aggregate validation relative L2 error | u: 0.84%; w: 1.62%; p: 3.72% |
| Mean validation Cp RMSE | 0.02099 |
| Median recorded CFD solve time | 93.9 s |
| In-memory POD–MLP inference | 6.20 ms |
| Median recorded speed-up | about 15,148x |

Sources: [`results/summary.json`](results/summary.json) and [`results/aerodynamic_validation/engineering_summary.json`](results/aerodynamic_validation/engineering_summary.json).

The earlier field-error summary indexed the first three cases while labeling
them as `u`, `w`, and `p`. This version computes each field's norm across all
seven validation cases. `results/baseline-v1/validation_field_errors.csv`
provides the corresponding per-case values.

The timing comparison is a completed CFD solve versus an already-loaded Python model performing field inference. It excludes mesh generation, disk I/O, model loading, and plotting.

## Governing equations and learned model

The recorded CFD setup is steady, incompressible Reynolds-averaged Navier–Stokes (RANS). Its mean-flow equations are

```text
div(U) = 0

rho (U . grad) U = -grad(p) + div[2 (mu + mu_t) S]
```

where `S` is the mean strain-rate tensor and `mu_t` is the turbulent eddy viscosity. In an SST k–omega closure, `mu_t` is obtained from two additional transport equations for turbulent kinetic energy `k` and specific dissipation rate `omega`, with SST blending and a shear-stress limiter.

The original `0/`, `constant/`, and `system/` files for training case T01 are
committed under `cfd/reference_case_T01/`, with case metadata and evidence for
all 47 runs under `cfd/evidence/`. T01 records OpenFOAM v2312 and `kOmegaSST`.
These files provide a concrete reference configuration, while the full mesh and
per-case solver outputs are still needed to audit or rerun every CFD case.

For the reduced-order model, each sampled snapshot is scaled by field and mean-centred. With the snapshot state `q(mu)`, mean `q_bar`, and POD modes `phi_i`,

```text
q(mu) ~= q_bar + sum_i a_i(mu) phi_i
```

The modes come from the singular value decomposition of the centred snapshot matrix. The smallest rank meeting the requested cumulative energy threshold is retained. Here, four modes exceed the 99.9% target.

A scikit-learn MLP with architecture `2 -> 64 -> 64 -> 4`, ReLU activations, and standardized inputs/targets learns

```text
(Re, AoA) -> (a_1, a_2, a_3, a_4)
```

using mean-squared error on standardized POD coefficients. This is supervised regression over CFD-derived coefficients; no governing-equation residual appears in the training loss.

## Data used by the workflow

The committed manifests define the 40 training and 7 validation cases:

- [`data/manifests/training_cases.csv`](data/manifests/training_cases.csv)
- [`data/manifests/validation_cases.csv`](data/manifests/validation_cases.csv)

Training uses Re = 2M, 3M, 4M, 5M, 6M and AoA = -4°, 0°, 4°, 6°, 8°, 10°, 12°, 14°. Validation uses 2.5M/2°, 2.5M/7°, 3.5M/2°, 3.5M/7°, 4.5M/5°, 4.5M/11°, and 5.5M/5°.

The POD script reads only the OpenFOAM VTK point fields `U` and `p` from

```text
cases/<split>/<case_id>/VTK/<case_id>_<latest_time>/internal.vtu
```

and forms the x–z-plane state `(U_x, U_z, p)`. The turbulence fields `k`, `omega`, and `nut` are not inputs to the current POD state or MLP.

The trained `.joblib` models, POD basis, and processed training and validation
arrays are committed through Git LFS. A clean clone with Git LFS can predict and
retrain the POD–MLP stage. Raw VTK exports, complete meshes, force histories,
and solver logs remain outside the repository; they are required for a fresh
CFD run and for regenerating Cp and force metrics from raw CFD output.

## Method

1. Filter the manifests to cases marked `converged=True` and `status=outputs_verified`.
2. Collapse duplicate points from the thin OpenFOAM extrusion and interpolate every case to a common 192 x 128 x–z grid.
3. Mask the analytic NACA 0012 interior.
4. Scale `u`, `w`, and `p` separately, mean-centre the states, and compute the POD basis with SVD.
5. Train the MLP from `(Re, AoA)` to retained POD coefficients.
6. Reconstruct withheld fields and report relative L2 errors and surface-pressure error.
7. Train a separate `2 -> 64 -> 64 -> 2` MLP for CL and CD.

CL and CD are not obtained by integrating the reconstructed `u,w,p` field. Drag needs wall-shear information that is absent from the POD state, so the companion MLP learns CFD force coefficients directly.

## Repository layout

```text
src/
  pod_naca0012.py             POD basis, field MLP, and validation predictions
  predict.py                  Load the trained baseline and predict fields and CL/CD
  check_baseline.py           Verify artifact hashes and per-case field errors
  retrain_aero.py             Refit companion CL/CD MLP from committed CFD coefficient tables
  report_pod_validation.py    Field plots and force-tail review
  evaluate_aerodynamics.py    Cp, CL/CD, and runtime evaluation
  download_and_catalog.py     Optional AirfRANS catalogue utility; not used by the committed OpenFOAM study
data/manifests/               Committed case catalogues
data/processed/baseline-v1/    Common-grid arrays (Git LFS)
models/baseline-v1/            Trained POD and MLP artifacts (Git LFS)
cfd/reference_case_T01/       Example original SST solver dictionaries and geometry
cfd/evidence/                 Metadata and file hashes for all 47 CFD cases
results/                      Committed summaries, tables, and figures
```

The AirfRANS catalogue utility belongs to an earlier/optional data-discovery path. It requires the separately installed `airfrans` package and does not recreate the committed OpenFOAM case family.

## Installation

The baseline was verified here with Python 3.14.3 and the exact versions in
`requirements-verified.txt`. The original training environment was not saved.
Install Git LFS before cloning so the model and array files are fetched.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Use and check the committed baseline

```powershell
python .\src\check_baseline.py
python .\src\predict.py --re 3500000 --aoa 5 --output .\outputs\example.npz
```

The output array has shape `(3, 128, 192)` in `(u, w, p)` order, with NaNs
inside the airfoil. The companion CL/CD MLP uses the same `(Re, AoA)` input;
those coefficients are not integrated from the predicted field. Load `.joblib`
files only from a trusted source and check `data/manifests/artifacts.json`.

Retrain the POD basis and field MLP from the committed processed arrays:

```powershell
python .\src\pod_naca0012.py `
  --training-manifest .\data\manifests\training_cases.csv `
  --validation-manifest .\data\manifests\validation_cases.csv `
  --snapshots-dir .\data\processed\baseline-v1 `
  --output-dir .\outputs\retrained --nx 192 --nz 128 --energy 0.999
```

This path reproduced the committed `results/summary.json` values in the
verified environment. See `docs/reproducibility.md` for what is still needed
to regenerate the CFD, Cp plots, and force coefficients from scratch.

The companion CL/CD MLP can also be refit from the committed coefficient
tables with `python src/retrain_aero.py`. Its validation errors are close to
the published values; the published serialized model remains the reference.

## Reproducing the workflow with the missing CFD cases

Place the original cases in this local structure:

```text
cases/
  training/T01 ... T40/
  validation/V01 ... V07/
```

The committed manifests contain paths from the original CFD machine. Create local copies without modifying the evidence manifests:

```powershell
$projectRoot = (Get-Location).Path
$train = Import-Csv .\data\manifests\training_cases.csv
$valid = Import-Csv .\data\manifests\validation_cases.csv
foreach ($row in $train) { $row.case_path = Join-Path $projectRoot "cases\training\$($row.id)" }
foreach ($row in $valid) { $row.case_path = Join-Path $projectRoot "cases\validation\$($row.id)" }
$train | Export-Csv .\data\manifests\training_cases_local.csv -NoTypeInformation
$valid | Export-Csv .\data\manifests\validation_cases_local.csv -NoTypeInformation
```

Build the POD basis and primary field surrogate:

```powershell
python .\src\pod_naca0012.py `
  --training-manifest .\data\manifests\training_cases_local.csv `
  --validation-manifest .\data\manifests\validation_cases_local.csv `
  --output-dir .\outputs\pod_model `
  --nx 192 --nz 128 --energy 0.999
```

Generate field-validation figures and the force-tail review:

```powershell
python .\src\report_pod_validation.py `
  --pod-dir .\outputs\pod_model `
  --training-manifest .\data\manifests\training_cases_local.csv
```

Evaluate Cp, the companion CL/CD MLP, and runtime:

```powershell
python .\src\evaluate_aerodynamics.py `
  --pod-dir .\outputs\pod_model `
  --training-manifest .\data\manifests\training_cases_local.csv `
  --validation-manifest .\data\manifests\validation_cases_local.csv
```

The reporting scripts write below `outputs/pod_model/reports/` and `outputs/pod_model/engineering_evaluation/`.

## Interpretation limits

Many manifest rows are marked `force_stable=False`, including all seven validation cases. Their warnings state that force coefficients were not steady. The field snapshots are recorded as converged and output-verified, but CL/CD metrics should be treated as results of the chosen tail-averaging procedure rather than fully steady force solutions.

The 12° and 14° cases may be near stall. Steady RANS solutions there are steady approximations and do not establish time-accurate separated-flow behavior.

Predictions are supported only within the sampled NACA 0012 operating domain. Extrapolation to another geometry, outside the Reynolds/AoA range, or into a different flow regime needs new CFD data and validation.

## Extending the study

For a full CFD rerun, separately archive the remaining per-case meshes, VTK
exports, force histories, and logs. For a different airfoil, generate a
consistent CFD database and update the solid mask in `src/pod_naca0012.py`.
If a PINN is desired, implement it separately with explicit PDE residuals,
boundary conditions, nondimensionalization, and validation.
