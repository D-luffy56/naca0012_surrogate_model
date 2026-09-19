# NACA 0012 POD–MLP Aerodynamic Surrogate

A reproducible AE646 SciML project that turns a family of high-fidelity CFD solutions into a fast reduced-order model. The workflow learns the relationship

`(Reynolds number, angle of attack) → POD coefficients → u, w, p flow field`

for a fixed NACA 0012 airfoil.

## Why this project

CFD is reliable but each new operating condition requires another solver run. Here, CFD is used once to generate a trusted snapshot library. Proper Orthogonal Decomposition (POD) compresses those snapshots into a small set of dominant flow patterns, and a neural network predicts their coefficients at a new operating condition. The result is intended for rapid aerodynamic exploration, not as a replacement for final high-fidelity verification.

## Headline results

| Item | Result |
| --- | --- |
| Training CFD cases | 40 |
| Independent validation cases | 7 |
| Flow solver | OpenFOAM `simpleFoam`: steady, incompressible turbulent RANS; SIMPLE; k–ω SST |
| Reynolds-number range | 2.0M–6.0M |
| Training angle-of-attack range | −4° to 14° |
| POD shared grid | 192 × 128 points; 24,468 fluid locations |
| State per snapshot | `u`, `w`, `p` (73,404 scalar values) |
| POD modes kept | 4 |
| Scaled POD energy retained | 99.9533% |
| Validation relative L2 error | u: 12.22%, w: 4.57%, p: 11.38% |
| Mean validation Cp RMSE | 0.02099 |
| Median CFD solve time | 93.9 s |
| In-memory POD–MLP inference | 6.20 ms |
| Solver-to-inference speed-up | ~15,148× |

The speed-up compares a completed CFD solve with already-loaded model inference; it does not include disk I/O, mesh generation, loading a model, or plotting.

## Method

1. **CFD snapshots** — Solve 2D, incompressible, turbulent flow around NACA 0012 on a common mesh. The thin 3D extrusion used by OpenFOAM lets the 2D solver operate, while the physical flow plane is x–z.
2. **Common-grid sampling** — Export CFD fields to VTK, collapse duplicate thin-extrusion points, then use Delaunay triangulation and barycentric interpolation to sample every case on the same 192 × 128 x–z grid. Airfoil-interior points are masked out.
3. **POD/SVD** — Per-field scale and mean-centre all snapshot states, then use singular value decomposition to find dominant modes.
4. **MLP surrogate** — Train an MLP with architecture 2 → 64 → 64 → 4 to map `(Re, AoA)` to the four POD coefficients.
5. **Validation** — Reconstruct withheld flow fields, compare relative L2 errors and Cp, and predict CL/CD with a separate companion MLP.

At 12° and 14°, the steady RANS results may be near stall and should be interpreted as steady approximations; they do not establish time-accurate unsteady separation.

## Repository layout

```text
src/                         Reproducible workflow scripts
data/manifests/              Training and validation case catalogues
results/                     Compact figures, tables, and result summaries
  field_validation/          Reconstructed field comparisons (V01–V07)
  aerodynamic_validation/    Cp, CL/CD, and runtime evaluations
```

The raw OpenFOAM case directories and VTK exports are deliberately excluded: they are approximately 1 GB and contain generated CFD data. The manifests document every required case and the scripts expect the following local structure:

```text
cases/
  training/T01 ... T40/
  validation/V01 ... V07/
```

Each case must contain `VTK/<case_id>_<latest_time>/internal.vtu` exported with the fields `U`, `p`, `k`, `omega`, and `nut`.

## Installation

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the pipeline

The copied manifests contain the original CFD-machine paths. Create local versions that point to the `cases` directory before running:

```powershell
$root = (Get-Location).Path
$train = Import-Csv .\data\manifests\training_cases.csv
$valid = Import-Csv .\data\manifests\validation_cases.csv
foreach ($row in $train) { $row.case_path = Join-Path $root "cases\training\$($row.id)" }
foreach ($row in $valid) { $row.case_path = Join-Path $root "cases\validation\$($row.id)" }
$train | Export-Csv .\data\manifests\training_cases_local.csv -NoTypeInformation
$valid | Export-Csv .\data\manifests\validation_cases_local.csv -NoTypeInformation
```

Build POD and the primary field surrogate:

```powershell
python .\src\pod_naca0012.py `
  --training-manifest .\data\manifests\training_cases_local.csv `
  --validation-manifest .\data\manifests\validation_cases_local.csv `
  --output-dir .\outputs\pod_model `
  --nx 192 --nz 128 --energy 0.999
```

Generate field-validation figures and inspect force stability:

```powershell
python .\src\report_pod_validation.py `
  --model-dir .\outputs\pod_model `
  --validation-manifest .\data\manifests\validation_cases_local.csv `
  --output-dir .\outputs\reports
```

Evaluate Cp, the companion CL/CD surrogate, and runtime:

```powershell
python .\src\evaluate_aerodynamics.py `
  --model-dir .\outputs\pod_model `
  --validation-manifest .\data\manifests\validation_cases_local.csv `
  --output-dir .\outputs\engineering_evaluation
```

## Important modelling note

The POD field state is only `u`, `w`, and `p`. Wall shear stress is not reconstructed, so CL and CD are **not** computed by integrating the POD field. Instead, a companion 2 → 64 → 64 → 2 MLP is trained directly on the CFD force-coefficient data. Its validation MSE was 5.45e-05 for CL and 4.36e-08 for CD.

## Case coverage

Training uses Re = 2M, 3M, 4M, 5M, 6M and AoA = −4°, 0°, 4°, 6°, 8°, 10°, 12°, 14°. The seven validation cases lie between those samples: 2.5M/2°, 2.5M/7°, 3.5M/2°, 3.5M/7°, 4.5M/5°, 4.5M/11°, and 5.5M/5°.

## Reproducing or extending the study

For a different airfoil, use a common CFD mesh and modify the airfoil polygon in `src/pod_naca0012.py`. For better off-design accuracy, add CFD cases near flow-regime changes and consider an unsteady solver for post-stall conditions.
