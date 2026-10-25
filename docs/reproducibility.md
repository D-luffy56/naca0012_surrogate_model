# Baseline reproduction and data inventory

The baseline is the 40 training / 7 validation OpenFOAM NACA 0012 study on a
common 192 × 128 x–z grid. The model state contains `u`, `w`, and kinematic
pressure `p`; the four retained POD modes capture 99.9533% of scaled training
energy. The MLP maps `(Re, AoA_deg)` to POD coefficients. A separate MLP maps
the same inputs to CFD-derived CL/CD.

The field relative L2 summary was corrected when publishing these artifacts.
The former implementation indexed case `i` for field `i`; it therefore did
not report field errors. The current script indexes every case for each field.
Validation aggregate relative L2 is 0.84% for u, 1.62% for w, and 3.72% for p.

## Available in a Git LFS clone

| Material | Location | Use |
| --- | --- | --- |
| Train/validation case IDs and conditions | `data/manifests/` | Fixed split and provenance |
| 40 training and 7 validation common-grid arrays | `data/processed/baseline-v1/` | Retrain and validate without VTK |
| Baseline POD and both MLPs | `models/baseline-v1/` | Immediate field and CL/CD inference |
| File sizes and SHA256 hashes | `data/manifests/artifacts.json` | Artifact integrity check |
| T01 OpenFOAM dictionaries and geometry | `cfd/reference_case_T01/` | Representative solver configuration |
| Case metadata and evidence hashes | `cfd/evidence/T01` through `T40`, `V01` through `V07` | Audit the 47 recorded cases |
| Numeric and plotted results | `results/` | Compare reproduced outputs |

Git LFS is required. `git lfs pull` fetches the binary artifacts after cloning.
Run `python src/check_baseline.py` to verify the six binary hashes, case order,
and array shapes; it also writes per-case field relative L2 values.

The `requirements-verified.txt` versions with Python 3.14.3 reproduced the
committed `results/summary.json` metrics. The original training environment
was not archived, so these versions are a verified rerun environment rather
than a historical record.

## Reproduce the machine learning stage

Use the command in the README with `--snapshots-dir` to regenerate the POD
basis, field MLP, and validation predictions. The script verifies snapshot IDs
and array dimensions against the manifests and requested grid. The committed
arrays retain all 40 training and 7 validation fields after interpolation and
airfoil masking. The split must remain unchanged: fit field scaling, mean,
POD modes, input scaling, and output scaling on training cases only.

`src/predict.py` loads the published basis and both MLPs. Its field result has
shape `(3, 128, 192)` with `(u, w, p)` ordering and NaN values inside the
airfoil. It rejects points outside the training bounds Re 2M–6M and AoA
-4°–14°. The model is validated only for NACA 0012 in this domain.

`python src/retrain_aero.py` refits the companion MLP from the committed
training CFD coefficient table. On the verified environment it returned
CL MSE 5.47×10⁻⁵ and CD MSE 4.36×10⁻⁸, close to the published 5.45×10⁻⁵
and 4.36×10⁻⁸. The serialized published model is the reference for exact
inference; the tabular rerun can vary slightly with library versions or
rounding in exported CSV values.

## CFD material needed for a full rerun

The original raw case tree is about 1 GB and is not hosted in this repository.
It needs a stable shared archive with a version, download instructions, size,
and SHA256 checksum before a new machine can regenerate the CFD results. The
path recorded in the manifests (`/mnt/d/CFD/NACA0012/cases/...`) belongs to
the original machine. It is provenance, not a portable input path.

Place the archive under `cases/training/T01...T40` and
`cases/validation/V01...V07`, then create local copies of the manifests with
`case_path` changed as shown in the README. The existing scripts need:

- `VTK/<id>_<latest_time>/internal.vtu` with `U` and `p` for every case;
- `VTK/<id>_<latest_time>/boundary/suction.vtp` and `pressure.vtp` for Cp;
- `postProcessing/forceCoeffs/0/coefficient.dat` for CL/CD;
- `log.simpleFoam` for the recorded solve time;
- mesh, geometry, `0/`, `constant/`, `system/`, and solver provenance for a CFD rerun.

The published CL/CD metrics use the last 20 force-history samples. Only 10 of
40 training cases and none of the seven validation cases are marked
`force_stable=True`; these metrics require that qualification. Cp errors also
depend on sampling reconstructed pressure at the CFD surface, so the boundary
VTK exports are needed to regenerate them.

## CFD and model checks

1. Confirm all 47 case IDs, Re/AoA values, latest times, and hashes against
   the committed manifests and `cfd/evidence/`.
2. Confirm the same mesh coordinate ordering before reusing interpolation
   indices across cases. `src/pod_naca0012.py` derives those indices from the
   first training case.
3. Confirm OpenFOAM version, kOmegaSST selection, boundary conditions, wall
   treatment, transport properties, discretization, and solver tolerances for
   every run. The T01 dictionaries provide a reference, not proof that every
   per-case file is identical.
4. Confirm grid bounds, interpolation, analytic NACA 0012 mask, pressure
   convention, field order, and per-field scaling before comparing POD modes.
5. Compare aggregate and per-case field errors, Cp RMSE, CL/CD errors, and
   force-stability flags. Keep POD projection error separate from the total
   MLP reconstruction error.
6. Interpret the 15,148× speedup as reported: median completed CFD solve time
   versus an already-loaded field inference. It excludes I/O, model loading,
   mesh generation, and plotting.

`src/download_and_catalog.py` is an optional AirfRANS catalogue utility from
the earlier exploration. It does not generate these OpenFOAM cases.
