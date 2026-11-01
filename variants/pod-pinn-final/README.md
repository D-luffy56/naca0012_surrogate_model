# NACA0012 POD–PINN: TA run instructions

## Package contents

- `POD_PINN_Combined_Workflow.ipynb` — complete sequential workflow.
- `POD_PINN_Clean/` — transparent Python scripts, trained checkpoints, metrics,
  histories, plots, and the original results notebook.
- `NACA0012_PINN_Study/data/native/` — 40 training and 8 held-out validation CFD
  arrays plus common geometry and manifest.
- `COMBINED_ENGINEERING_REPORT.md` — engineering interpretation of fixed and
  self-adaptive approaches.

No previous full-field neural-network model is used. Both approaches predict 25 POD
coefficients and reconstruct the fields through a native-mesh POD basis.

## Recommended environment

- Python 3.11 or 3.12
- Approximately 4 GB free RAM
- Approximately 1 GB free disk space
- CPU execution is sufficient; no GPU is required

From the extracted package directory:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
python -m pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2 nbformat jupyter
jupyter notebook POD_PINN_Combined_Workflow.ipynb
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
python -m pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2 nbformat jupyter
jupyter notebook POD_PINN_Combined_Workflow.ipynb
```

Open the notebook and use **Run All**. The cell `RETRAIN_MODELS` controls whether
the included checkpoints are reused (`False`, fast review) or all four models are
trained again (`True`, full reproduction). The notebook is delivered with
`RETRAIN_MODELS = True`.

Expected CPU runtime after dependencies are installed is roughly 3–10 minutes,
depending on the processor. L-BFGS output is verbose but normal.

## Workflow order

1. Verify the manifest, common mesh, fields, and 40/8 split.
2. Build the cell-volume-weighted rank-25 POD using training cases only.
3. Validate Delaunay interpolation and manufactured finite-difference derivatives.
4. Build the wall-safe structured-grid physics operator.
5. Train fixed-weight 32×32, 64×64, and 128×128 POD–PINNs.
6. Train the self-adaptive 64×64 POD–PINN.
7. Evaluate all models only on the eight held-out validation cases.
8. Recreate comparison, convergence, adaptive-weight, and field-contour figures.
9. Generate the extended aerodynamic, case-wise, POD, wake, and residual analysis.

## Principal output locations

- Fixed checkpoints: `POD_PINN_Clean/artifacts/models/pod_pinn_2x*_rank25.pt`
- Adaptive checkpoint:
  `POD_PINN_Clean/artifacts/models/pod_pinn_2x64_rank25_self_adaptive.pt`
- Fixed results: `POD_PINN_Clean/artifacts/results/`
- Adaptive results: `POD_PINN_Clean/artifacts/self_adaptive_results/`
- Extended analysis: `POD_PINN_Clean/artifacts/extended_analysis/`
- Operator validation:
  `POD_PINN_Clean/artifacts/operator/operator_validation.json`

## Reproducibility notes

- Random seeds, collocation indices, POD rank, architectures, L2 coefficient, and
  L-BFGS iteration count are fixed in the scripts.
- The eight validation cases never enter POD fitting or neural-network training.
- The Delaunay grid is used for physics derivatives and plots; native CFD cell values
  remain the data/POD truth.
- The 0.02-chord wall-adjacent band is excluded from the uniform-grid PDE residual.
- Pressure-force metrics exclude viscous drag.
- Only 10 of 48 source CFD cases pass the supplied conservative force-tail stability
  flag, so force claims should be presented cautiously.

