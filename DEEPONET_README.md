# DeepONet extension for the NACA0012 POD project

This add-on replaces the coefficient predictor in the **native six-field SST POD pipeline** with a branch/trunk DeepONet. It provides trained data-only and physics-informed checkpoints, comparisons with MLPs, five-fold cross-validation, numerical tests and inference code. Existing project files are preserved. See `DEEPONET_RESULTS.md` for measured results.

## Install and run on Windows

Extract this add-on ZIP into `D:\Aniket\IITK PhD\SCIML`, at the same level as the analysis archive's `src` and `outputs` directories. Merge the directories. The existing `outputs/native_sst/native_sst_snapshots.npz`, baseline folders and `outputs/fresh_test_cfd_v2` must be present. This is an add-on, not another copy of the CFD data.

```powershell
cd 'D:\Aniket\IITK PhD\SCIML'
python -m pip install -r requirements_deeponet.txt
# Use the supplied trained model immediately:
python src/predict_deeponet.py --re 3500000 --aoa 7
# Evaluate supplied checkpoints and reproduce the report data:
python src/evaluate_deeponet.py
# Retrain, cross-validate, evaluate and run tests:
python src/run_deeponet_experiment.py
```

`predict_deeponet.py` writes 16 coefficients to `outputs/deeponet_prediction/pod_coefficients.csv` and dimensional fields to `fields.npz`. Add `--data-only` to use the pre-physics checkpoint. Neither prediction command needs CFD coefficients. Rerunning the experiment replaces only its DeepONet output folders. Training uses the CPU and can take several minutes or longer on your machine. Package versions used for the supplied run are in `outputs/deeponet_sst/environment.json`; exact numerical reproducibility across platforms is not guaranteed.

## Inputs, labels and architecture

The **branch input** is the two-dimensional standardized vector `[Re, AoA_deg]`. The **trunk input** is a one-hot vector identifying one of the 16 retained POD modes. Both networks have two 32-neuron tanh hidden layers and a 32-dimensional output. Their dot product predicts one standardized coefficient:

\[
\widehat a'_k(\mu)=\frac{1}{\sqrt{32}}\sum_{j=1}^{32}B_j(\mu)T_j(e_k)+b_k,
\qquad \mu=(Re,\alpha).
\]

All mode queries are evaluated together, producing a batch-by-16 coefficient matrix. A one-hot query treats modes as categories: the arbitrary numbering of POD modes is not assumed to be a smooth physical coordinate. Actual POD coefficients are the supervised **targets**, not an input at prediction time. Feeding the desired coefficient into the predictor would leak the answer.

This is a **parametric, coefficient-query DeepONet**. Its trunk queries POD modes, not spatial coordinates. The field decoder remains the fixed POD basis. A classical spatial DeepONet instead queries `(x,z)`; a conventional POD-DeepONet uses fixed POD modes as its trunk. The original POD-MLP already resembles that fixed-trunk construction. The new learned trunk is a factorized multi-output predictor: because mode queries are fixed, its evaluated trunk matrix can be folded into a linear output map. It does not automatically confer greater expressive power than an MLP or generalization to unseen modes, geometries or arbitrary boundary functions. Both branch and trunk here themselves use MLP layers.

## Data and POD reconstruction

The native snapshot archive contains 40 training cases and seven previously inspected validation cases, with 38,610 mesh points and fields `[u,w,p,k,omega,nut]`. The code follows the original nondimensionalization for unit chord:

\[
q=(u/U_\infty,w/U_\infty,p/U_\infty^2,k/U_\infty^2,
\omega/U_\infty,\nu_t/U_\infty),\qquad U_\infty=Re\,(1.56\times10^{-5}).
\]

OpenFOAM `p` is kinematic pressure. Fit field scales `s`, state mean and POD modes on training cases only. With each case flattened across six fields and mesh points:

\[
x_i=\operatorname{vec}(q_i/s),\quad a_i=(x_i-\bar x)\Phi^T,
\quad \widehat q_i=s\odot\operatorname{reshape}(\bar x+\widehat a_i\Phi).
\]

Here `Phi` is a 16-by-231660 matrix. A 40-by-40 Gram eigendecomposition computes the POD basis. Fit separate training-only scalers for inputs and coefficients. Predictions are inverse-scaled before reconstruction. The supplied actual coefficient targets are obtained by projecting the CFD snapshots into this basis; they are not coefficients from the incompatible original four-mode, three-field basis.

## Training losses and physics

Stage 1 fits standardized coefficients for 1,500 full-batch Adam steps at learning rate 0.001, followed by at most 500 L-BFGS iterations with strong-Wolfe line search. No validation score controls stopping. Its objective is coefficient MSE plus the L2 penalty below. The checkpoint is `deeponet_data.pt`.

Stage 2 starts from that checkpoint and retains the existing project's approximate steady SST objective:

\[
L=L_a+L_q+0.1L_{wall}+10^{-4}L_{physics}+L_{L2},
\]

- `L_a`: mean squared error over training cases and standardized coefficients.
- `L_q`: mean squared error of reconstructed nondimensional fields divided by training field scales, across all six fields and mesh points.
- `L_wall`: similarly scaled error at 400 wall nodes against the separately exported CFD wall values. These are supervised wall values; this is not an exact wall-function implementation.
- `L_physics`: average of six residual MSEs on valid graph stencils, with each residual divided by its training-CFD residual RMS (floor `1e-5`).
- `L_L2 = alpha/(2*N) * sum(W**2)`, `alpha=1e-4`, `N=40`, over linear-layer weights in both networks. Biases are excluded. The MSE convention and optimization path differ from sklearn even when `alpha` has the same value.

Stage 2 uses 20 full-training-set L-BFGS updates, learning rate 0.1, maximum one internal iteration per update, history 20, and no line search, matching the existing SST fine-tuning approach. Batches of four accumulate the same full-set gradient. Physics and wall losses can be controlled with `--physics-weight` and `--wall-weight`; the saved data-only checkpoint remains the clean coefficient-only comparison.

The six inherited residuals are continuity, x momentum, z momentum, turbulent kinetic energy transport, specific dissipation transport, and eddy-viscosity closure. Define `D_x,D_z` as native-mesh graph least-squares derivatives, `d=D_x u+D_z w`, `nu=1/Re`, and `nu_e=nu+nu_t`:

\[
r_c=d,\qquad
r_x=uD_xu+wD_zu+D_xp-D_x\tau_{xx}-D_z\tau_{xz},
\]
\[
r_z=uD_xw+wD_zw+D_zp-D_x\tau_{xz}-D_z\tau_{zz},
\]
\[
\tau_{xx}=\nu_e(2D_xu-2d/3),\quad
\tau_{zz}=\nu_e(2D_zw-2d/3),\quad
\tau_{xz}=\nu_e(D_zu+D_xw).
\]

Using the original SST blending functions `F1,F2`, production limiters, and blended constants:

\[
r_k=uD_xk+wD_zk-\nabla\cdot[(\nu+\alpha_k\nu_t)\nabla k]
-P_k+\tfrac23dk+\beta^*\omega k,
\]
\[
r_\omega=uD_x\omega+wD_z\omega-\nabla\cdot[(\nu+\alpha_\omega\nu_t)\nabla\omega]
-\gamma\widetilde G+\tfrac23\gamma d\omega+\beta\omega^2
-\frac{2(1-F_1)\alpha_{\omega2}}{\omega}\nabla k\cdot\nabla\omega,
\]
\[
r_{\nu_t}=\nu_t-\frac{a_1k}{\max(a_1\omega,b_1F_2\sqrt{S^2})}.
\]

`sst_deeponet_physics.py` contains the full preserved equations, constants, blending functions and limiters. In particular, `beta*=0.09`, `a1=0.31`, `b1=1`, production limiter `c1=10`; `(alpha_k1,alpha_k2)=(0.85,1)`, `(alpha_omega1,alpha_omega2)=(0.5,0.856)`, `(gamma1,gamma2)=(5/9,0.44)`, `(beta1,beta2)=(0.075,0.0828)`. The residual calculation clamps `k`, `omega` and `nut` internally; decoded fields are not positivity constrained, so their negative fractions are reported.

Physics gradients flow through `residuals -> reconstructed fields -> predicted POD coefficients -> branch and trunk weights`. Spatial derivatives are graph derivatives of the decoded field, not derivatives with respect to Reynolds number or angle of attack. The copied residual equations are unchanged from the project. They approximate OpenFOAM and do not reproduce its finite-volume operators or nonlinear omega wall function. Even CFD snapshots have nonzero residuals under these graph operators. Each term is referenced to training CFD, but the `1e-5` normalization floor means the training-CFD aggregate need not equal one. It is about 0.8494 in this run. A score near that reference is not proof of exact PDE satisfaction.

## Evaluation and limitations

`evaluate_deeponet.py` evaluates training cases, seven validation cases and seven additional archived CFD cases. It reports:

- Coefficient RMSE, MAE, standardized RMSE, relative L2 and uniform-average R2; each actual/predicted coefficient in CSV.
- Six-field global, near-wall and wake relative L2, nondimensional RMSE and MAE; per-case errors.
- CFD projection through the retained POD basis, separating representation loss from regression loss. Projection minimizes the combined scaled-state Euclidean error, not necessarily every individual field's relative error.
- Separate wall-field normalized RMSE, validation surface Cp RMSE/curves, negative turbulence-field fractions.
- All six physics residual RMS values and their normalized aggregate, including CFD references and POD projections.
- Original three-field POD-MLP comparison on its original Cartesian grid, using its dimensional CFD targets and the project's barycentric interpolation convention.
- Warm resident CPU inference timing, coefficient parity and field comparison plots.

Relative L2 is `norm(pred-truth)/norm(truth)`, reported as a percentage where indicated. This regression task has no classification accuracy; `100-error%` is not a meaningful general accuracy measure. Global field errors use equal point weighting, as in the original project, not mesh cell-volume weighting. Coefficient R2 can be sensitive to low-variance modes.

Five-fold CV in `cross_validate_deeponet.py` refits POD and both scalers inside each training fold. It compares fixed data-only DeepONet and tanh MLP settings; it is not a hyperparameter search and does not cross-validate physics fine-tuning. Seeds, capacities and optimizer budgets matter, so this is evidence for these runs, not a universal claim about architectures. The saved original MLP-SST checkpoints are also evaluated; the paired data-only MLP is explicitly refit because its original estimator was not retained.

The seven additional CFD cases were already examined in the source project. They are not newly untouched test cases. Their supplied quality records say residual-control convergence was achieved, but the stricter last-50 force-stability criterion failed. No evaluation data were used to tune this DeepONet run. The original 40+7 raw CFD archive failed to transfer; training uses the supplied processed snapshots, and an additional case supplies only a coordinate-checked copy of the common mesh topology. No new CFD solves were performed.

Total CL/CD comparison is not claimed: the previous project used a separate force-coefficient MLP, and total drag requires appropriate wall-shear treatment. Cp comparison is from the reconstructed pressure field. The four-mode original POD-MLP and this 16-mode six-field model differ in representation; their common-grid comparison is not an isolated architecture ablation.

## Files

- `src/coefficient_deeponet.py`: architecture, weight penalty and coefficient pretraining.
- `src/train_deeponet_sst.py`: native POD pipeline, physics fine-tuning and checkpoint inference.
- `src/sst_deeponet_physics.py`: inherited SST residual equations.
- `src/evaluate_deeponet.py`: metrics, MLP comparisons, CSVs and figures.
- `src/cross_validate_deeponet.py`: leakage-controlled five-fold comparison.
- `src/predict_deeponet.py`: command-line inference.
- `src/run_deeponet_experiment.py`: reproducible end-to-end command.
- `tests/test_deeponet.py`: shape/gradient, checkpoint, known-target fit and physics-backpropagation tests.
- `outputs/deeponet_sst/`: basis, both trained checkpoints, history, predictions, evaluation and environment.
- `outputs/deeponet_cv/`: fold errors and aggregate results.

References for architectural context: [Lu et al., DeepONet](https://arxiv.org/abs/1910.03193), [physics-informed DeepONets](https://pmc.ncbi.nlm.nih.gov/articles/PMC8480920/), and [POD-DeepONet extensions](https://www.osti.gov/servlets/purl/1976975). The numerical results here come from the supplied project data and this implementation.
