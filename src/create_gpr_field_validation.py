"""Create seven CFD/POD-GPR field comparisons like the existing POD-MLP plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from evaluate_fresh_openfoam_tests import basis_data, scale_inputs_like_sklearn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/gpr_field_validation"
SNAPSHOTS = ROOT / "outputs/native_sst/native_sst_snapshots.npz"
FIELDS = ("u", "w", "p")


def relative_l2(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - truth) /
                 max(np.linalg.norm(truth), 1e-12))


def naca0012_solid(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Mask the airfoil for visualization only; metrics use native CFD points."""
    xc = np.clip(x, 0, 1)
    thickness = 0.6 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                       + 0.2843 * xc**3 - 0.1015 * xc**4)
    return (x >= 0) & (x <= 1) & (np.abs(z) <= thickness)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(SNAPSHOTS) as saved:
        raw = saved["fields"].copy()
        ids = saved["ids"].copy()
        reynolds = saved["reynolds"].copy()
        aoa = saved["aoa_deg"].copy()
        uinf = saved["uinf_mps"].copy()
        points = saved["points_xz"].copy()
        train_count = int(saved["train_count"])
    if train_count != 40 or len(raw) != 47:
        raise ValueError("Expected 40 training and seven validation cases")
    basis = basis_data(ROOT / "outputs/sst_pod_model")
    units = np.stack((uinf, uinf, uinf**2, uinf**2, uinf, uinf), axis=1)
    nondimensional = raw / units[:, :, None]
    scaled = (nondimensional[:train_count] / basis["scale"][None, :, None]).reshape(train_count, -1)
    coefficients = (scaled - basis["mean"]) @ basis["modes"].T
    standardized_y = coefficients.astype(np.float32, copy=True)
    standardized_y -= basis["coefficient_mean"]
    standardized_y /= basis["coefficient_scale"]
    conditions = np.column_stack((reynolds, aoa))
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
               Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)),
        alpha=1e-8, normalize_y=False, n_restarts_optimizer=1, random_state=42,
    ).fit(scale_inputs_like_sklearn(conditions[:train_count], basis), standardized_y)
    predicted_standardized = model.predict(scale_inputs_like_sklearn(conditions[train_count:], basis))
    predicted_coef = (predicted_standardized * basis["coefficient_scale"]
                      + basis["coefficient_mean"])
    predicted_nondimensional = ((basis["mean"] + predicted_coef @ basis["modes"])
                                .reshape(7, 6, -1) * basis["scale"][None, :, None])
    actual = raw[train_count:]
    predicted = predicted_nondimensional * units[train_count:, :, None]

    # One interpolation geometry for all panels; CFD comparison and L2 stay on the native mesh.
    x = np.linspace(-2, 4, 192)
    z = np.linspace(-1.5, 1.5, 128)
    xx, zz = np.meshgrid(x, z)
    solid = naca0012_solid(xx, zz)
    triangulation = Delaunay(points)
    rows = []
    for case, case_id in enumerate(ids[train_count:]):
        case_name = str(case_id)
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        fig.suptitle(f"{case_name}: CFD vs POD-GPR reconstruction", fontsize=15)
        for field_index, field in enumerate(FIELDS):
            reference = actual[case, field_index]
            estimate = predicted[case, field_index]
            reference_grid = LinearNDInterpolator(triangulation, reference)(xx, zz)
            estimate_grid = LinearNDInterpolator(triangulation, estimate)(xx, zz)
            reference_grid[solid] = np.nan
            estimate_grid[solid] = np.nan
            difference_grid = estimate_grid - reference_grid
            bound = max(np.nanmax(np.abs(reference_grid)), np.nanmax(np.abs(estimate_grid)))
            difference_bound = max(np.nanmax(np.abs(difference_grid)), 1e-12)
            for axis, values, title, cmap, vmin, vmax in (
                (axes[field_index, 0], reference_grid, "CFD", "viridis", -bound, bound),
                (axes[field_index, 1], estimate_grid, "POD-GPR", "viridis", -bound, bound),
                (axes[field_index, 2], difference_grid, "GPR - CFD", "coolwarm", -difference_bound, difference_bound),
            ):
                image = axis.pcolormesh(x, z, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_aspect("equal")
                axis.set_title(f"{field}: {title}")
                axis.set_xlabel("x/c")
                axis.set_ylabel("z/c")
                fig.colorbar(image, ax=axis, shrink=0.82)
            rows.append({"id": case_name, "Re": float(reynolds[train_count+case]),
                         "AoA_deg": float(aoa[train_count+case]), "field": field,
                         "relative_l2_native": relative_l2(reference, estimate)})
        fig.savefig(OUTPUT / f"validation_{case_name}.png", dpi=180)
        plt.close(fig)
        print(f"Created {case_name}", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "validation_field_errors.csv", index=False)
    # Pool nondimensional case fields, matching the other native-mesh comparisons.
    aggregate = {field: relative_l2(nondimensional[train_count:, i], predicted_nondimensional[:, i])
                 for i, field in enumerate(FIELDS)}
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "training_cases": train_count, "validation_cases": 7,
        "pod_modes": len(basis["modes"]), "kernel": str(model.kernel_),
        "native_points": len(points), "aggregate_relative_l2_native": aggregate,
        "aggregate_note": "Pooled after each case is nondimensionalized by its own Uinf; per-case relative L2 is unchanged by this scaling.",
        "plot_note": "Plots interpolate the native CFD and GPR fields to a regular display grid and mask a NACA 0012 solid. All L2 errors are calculated on the native mesh, before display interpolation.",
        "pressure_note": "OpenFOAM kinematic pressure in m^2/s^2, not Pa.",
    }, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
