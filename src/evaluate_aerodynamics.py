"""Evaluate Cp, force coefficients, runtime, and speed-up for the POD-MLP twin.

Cp is sampled from the POD-MLP reconstructed pressure field at the CFD airfoil
surface. CL/CD are evaluated with a companion operating-condition MLP because
the first POD state contains u, w, p only and does not retain wall-shear data
needed to integrate viscous drag from the reconstructed field.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def windows_path(value: str) -> Path:
    return Path("D:/" + value.removeprefix("/mnt/d/")) if value.startswith("/mnt/d/") else Path(value)


def restore_grid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.size, np.nan, dtype=float)
    grid[mask.ravel()] = values
    return grid.reshape(mask.shape)


def force_history(case: Path) -> pd.DataFrame:
    path = case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    names = ("time", "Cd", "Cd_f", "Cd_r", "Cl", "Cl_f", "Cl_r", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs_f", "Cs_r")
    return pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=names)


def tail_coefficients(case: Path) -> tuple[float, float]:
    tail = force_history(case).tail(20)
    return float(tail.Cl.mean()), float(tail.Cd.mean())


def execution_seconds(case: Path) -> float:
    text = (case / "log.simpleFoam").read_text(encoding="utf-8", errors="ignore")
    values = re.findall(r"ExecutionTime\s*=\s*([0-9.]+)\s*s", text)
    if not values:
        raise ValueError(f"No ExecutionTime in {case / 'log.simpleFoam'}")
    return float(values[-1])


def surface_cp(case: Path, case_id: str, latest_time: int, u_inf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = case / "VTK" / f"{case_id}_{latest_time}" / "boundary"
    surfaces = []
    for label in ("suction", "pressure"):
        mesh = pv.read(base / f"{label}.vtp")
        xz = mesh.points[:, [0, 2]]
        cp = np.asarray(mesh.point_data["p"], dtype=float) / (0.5 * u_inf**2)
        surfaces.append((label, xz, cp))
    return surfaces  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.pod_dir / "engineering_evaluation"
    output.mkdir(exist_ok=True)

    train = pd.read_csv(args.training_manifest)
    valid = pd.read_csv(args.validation_manifest)
    train = train[(train.converged.astype(str).str.lower() == "true") & (train.status == "outputs_verified")].copy()
    valid = valid[(valid.converged.astype(str).str.lower() == "true") & (valid.status == "outputs_verified")].copy()
    for frame in (train, valid):
        values = [tail_coefficients(windows_path(row.case_path)) for _, row in frame.iterrows()]
        frame[["Cl_cfd", "Cd_cfd"]] = values
        frame["cfd_runtime_s"] = [execution_seconds(windows_path(row.case_path)) for _, row in frame.iterrows()]

    # A companion MLP supplies aerodynamic coefficients. It is evaluated
    # separately from the field decoder because drag requires wall shear.
    x_scaler = StandardScaler().fit(train[["Re", "AoA_deg"]])
    y_scaler = StandardScaler().fit(train[["Cl_cfd", "Cd_cfd"]])
    aero_model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs", max_iter=5000, random_state=42)
    aero_model.fit(x_scaler.transform(train[["Re", "AoA_deg"]]), y_scaler.transform(train[["Cl_cfd", "Cd_cfd"]]))
    train[["Cl_mlp", "Cd_mlp"]] = y_scaler.inverse_transform(aero_model.predict(x_scaler.transform(train[["Re", "AoA_deg"]])))
    valid[["Cl_mlp", "Cd_mlp"]] = y_scaler.inverse_transform(aero_model.predict(x_scaler.transform(valid[["Re", "AoA_deg"]])))
    joblib.dump({"model": aero_model, "x_scaler": x_scaler, "y_scaler": y_scaler}, output / "aero_coefficients_mlp.joblib")

    # End-to-end POD-MLP inference: scale inputs, predict coefficients, decode field.
    basis = np.load(args.pod_dir / "pod_basis.npz")
    pod_model = joblib.load(args.pod_dir / "pod_mlp.joblib")
    sample = valid[["Re", "AoA_deg"]].iloc[[0]]
    repeats = 1000
    start = time.perf_counter()
    for _ in range(repeats):
        coeff = pod_model["y_scaler"].inverse_transform(pod_model["model"].predict(pod_model["x_scaler"].transform(sample)))
        _ = basis["mean_state"] + coeff @ basis["modes"]
    pod_inference_s = (time.perf_counter() - start) / repeats
    valid["pod_mlp_inference_s"] = pod_inference_s
    valid["speedup"] = valid.cfd_runtime_s / pod_inference_s

    # Cp plots: CFD surface pressure versus the reconstructed grid pressure.
    prediction = np.load(args.pod_dir / "validation_predictions.npz", allow_pickle=True)
    case_index = {str(case_id): index for index, case_id in enumerate(prediction["ids"])}
    mask, x, z = basis["fluid_mask"], basis["x"], basis["z"]
    cp_rows = []
    for _, row in valid.iterrows():
        index = case_index[str(row.id)]
        pressure_grid = restore_grid(prediction["fields"][index, 2], mask)
        interpolator = RegularGridInterpolator((z, x), pressure_grid, bounds_error=False, fill_value=np.nan)
        surfaces = surface_cp(windows_path(row.case_path), str(row.id), int(row.latest_time), float(row.Uinf_mps))
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        total_squared_error, total_points = 0.0, 0
        for label, xz, cfd_cp in surfaces:
            predicted_cp = interpolator(np.c_[xz[:, 1], xz[:, 0]]) / (0.5 * float(row.Uinf_mps) ** 2)
            finite = np.isfinite(predicted_cp)
            total_squared_error += float(np.sum((predicted_cp[finite] - cfd_cp[finite]) ** 2))
            total_points += int(finite.sum())
            axis.plot(xz[:, 0], cfd_cp, lw=2, label=f"CFD {label}")
            axis.plot(xz[finite, 0], predicted_cp[finite], "--", lw=1.5, label=f"POD-MLP {label}")
        cp_rmse = float(np.sqrt(total_squared_error / total_points))
        cp_rows.append({"id": row.id, "Re": row.Re, "AoA_deg": row.AoA_deg, "Cp_rmse": cp_rmse})
        axis.invert_yaxis()
        axis.set_xlabel("x/c")
        axis.set_ylabel("Cp")
        axis.set_title(f"{row.id}: surface pressure coefficient")
        axis.grid(alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
        fig.savefig(output / f"Cp_{row.id}.png", dpi=180)
        plt.close(fig)

    cp_metrics = pd.DataFrame(cp_rows)
    valid = valid.merge(cp_metrics, on=["id", "Re", "AoA_deg"])
    valid.to_csv(output / "validation_engineering_metrics.csv", index=False)
    train.to_csv(output / "training_aero_coefficients.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    for axis, cfd, predicted, label in ((axes[0], valid.Cl_cfd, valid.Cl_mlp, "CL"), (axes[1], valid.Cd_cfd, valid.Cd_mlp, "CD")):
        lower, upper = min(cfd.min(), predicted.min()), max(cfd.max(), predicted.max())
        axis.scatter(cfd, predicted, color="tab:blue")
        axis.plot([lower, upper], [lower, upper], "k--")
        axis.set_xlabel(f"CFD {label}")
        axis.set_ylabel(f"MLP {label}")
        axis.set_title(f"{label}: validation")
        axis.grid(alpha=0.3)
    fig.savefig(output / "aero_coefficients_true_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    axis.bar(["CFD\nmedian", "POD-MLP\ninference"], [valid.cfd_runtime_s.median(), pod_inference_s], color=["tab:orange", "tab:blue"])
    axis.set_ylabel("Time (s, logarithmic scale)")
    axis.set_yscale("log")
    axis.set_title(f"Median speed-up: {valid.speedup.median():,.0f}×")
    fig.savefig(output / "runtime_speedup.png", dpi=180)
    plt.close(fig)

    summary = {
        "validation_cases": len(valid),
        "Cp_rmse_mean": float(valid.Cp_rmse.mean()),
        "Cl_mse": float(np.mean((valid.Cl_mlp - valid.Cl_cfd) ** 2)),
        "Cd_mse": float(np.mean((valid.Cd_mlp - valid.Cd_cfd) ** 2)),
        "median_cfd_runtime_s": float(valid.cfd_runtime_s.median()),
        "pod_mlp_inference_s": float(pod_inference_s),
        "median_speedup": float(valid.speedup.median()),
        "coefficient_method_note": "CL/CD predicted by companion 2→64→64→2 ReLU MLP; not integrated from the u,w,p POD reconstruction.",
    }
    (output / "engineering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
