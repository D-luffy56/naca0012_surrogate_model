"""Build a POD basis and MLP surrogate from fixed-geometry OpenFOAM VTK exports.

Inputs are the verified training/validation manifests produced by the CFD batch
runner. The CFD mesh is common to every case; fields are sampled onto a shared
2D x-z grid before POD. The solid NACA 0012 region is excluded from the state
vector, so no artificial values inside the airfoil influence the SVD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


FIELDS = ("u", "w", "p")  # OpenFOAM 2D plane is x-z; U=(u, v_thin, w).


def windows_path(value: str) -> Path:
    """Convert a manifest's WSL /mnt/d path into a Windows path when needed."""
    if value.startswith("/mnt/d/"):
        return Path("D:/" + value.removeprefix("/mnt/d/"))
    return Path(value)


def latest_internal_vtu(row: pd.Series) -> Path:
    case = windows_path(row.case_path)
    candidate = case / "VTK" / f"{row.id}_{row.latest_time}" / "internal.vtu"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing CFD field export: {candidate}")
    return candidate


def naca0012_polygon(points: int = 401) -> np.ndarray:
    """Return a closed NACA 0012 x-z boundary in chord coordinates."""
    x = (1.0 - np.cos(np.linspace(0.0, np.pi, points))) / 2.0
    thickness = 5 * 0.12 * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return np.vstack((np.c_[x, thickness], np.c_[x[::-1], -thickness[::-1]]))


def interpolation_plan(vtu_path: Path, grid_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reusable triangle interpolation weights from the common CFD mesh."""
    mesh = meshio.read(vtu_path)
    # The 2D OpenFOAM mesh is extruded in y. Collapse duplicate x-z points.
    xy = mesh.points[:, [0, 2]]
    unique_xy, unique_indices = np.unique(xy, axis=0, return_index=True)
    triangulation = Delaunay(unique_xy)
    simplices = triangulation.find_simplex(grid_xy)
    if np.any(simplices < 0):
        raise ValueError("Chosen POD grid extends outside the CFD mesh.")
    transforms = triangulation.transform[simplices]
    delta = grid_xy - transforms[:, 2]
    barycentric = np.einsum("nij,nj->ni", transforms[:, :2], delta)
    weights = np.c_[barycentric, 1.0 - barycentric.sum(axis=1)]
    vertices = triangulation.simplices[simplices]
    return unique_indices, vertices, weights


def sample_case(vtu_path: Path, unique_indices: np.ndarray, vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mesh = meshio.read(vtu_path)
    velocity = mesh.point_data["U"][unique_indices]
    pressure = mesh.point_data["p"][unique_indices]
    raw_fields = (velocity[:, 0], velocity[:, 2], pressure)
    return np.stack([(raw[vertices] * weights).sum(axis=1) for raw in raw_fields])


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "Re", "AoA_deg", "latest_time", "case_path", "converged", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame[(frame.converged.astype(str).str.lower() == "true") & (frame.status == "outputs_verified")].copy()
    if frame.empty:
        raise ValueError(f"No verified converged cases in {path}")
    return frame.sort_values("id").reset_index(drop=True)


def error_by_field(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        field: float(np.linalg.norm(prediction[i] - reference[i]) / np.linalg.norm(reference[i]))
        for i, field in enumerate(FIELDS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.999)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train = load_manifest(args.training_manifest)
    validation = load_manifest(args.validation_manifest) if args.validation_manifest else None

    x = np.linspace(-2.0, 4.0, args.nx)
    z = np.linspace(-1.5, 1.5, args.nz)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    grid_xy = np.c_[xx.ravel(), zz.ravel()]
    solid_mask = MplPath(naca0012_polygon()).contains_points(grid_xy)
    fluid_mask = ~solid_mask

    unique_indices, vertices, weights = interpolation_plan(latest_internal_vtu(train.iloc[0]), grid_xy)

    def fields_for(frame: pd.DataFrame) -> np.ndarray:
        result = []
        for _, row in frame.iterrows():
            sampled = sample_case(latest_internal_vtu(row), unique_indices, vertices, weights)
            result.append(sampled[:, fluid_mask])
        return np.asarray(result, dtype=np.float32)

    train_fields = fields_for(train)
    validation_fields = fields_for(validation) if validation is not None else None

    # Per-field scale avoids pressure magnitude dominating the velocity POD modes.
    field_scale = train_fields.reshape(len(train), len(FIELDS), -1).std(axis=(0, 2))
    field_scale[field_scale == 0] = 1.0
    scaled = train_fields / field_scale[None, :, None]
    state = scaled.reshape(len(train), -1)
    mean_state = state.mean(axis=0)
    centred = state - mean_state
    modes, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    rank = int(np.searchsorted(energy, args.energy) + 1)
    pod_modes = right_vectors[:rank]
    coefficients = centred @ pod_modes.T
    train_reconstruction = (mean_state + coefficients @ pod_modes).reshape(train_fields.shape) * field_scale[None, :, None]

    np.savez_compressed(
        output / "pod_basis.npz",
        x=x,
        z=z,
        fluid_mask=fluid_mask.reshape(zz.shape),
        field_scale=field_scale,
        mean_state=mean_state,
        modes=pod_modes,
        singular_values=singular_values,
        energy=energy,
        coefficients=coefficients,
    )
    np.savez_compressed(output / "training_snapshots.npz", fields=train_fields, ids=train.id.to_numpy(dtype=str))
    train.to_csv(output / "training_cases.csv", index=False)

    # The MLP maps operating conditions to POD coefficients. This is the
    # architecture specified in the AE646 project brief: 2 → 64 → 64 → r.
    # MLPRegressor minimizes mean-squared error on the standardized coefficients.
    x_scaler = StandardScaler().fit(train[["Re", "AoA_deg"]])
    y_scaler = StandardScaler().fit(coefficients)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs",
        max_iter=5000, random_state=42,
    ).fit(x_scaler.transform(train[["Re", "AoA_deg"]]), y_scaler.transform(coefficients))
    joblib.dump({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}, output / "pod_mlp.joblib")

    train_prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(train[["Re", "AoA_deg"]])))
    train_mlp_fields = (mean_state + train_prediction @ pod_modes).reshape(train_fields.shape) * field_scale[None, :, None]
    summary: dict[str, object] = {
        "training_cases": int(len(train)),
        "validation_cases": int(0 if validation is None else len(validation)),
        "grid_shape": [args.nz, args.nx],
        "state_fields": list(FIELDS),
        "pod_energy_threshold": args.energy,
        "retained_modes": rank,
        "retained_energy": float(energy[rank - 1]),
        "training_pod_relative_l2": error_by_field(train_fields, train_reconstruction),
        "training_mlp_relative_l2": error_by_field(train_fields, train_mlp_fields),
        "mlp_architecture": [2, 64, 64, rank],
        "mlp_activation": "ReLU",
        "mlp_loss": "MSE on standardized POD coefficients",
        "training_coefficient_mse": float(np.mean((train_prediction - coefficients) ** 2)),
    }

    if validation is not None:
        validation.to_csv(output / "validation_cases.csv", index=False)
        np.savez_compressed(output / "validation_snapshots.npz", fields=validation_fields, ids=validation.id.to_numpy(dtype=str))
        predicted_coefficients = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(validation[["Re", "AoA_deg"]]))
        )
        validation_state = (validation_fields / field_scale[None, :, None]).reshape(len(validation), -1)
        validation_coefficients = (validation_state - mean_state) @ pod_modes.T
        predicted_fields = (mean_state + predicted_coefficients @ pod_modes).reshape(validation_fields.shape)
        predicted_fields *= field_scale[None, :, None]
        summary["validation_mlp_relative_l2"] = error_by_field(validation_fields, predicted_fields)
        summary["validation_coefficient_mse"] = float(np.mean((predicted_coefficients - validation_coefficients) ** 2))
        np.savez_compressed(
            output / "validation_predictions.npz",
            fields=predicted_fields,
            ids=validation.id.to_numpy(dtype=str),
            true_coefficients=validation_coefficients,
            predicted_coefficients=predicted_coefficients,
        )

    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(energy) + 1), energy, marker="o", ms=3)
    plt.axhline(args.energy, color="tab:red", ls="--", label=f"target = {args.energy:.3%}")
    plt.axvline(rank, color="tab:green", ls="--", label=f"retained modes = {rank}")
    plt.xlabel("POD modes retained")
    plt.ylabel("Cumulative energy")
    plt.ylim(0.0, 1.01)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "pod_energy.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
