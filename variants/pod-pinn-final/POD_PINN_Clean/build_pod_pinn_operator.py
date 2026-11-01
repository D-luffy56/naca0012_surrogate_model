from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
POD_FILE = ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz"
GRID_FILE = ROOT / "artifacts" / "operator" / "delaunay_grid_and_masks.npz"
OUT = ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz"
SUMMARY_OUT = ROOT / "artifacts" / "operator" / "pod_pinn_operator_summary.json"

N_COLLOCATION = 4096
COLLOCATION_SEED = 20261031


def load_boundary(case_id: str) -> tuple[np.ndarray, dict]:
    with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
        boundary = archive["boundary"][:, :4].astype(np.float64)
        metadata = json.loads(archive["metadata"].item())
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    boundary /= np.array((u_inf, u_inf, u_inf**2, u_inf * chord))
    return boundary, metadata


def weighted_relative_l2(
    reference: np.ndarray, estimate: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    numerator = np.sum(weights[None, :, None] * (estimate - reference) ** 2, axis=(0, 1))
    denominator = np.sum(weights[None, :, None] * reference**2, axis=(0, 1))
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def main() -> None:
    with np.load(POD_FILE, allow_pickle=False) as pod:
        native_xz = pod["xz"].astype(np.float64)
        weights = pod["weights"].astype(np.float64)
        mean = pod["mean"].astype(np.float64)
        field_rms = pod["field_rms"].astype(np.float64)
        basis = pod["basis"].astype(np.float64)
        train_coefficients = pod["train_coefficients"].astype(np.float64)
        validation_coefficients = pod["validation_coefficients"].astype(np.float64)
        train_ids = pod["train_ids"].astype(str)
        validation_ids = pod["validation_ids"].astype(str)
    with np.load(GRID_FILE, allow_pickle=False) as grid:
        x = grid["x"].astype(np.float64)
        z = grid["z"].astype(np.float64)
        valid_mask = grid["valid_mask"].astype(bool)
        physics_mask = grid["physics_mask"].astype(bool)
    with np.load(DATA / "geometry.npz", allow_pickle=False) as geometry:
        boundary_xz = geometry["boundary_xz"].astype(np.float64)
        boundary_areas = geometry["boundary_areas"].astype(np.float64)
        boundary_patch = geometry["patch"].astype(np.int64)

    rank = basis.shape[0]
    sqrt_weights = np.sqrt(weights)
    native_modes = (
        basis.reshape(rank, len(native_xz), 4)
        / sqrt_weights[None, :, None]
        * field_rms[None, None, :]
    )
    expansion = np.concatenate((mean[None, :, :], native_modes), axis=0)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((xx.ravel(), zz.ravel()))
    triangulation = Delaunay(native_xz)
    grid_expansion = LinearNDInterpolator(
        triangulation, np.moveaxis(expansion, 0, 1).reshape(len(native_xz), -1), fill_value=np.nan
    )(query).reshape(len(z), len(x), rank + 1, 4)
    if not np.isfinite(grid_expansion).all():
        raise RuntimeError("Non-finite POD interpolation values")

    dx, dz = x[1] - x[0], z[1] - z[0]
    derivative_x = np.gradient(grid_expansion, dx, axis=1, edge_order=2)
    derivative_z = np.gradient(grid_expansion, dz, axis=0, edge_order=2)
    derivative_xx = np.gradient(derivative_x, dx, axis=1, edge_order=2)
    derivative_zz = np.gradient(derivative_z, dz, axis=0, edge_order=2)
    derivative_xz = 0.5 * (
        np.gradient(derivative_x, dz, axis=0, edge_order=2)
        + np.gradient(derivative_z, dx, axis=1, edge_order=2)
    )

    physics_indices = np.flatnonzero(physics_mask.ravel())
    if len(physics_indices) < N_COLLOCATION:
        raise RuntimeError("Insufficient safe physics points")
    rng = np.random.default_rng(COLLOCATION_SEED)
    collocation_indices = np.sort(
        rng.choice(physics_indices, size=N_COLLOCATION, replace=False)
    )
    coordinates = query[collocation_indices]

    def take(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, rank + 1, 4)[collocation_indices]

    train_boundary = np.stack([load_boundary(case_id)[0] for case_id in train_ids])
    validation_boundary = np.stack([load_boundary(case_id)[0] for case_id in validation_ids])
    augmented_train = np.column_stack((np.ones(len(train_ids)), train_coefficients))
    affine_map, *_ = np.linalg.lstsq(
        augmented_train, train_boundary.reshape(len(train_ids), -1), rcond=None
    )
    train_boundary_estimate = (augmented_train @ affine_map).reshape(train_boundary.shape)
    validation_boundary_estimate = (
        np.column_stack((np.ones(len(validation_ids)), validation_coefficients)) @ affine_map
    ).reshape(validation_boundary.shape)
    area_weights = boundary_areas / boundary_areas.sum()
    train_boundary_error = weighted_relative_l2(
        train_boundary, train_boundary_estimate, area_weights
    )
    validation_boundary_error = weighted_relative_l2(
        validation_boundary, validation_boundary_estimate, area_weights
    )
    boundary_scales = np.sqrt(np.mean(np.square(train_boundary), axis=(0, 1)))

    coefficient_mean = train_coefficients.mean(axis=0)
    coefficient_scale = train_coefficients.std(axis=0, ddof=0)
    if np.any(coefficient_scale <= 1e-12):
        raise RuntimeError("Degenerate POD coefficient scale")

    train_parameters = []
    validation_parameters = []
    for case_id in train_ids:
        metadata = load_boundary(case_id)[1]
        train_parameters.append((float(metadata["Re"]), float(metadata["AoA_deg"])))
    for case_id in validation_ids:
        metadata = load_boundary(case_id)[1]
        validation_parameters.append((float(metadata["Re"]), float(metadata["AoA_deg"])))

    parameter_center = np.array((4.0e6, 5.0), dtype=np.float64)
    parameter_scale = np.array((2.0e6, 9.0), dtype=np.float64)
    grid_mean = grid_expansion[:, :, 0]
    grid_modes = np.moveaxis(grid_expansion[:, :, 1:], 2, 0)
    np.savez_compressed(
        OUT,
        x=x,
        z=z,
        valid_mask=valid_mask,
        physics_mask=physics_mask,
        collocation_indices=collocation_indices,
        collocation_xz=coordinates,
        expansion_values=take(grid_expansion).astype(np.float32),
        expansion_dx=take(derivative_x).astype(np.float32),
        expansion_dz=take(derivative_z).astype(np.float32),
        expansion_dxx=take(derivative_xx).astype(np.float32),
        expansion_dxz=take(derivative_xz).astype(np.float32),
        expansion_dzz=take(derivative_zz).astype(np.float32),
        grid_mean=grid_mean.astype(np.float32),
        grid_modes=grid_modes.astype(np.float32),
        train_coefficients=train_coefficients,
        validation_coefficients=validation_coefficients,
        coefficient_mean=coefficient_mean,
        coefficient_scale=coefficient_scale,
        train_parameters=np.asarray(train_parameters),
        validation_parameters=np.asarray(validation_parameters),
        parameter_center=parameter_center,
        parameter_scale=parameter_scale,
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_boundary=train_boundary.astype(np.float32),
        validation_boundary=validation_boundary.astype(np.float32),
        boundary_affine_intercept=affine_map[0].reshape(len(boundary_xz), 4).astype(np.float32),
        boundary_affine_modes=affine_map[1:].reshape(rank, len(boundary_xz), 4).astype(np.float32),
        boundary_scales=boundary_scales,
        boundary_xz=boundary_xz,
        boundary_areas=boundary_areas,
        boundary_patch=boundary_patch,
    )
    summary = {
        "rank": rank,
        "collocation_points": N_COLLOCATION,
        "collocation_seed": COLLOCATION_SEED,
        "parameter_center": parameter_center.tolist(),
        "parameter_scale": parameter_scale.tolist(),
        "coefficient_scale_min": float(coefficient_scale.min()),
        "coefficient_scale_max": float(coefficient_scale.max()),
        "boundary_scales": dict(zip(("u", "w", "p", "nut"), map(float, boundary_scales))),
        "train_boundary_affine_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, train_boundary_error))
        ),
        "validation_boundary_affine_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, validation_boundary_error))
        ),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
