from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
OUT = ROOT / "artifacts" / "operator"

NX = 251
NZ = 201
X_LIMITS = (-0.5, 2.0)
Z_LIMITS = (-0.5, 0.5)
WALL_EXCLUSION = 0.02


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def ddx(values: np.ndarray, dx: float) -> np.ndarray:
    return np.gradient(values, dx, axis=1, edge_order=2)


def ddz(values: np.ndarray, dz: float) -> np.ndarray:
    return np.gradient(values, dz, axis=0, edge_order=2)


def airfoil_polygon(segments: np.ndarray, patch: np.ndarray) -> np.ndarray:
    # Patch 0 runs lower trailing-edge -> leading-edge; patch 1 runs upper
    # leading-edge -> trailing-edge. Retain that ordering to form a closed foil.
    curves = []
    for label in (0, 1):
        selected = segments[patch == label]
        curves.append(np.vstack((selected[0, 0], selected[:, 1])))
    return np.vstack((curves[0], curves[1][1:]))


def manufactured_derivative_check(x: np.ndarray, z: np.ndarray) -> dict[str, float]:
    xx, zz = np.meshgrid(x, z, indexing="xy")
    ax, az = 2.1, 1.7
    field = np.sin(ax * xx) * np.cos(az * zz) + 0.1 * xx**3 - 0.2 * zz**3
    exact = {
        "dx": ax * np.cos(ax * xx) * np.cos(az * zz) + 0.3 * xx**2,
        "dz": -az * np.sin(ax * xx) * np.sin(az * zz) - 0.6 * zz**2,
        "dxx": -(ax**2) * np.sin(ax * xx) * np.cos(az * zz) + 0.6 * xx,
        "dzz": -(az**2) * np.sin(ax * xx) * np.cos(az * zz) - 1.2 * zz,
        "dxz": -(ax * az) * np.cos(ax * xx) * np.sin(az * zz),
    }
    dx, dz = x[1] - x[0], z[1] - z[0]
    fx, fz = ddx(field, dx), ddz(field, dz)
    numeric = {
        "dx": fx,
        "dz": fz,
        "dxx": ddx(fx, dx),
        "dzz": ddz(fz, dz),
        "dxz": 0.5 * (ddz(fx, dz) + ddx(fz, dx)),
    }
    interior = np.zeros(field.shape, dtype=bool)
    interior[3:-3, 3:-3] = True
    return {
        name: rms((numeric[name] - truth)[interior]) / max(rms(truth[interior]), 1e-15)
        for name, truth in exact.items()
    }


def nondimensional_fields(case_path: Path) -> tuple[np.ndarray, dict]:
    with np.load(case_path, allow_pickle=False) as archive:
        fields = archive["fields"][:, :4].astype(np.float64)
        metadata = json.loads(archive["metadata"].item())
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    fields /= np.array((u_inf, u_inf, u_inf**2, u_inf * chord))
    return fields, metadata


def residual_metrics(
    grid_fields: np.ndarray,
    safe_mask: np.ndarray,
    dx: float,
    dz: float,
    reynolds: float,
) -> dict[str, float]:
    u, w, p, nut = np.moveaxis(grid_fields, -1, 0)
    ux, uz = ddx(u, dx), ddz(u, dz)
    wx, wz = ddx(w, dx), ddz(w, dz)
    divergence = ux + wz
    nu_eff = 1.0 / reynolds + nut
    tau_xx = nu_eff * (2.0 * ux - (2.0 / 3.0) * divergence)
    tau_xz = nu_eff * (uz + wx)
    tau_zz = nu_eff * (2.0 * wz - (2.0 / 3.0) * divergence)
    pressure_x, pressure_z = ddx(p, dx), ddz(p, dz)
    residual_x = u * ux + w * uz + pressure_x - ddx(tau_xx, dx) - ddz(tau_xz, dz)
    residual_z = u * wx + w * wz + pressure_z - ddx(tau_xz, dx) - ddz(tau_zz, dz)

    continuity_scale = rms(np.stack((ux[safe_mask], wz[safe_mask]), axis=-1)) * np.sqrt(2.0)
    x_terms = np.stack(
        (u * ux, w * uz, pressure_x, -ddx(tau_xx, dx), -ddz(tau_xz, dz)), axis=-1
    )[safe_mask]
    z_terms = np.stack(
        (u * wx, w * wz, pressure_z, -ddx(tau_xz, dx), -ddz(tau_zz, dz)), axis=-1
    )[safe_mask]
    x_scale = rms(x_terms) * np.sqrt(x_terms.shape[-1])
    z_scale = rms(z_terms) * np.sqrt(z_terms.shape[-1])
    return {
        "continuity_rms": rms(divergence[safe_mask]),
        "momentum_x_rms": rms(residual_x[safe_mask]),
        "momentum_z_rms": rms(residual_z[safe_mask]),
        "continuity_normalized": rms(divergence[safe_mask]) / max(continuity_scale, 1e-15),
        "momentum_x_normalized": rms(residual_x[safe_mask]) / max(x_scale, 1e-15),
        "momentum_z_normalized": rms(residual_z[safe_mask]) / max(z_scale, 1e-15),
        "continuity_abs_p95": float(np.percentile(np.abs(divergence[safe_mask]), 95)),
        "momentum_x_abs_p95": float(np.percentile(np.abs(residual_x[safe_mask]), 95)),
        "momentum_z_abs_p95": float(np.percentile(np.abs(residual_z[safe_mask]), 95)),
    }


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    case_ids = sorted(record["id"] for record in manifest)
    with np.load(DATA / "geometry.npz", allow_pickle=False) as geometry:
        native_xz = geometry["xz"].astype(np.float64)
        native_volumes = geometry["volumes"].astype(np.float64)
        segments = geometry["segments"].astype(np.float64)
        patch = geometry["patch"].astype(np.int64)

    x = np.linspace(*X_LIMITS, NX)
    z = np.linspace(*Z_LIMITS, NZ)
    dx, dz = x[1] - x[0], z[1] - z[0]
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((xx.ravel(), zz.ravel()))
    triangulation = Delaunay(native_xz)
    polygon = airfoil_polygon(segments, patch)
    inside_airfoil = PolygonPath(polygon, closed=True).contains_points(query).reshape(xx.shape)
    inside_hull = (triangulation.find_simplex(query) >= 0).reshape(xx.shape)
    valid = inside_hull & ~inside_airfoil
    stencil_safe = binary_erosion(valid, structure=np.ones((5, 5), dtype=bool), border_value=0)
    wall_distance = distance_transform_edt(~inside_airfoil, sampling=(dz, dx))
    safe = stencil_safe & (wall_distance >= WALL_EXCLUSION)
    if safe.sum() < 0.8 * valid.sum():
        raise RuntimeError("The wall-safe physics mask discarded unexpectedly many grid points")

    derivative_errors = manufactured_derivative_check(x, z)
    if max(derivative_errors.values()) > 2e-3:
        raise RuntimeError(f"Manufactured derivative test failed: {derivative_errors}")

    case_metrics: dict[str, dict[str, float]] = {}
    roundtrip_errors: dict[str, dict[str, float]] = {}
    nearfield_native = (
        (native_xz[:, 0] >= x[2])
        & (native_xz[:, 0] <= x[-3])
        & (native_xz[:, 1] >= z[2])
        & (native_xz[:, 1] <= z[-3])
    )
    native_weights = native_volumes[nearfield_native]
    native_weights /= native_weights.sum()

    for case_id in case_ids:
        fields, metadata = nondimensional_fields(DATA / f"{case_id}.npz")
        grid_fields = LinearNDInterpolator(triangulation, fields, fill_value=np.nan)(query).reshape(
            NZ, NX, 4
        )
        if not np.isfinite(grid_fields).all():
            raise RuntimeError(f"Non-finite interpolation result for {case_id}")
        case_metrics[case_id] = residual_metrics(
            grid_fields, safe, dx, dz, float(metadata["Re"])
        )

        reconstructed_native = RegularGridInterpolator(
            (z, x), grid_fields, method="linear", bounds_error=True
        )(native_xz[nearfield_native, ::-1])
        reference_native = fields[nearfield_native]
        numerator = np.sum(
            native_weights[:, None] * np.square(reconstructed_native - reference_native), axis=0
        )
        denominator = np.sum(native_weights[:, None] * np.square(reference_native), axis=0)
        relative = np.sqrt(numerator / np.maximum(denominator, 1e-30))
        roundtrip_errors[case_id] = dict(zip(("u", "w", "p", "nut"), map(float, relative)))

    metric_names = tuple(next(iter(case_metrics.values())))
    aggregate = {
        metric: {
            "median": float(np.median([values[metric] for values in case_metrics.values()])),
            "p95": float(np.percentile([values[metric] for values in case_metrics.values()], 95)),
            "maximum": float(np.max([values[metric] for values in case_metrics.values()])),
        }
        for metric in metric_names
    }
    roundtrip_aggregate = {
        field: {
            "median": float(np.median([values[field] for values in roundtrip_errors.values()])),
            "p95": float(np.percentile([values[field] for values in roundtrip_errors.values()], 95)),
            "maximum": float(np.max([values[field] for values in roundtrip_errors.values()])),
        }
        for field in ("u", "w", "p", "nut")
    }
    summary = {
        "grid": {
            "nx": NX,
            "nz": NZ,
            "x_limits": X_LIMITS,
            "z_limits": Z_LIMITS,
            "dx": float(dx),
            "dz": float(dz),
            "valid_points": int(valid.sum()),
            "physics_points": int(safe.sum()),
            "wall_exclusion_chords": WALL_EXCLUSION,
        },
        "manufactured_relative_rms_derivative_errors": derivative_errors,
        "reference_cfd_residual_aggregate": aggregate,
        "delaunay_grid_roundtrip_relative_l2_aggregate": roundtrip_aggregate,
        "case_metrics": case_metrics,
        "roundtrip_errors": roundtrip_errors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "delaunay_grid_and_masks.npz",
        x=x,
        z=z,
        polygon=polygon,
        valid_mask=valid,
        physics_mask=safe,
        wall_distance=wall_distance,
    )
    (OUT / "operator_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("case_metrics", "roundtrip_errors")}, indent=2))


if __name__ == "__main__":
    main()
