from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from train_fixed_pod_pinn import pressure_coefficients


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
MODELS = ROOT / "artifacts" / "models"
OPERATOR = ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz"
POD = ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz"
OUT = ROOT / "artifacts" / "extended_analysis"
FIGURES = OUT / "figures"
FIELDS = ("u", "w", "p", "nut")


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
        fields = archive["fields"][:, :4].astype(np.float64)
        boundary = archive["boundary"][:, :4].astype(np.float64)
        metadata = json.loads(archive["metadata"].item())
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    scales = np.array((u_inf, u_inf, u_inf**2, u_inf * chord))
    return fields / scales, boundary / scales, metadata


def native_reconstruction(
    coefficients: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    field_rms: np.ndarray,
) -> np.ndarray:
    normalized = (
        (coefficients @ basis).reshape(len(coefficients), len(weights), 4)
        / np.sqrt(weights)[None, :, None]
    )
    return mean[None, :, :] + normalized * field_rms[None, None, :]


def case_relative_errors(
    reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    numerator = np.sum(weights[None, :, None] * (prediction - reference) ** 2, axis=1)
    denominator = np.sum(weights[None, :, None] * reference**2, axis=1)
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def reconstruct_grid(coefficients: np.ndarray, mean: np.ndarray, modes: np.ndarray) -> np.ndarray:
    return mean[None, :, :, :] + np.einsum("br,rzxf->bzxf", coefficients, modes)


def grid_residuals(fields: np.ndarray, x: np.ndarray, z: np.ndarray, reynolds: float):
    dx, dz = x[1] - x[0], z[1] - z[0]

    def ddx(values):
        return np.gradient(values, dx, axis=1, edge_order=2)

    def ddz(values):
        return np.gradient(values, dz, axis=0, edge_order=2)

    u, w, p, nut = np.moveaxis(fields, -1, 0)
    ux, uz = ddx(u), ddz(u)
    wx, wz = ddx(w), ddz(w)
    div = ux + wz
    nueff = 1.0 / reynolds + np.maximum(nut, 0.0)
    txx = nueff * (2.0 * ux - (2.0 / 3.0) * div)
    txz = nueff * (uz + wx)
    tzz = nueff * (2.0 * wz - (2.0 / 3.0) * div)
    rx = u * ux + w * uz + ddx(p) - ddx(txx) - ddz(txz)
    rz = u * wx + w * wz + ddz(p) - ddx(txz) - ddz(tzz)
    return div, rx, rz


def contour_figure(
    case_id: str,
    metadata: dict,
    reference: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    output: Path,
) -> None:
    reference = reference.copy()
    prediction = prediction.copy()
    reference[~valid_mask] = np.nan
    prediction[~valid_mask] = np.nan
    labels = (r"$u/U_\infty$", r"$w/U_\infty$", r"$p/U_\infty^2$", r"$\nu_t/(U_\infty c)$")
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 10.0), sharex=True, sharey=True, constrained_layout=True)
    extent = (x[0], x[-1], z[0], z[-1])
    for index, label in enumerate(labels):
        truth = reference[:, :, index]
        estimate = prediction[:, :, index]
        error = estimate - truth
        lower, upper = np.nanpercentile(np.concatenate((truth.ravel(), estimate.ravel())), (1, 99))
        error_limit = max(float(np.nanpercentile(np.abs(error), 99)), 1e-12)
        images = (
            axes[index, 0].imshow(truth, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[index, 1].imshow(estimate, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[index, 2].imshow(error, origin="lower", extent=extent, aspect="auto", vmin=-error_limit, vmax=error_limit, cmap="coolwarm"),
        )
        for column, image in enumerate(images):
            fig.colorbar(image, ax=axes[index, column], fraction=0.025, pad=0.02)
            axes[index, column].set_ylabel(f"z/c\n{label}")
    for column, title in enumerate(("CFD / Delaunay", "Adaptive POD–PINN", "Prediction − CFD")):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("x/c")
    fig.suptitle(f"Held-out {case_id}: Re={metadata['Re']:.2g}, AoA={metadata['AoA_deg']}°", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    with np.load(POD, allow_pickle=False) as pod:
        native_xz = pod["xz"].astype(np.float64)
        weights = pod["weights"].astype(np.float64)
        mean = pod["mean"].astype(np.float64)
        field_rms = pod["field_rms"].astype(np.float64)
        basis = pod["basis"].astype(np.float64)
        singular_values = pod["singular_values"].astype(np.float64)
        cumulative_energy = pod["cumulative_energy"].astype(np.float64)
        validation_ids = pod["validation_ids"].astype(str)
        validation_coefficients = pod["validation_coefficients"].astype(np.float64)
    with np.load(OPERATOR, allow_pickle=False) as operator:
        x = operator["x"].astype(np.float64)
        z = operator["z"].astype(np.float64)
        valid_mask = operator["valid_mask"].astype(bool)
        physics_mask = operator["physics_mask"].astype(bool)
        grid_mean = operator["grid_mean"].astype(np.float64)
        grid_modes = operator["grid_modes"].astype(np.float64)
        boundary_areas = operator["boundary_areas"].astype(np.float64)
        boundary_patch = operator["boundary_patch"].astype(np.int64)
    with np.load(DATA / "geometry.npz", allow_pickle=False) as geometry:
        boundary_xz = geometry["boundary_xz"].astype(np.float64)
        boundary_normals = geometry["boundary_normals"].astype(np.float64)

    fixed_file = MODELS / "pod_pinn_2x64_rank25_predictions.npz"
    adaptive_file = MODELS / "pod_pinn_2x64_rank25_self_adaptive_predictions.npz"
    with np.load(fixed_file, allow_pickle=False) as archive:
        fixed_coefficients = archive["validation_coefficients"].astype(np.float64)
        fixed_boundary = archive["validation_boundary"].astype(np.float64)
    with np.load(adaptive_file, allow_pickle=False) as archive:
        adaptive_coefficients = archive["validation_coefficients"].astype(np.float64)
        adaptive_boundary = archive["validation_boundary"].astype(np.float64)

    loaded = [load_case(case_id) for case_id in validation_ids]
    reference_native = np.stack([entry[0] for entry in loaded])
    reference_boundary = np.stack([entry[1] for entry in loaded])
    metadata = [entry[2] for entry in loaded]
    parameters = np.array([(row["Re"], row["AoA_deg"]) for row in metadata], dtype=float)
    fixed_native = native_reconstruction(fixed_coefficients, mean, basis, weights, field_rms)
    adaptive_native = native_reconstruction(adaptive_coefficients, mean, basis, weights, field_rms)
    fixed_errors = case_relative_errors(reference_native, fixed_native, weights)
    adaptive_errors = case_relative_errors(reference_native, adaptive_native, weights)

    rows = []
    for index, case_id in enumerate(validation_ids):
        rows.append(
            {
                "case_id": case_id,
                "Re": int(parameters[index, 0]),
                "AoA_deg": float(parameters[index, 1]),
                **{f"fixed_{field}": float(fixed_errors[index, j]) for j, field in enumerate(FIELDS)},
                **{f"adaptive_{field}": float(adaptive_errors[index, j]) for j, field in enumerate(FIELDS)},
            }
        )
    with (OUT / "casewise_validation_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # 1. Case-wise validation heatmaps.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    maximum = 100.0 * max(fixed_errors.max(), adaptive_errors.max())
    for ax, values, title in zip(axes, (fixed_errors, adaptive_errors), ("Fixed 64×64", "Self-adaptive 64×64")):
        image = ax.imshow(100.0 * values, aspect="auto", cmap="magma_r", vmin=0, vmax=maximum)
        ax.set_xticks(range(4), FIELDS)
        ax.set_yticks(range(len(validation_ids)), validation_ids)
        ax.set_title(title)
        ax.set_xlabel("Field")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{100*values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if 100*values[i,j] > 0.55*maximum else "black")
    fig.colorbar(image, ax=axes, label="Case relative $L_2$ error [%]", shrink=0.85)
    fig.savefig(FIGURES / "casewise_error_heatmaps.png", dpi=180)
    plt.close(fig)

    # 2. Error dependence on angle and Reynolds number.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    reynolds_values = sorted(set(parameters[:, 0]))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(reynolds_values)))
    for field_index, (ax, field) in enumerate(zip(axes.ravel(), FIELDS)):
        for color, reynolds in zip(colors, reynolds_values):
            chosen = np.where(parameters[:, 0] == reynolds)[0]
            order = chosen[np.argsort(parameters[chosen, 1])]
            ax.plot(parameters[order, 1], 100*fixed_errors[order, field_index], "o--", color=color, alpha=0.75)
            ax.plot(parameters[order, 1], 100*adaptive_errors[order, field_index], "s-", color=color,
                    label=f"Re={reynolds/1e6:.1f}M" if field_index == 0 else None)
        ax.set_title(field)
        ax.set_xlabel("AoA [deg]")
        ax.set_ylabel("Relative $L_2$ error [%]")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("Validation error versus Reynolds number and angle of attack\n"
                 "dashed circles: fixed; solid squares: self-adaptive")
    fig.savefig(FIGURES / "error_vs_parameters.png", dpi=180)
    plt.close(fig)

    # 3. Surface Cp comparisons on low, medium, and high incidence cases.
    selected_ids = ("V01", "V05", "V06")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    source_styles = (("CFD", "black"), ("Fixed", "tab:blue"), ("Adaptive", "tab:orange"))
    for ax, case_id in zip(axes, selected_ids):
        index = int(np.where(validation_ids == case_id)[0][0])
        for source_name, color in source_styles:
            values = {"CFD": reference_boundary, "Fixed": fixed_boundary, "Adaptive": adaptive_boundary}[source_name]
            for patch_label, linestyle, surface_name in ((1, "-", "upper"), (0, "--", "lower")):
                chosen = np.where(boundary_patch == patch_label)[0]
                order = chosen[np.argsort(boundary_xz[chosen, 0])]
                label = f"{source_name} {surface_name}"
                ax.plot(boundary_xz[order, 0], 2.0*values[index, order, 2], linestyle,
                        color=color, linewidth=1.5, label=label)
        ax.invert_yaxis()
        ax.set_title(f"{case_id}: Re={parameters[index,0]/1e6:.1f}M, AoA={parameters[index,1]:g}°")
        ax.set_xlabel("x/c")
        ax.set_ylabel("$C_p$")
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=7, ncols=2)
    fig.suptitle("Surface pressure-coefficient distributions")
    fig.savefig(FIGURES / "surface_cp_comparisons.png", dpi=180)
    plt.close(fig)

    # 4. Pressure-force parity plots.
    true_forces = pressure_coefficients(reference_boundary, parameters, boundary_areas,
                                        boundary_normals, boundary_patch)
    fixed_forces = pressure_coefficients(fixed_boundary, parameters, boundary_areas,
                                         boundary_normals, boundary_patch)
    adaptive_forces = pressure_coefficients(adaptive_boundary, parameters, boundary_areas,
                                            boundary_normals, boundary_patch)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for component, (ax, label) in enumerate(zip(axes, ("$C_{l,p}$", "$C_{d,p}$"))):
        all_values = np.concatenate((true_forces[:, component], fixed_forces[:, component], adaptive_forces[:, component]))
        lower, upper = all_values.min(), all_values.max()
        padding = 0.08 * max(upper - lower, 1e-6)
        ax.plot([lower-padding, upper+padding], [lower-padding, upper+padding], "k--", label="ideal")
        ax.scatter(true_forces[:, component], fixed_forces[:, component], marker="o", s=55, label="fixed")
        ax.scatter(true_forces[:, component], adaptive_forces[:, component], marker="s", s=55, label="adaptive")
        for i, case_id in enumerate(validation_ids):
            ax.annotate(case_id, (true_forces[i, component], adaptive_forces[i, component]),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel(f"CFD pressure-only {label}")
        ax.set_ylabel(f"Predicted pressure-only {label}")
        ax.set_title(f"{label} parity")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "pressure_force_parity.png", dpi=180)
    plt.close(fig)

    # 5. POD spectrum and cumulative energy.
    mode_number = np.arange(1, len(singular_values) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    axes[0].semilogy(mode_number, singular_values / singular_values[0], "o-")
    axes[0].axvline(25, color="tab:red", linestyle="--", label="fixed rank 25")
    axes[0].set_xlabel("Mode")
    axes[0].set_ylabel("Normalized singular value")
    axes[0].set_title("POD singular-value spectrum")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(mode_number, 100*cumulative_energy, "o-")
    axes[1].axhline(99.9, color="black", linestyle=":", label="99.9%")
    axes[1].axvline(25, color="tab:red", linestyle="--", label="fixed rank 25")
    axes[1].set_xlabel("Retained modes")
    axes[1].set_ylabel("Cumulative fluctuation energy [%]")
    axes[1].set_title("POD cumulative energy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(FIGURES / "pod_spectrum_and_energy.png", dpi=180)
    plt.close(fig)

    # 6. First three POD mode fields.
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    extent = (x[0], x[-1], z[0], z[-1])
    for mode in range(3):
        for field in range(4):
            values = grid_modes[mode, :, :, field].copy()
            values[~valid_mask] = np.nan
            limit = max(float(np.nanpercentile(np.abs(values), 99)), 1e-12)
            image = axes[mode, field].imshow(values, origin="lower", extent=extent, aspect="auto",
                                             vmin=-limit, vmax=limit, cmap="coolwarm")
            fig.colorbar(image, ax=axes[mode, field], fraction=0.03, pad=0.02)
            axes[mode, field].set_title(f"Mode {mode+1}: {FIELDS[field]}")
            if field == 0:
                axes[mode, field].set_ylabel("z/c")
            if mode == 2:
                axes[mode, field].set_xlabel("x/c")
    fig.suptitle("Leading physical POD modes on the Delaunay grid")
    fig.savefig(FIGURES / "leading_pod_modes.png", dpi=180)
    plt.close(fig)

    # 7. Leading coefficient parity.
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), constrained_layout=True)
    for mode, ax in enumerate(axes.ravel()):
        truth = validation_coefficients[:, mode]
        fixed_values = fixed_coefficients[:, mode]
        adaptive_values = adaptive_coefficients[:, mode]
        all_values = np.concatenate((truth, fixed_values, adaptive_values))
        lower, upper = all_values.min(), all_values.max()
        padding = 0.08 * max(upper - lower, 1e-9)
        ax.plot([lower-padding, upper+padding], [lower-padding, upper+padding], "k--")
        ax.scatter(truth, fixed_values, marker="o", label="fixed")
        ax.scatter(truth, adaptive_values, marker="s", label="adaptive")
        ax.set_title(f"POD coefficient {mode+1}")
        ax.set_xlabel("Oracle projection")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(FIGURES / "leading_coefficient_parity.png", dpi=180)
    plt.close(fig)

    # Structured-grid CFD truth and model fields used by the remaining plots.
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((xx.ravel(), zz.ravel()))
    triangulation = Delaunay(native_xz)
    reference_grids = []
    for case_index in range(len(validation_ids)):
        reference_grids.append(
            LinearNDInterpolator(triangulation, reference_native[case_index], fill_value=np.nan)(query)
            .reshape(len(z), len(x), 4)
        )
    reference_grids = np.stack(reference_grids)
    fixed_grids = reconstruct_grid(fixed_coefficients, grid_mean, grid_modes)
    adaptive_grids = reconstruct_grid(adaptive_coefficients, grid_mean, grid_modes)

    # 8. Three representative held-out contour figures.
    for case_id in selected_ids:
        index = int(np.where(validation_ids == case_id)[0][0])
        contour_figure(case_id, metadata[index], reference_grids[index], adaptive_grids[index],
                       valid_mask, x, z, FIGURES / f"adaptive_contours_{case_id}.png")

    # 9. Wake profiles for high-incidence V06.
    case_index = int(np.where(validation_ids == "V06")[0][0])
    stations = (0.5, 1.0, 1.5, 2.0)
    z_window = (z >= -0.3) & (z <= 0.3)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), sharey=True, constrained_layout=True)
    for column, station in enumerate(stations):
        x_index = int(np.argmin(np.abs(x - station)))
        for row, field in enumerate((0, 1)):
            ax = axes[row, column]
            ax.plot(reference_grids[case_index, z_window, x_index, field], z[z_window], "k-", label="CFD")
            ax.plot(fixed_grids[case_index, z_window, x_index, field], z[z_window], "--", label="fixed")
            ax.plot(adaptive_grids[case_index, z_window, x_index, field], z[z_window], "-.", label="adaptive")
            ax.set_title(f"x/c={x[x_index]:.2f}")
            ax.set_xlabel(r"$u/U_\infty$" if row == 0 else r"$w/U_\infty$")
            ax.grid(alpha=0.25)
            if column == 0:
                ax.set_ylabel("z/c")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("V06 wake profiles: CFD versus POD–PINNs")
    fig.savefig(FIGURES / "wake_profiles_V06.png", dpi=180)
    plt.close(fig)

    # 10. Spatial PDE residual contours for V06.
    fixed_residual = grid_residuals(fixed_grids[case_index], x, z, parameters[case_index, 0])
    adaptive_residual = grid_residuals(adaptive_grids[case_index], x, z, parameters[case_index, 0])
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), sharex=True, sharey=True, constrained_layout=True)
    extent = (x[0], x[-1], z[0], z[-1])
    for column, name in enumerate(("Continuity", "x-momentum", "z-momentum")):
        combined = np.concatenate((np.abs(fixed_residual[column][physics_mask]),
                                   np.abs(adaptive_residual[column][physics_mask])))
        limit = max(float(np.percentile(combined, 99)), 1e-12)
        for row, residual in enumerate((fixed_residual[column], adaptive_residual[column])):
            values = residual.copy()
            values[~physics_mask] = np.nan
            image = axes[row, column].imshow(values, origin="lower", extent=extent, aspect="auto",
                                             vmin=-limit, vmax=limit, cmap="coolwarm")
            fig.colorbar(image, ax=axes[row, column], fraction=0.03, pad=0.02)
            axes[row, column].set_title(name)
            axes[row, column].set_xlabel("x/c")
        axes[0, column].set_ylabel("Fixed\nz/c")
        axes[1, column].set_ylabel("Adaptive\nz/c")
    fig.suptitle("V06 spatial physics residuals (99th-percentile colour limits)")
    fig.savefig(FIGURES / "spatial_physics_residuals_V06.png", dpi=180)
    plt.close(fig)

    # 11. Loss components and actual weighted contributions.
    fixed_history = json.loads((MODELS / "pod_pinn_2x64_rank25_history.json").read_text())
    adaptive_history = json.loads((MODELS / "pod_pinn_2x64_rank25_self_adaptive_history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for name in ("data", "continuity", "momentum", "boundary"):
        axes[0].semilogy([row["closure"] for row in adaptive_history], [row[name] for row in adaptive_history], label=name)
        axes[1].semilogy([row["closure"] for row in adaptive_history],
                         [row[name]*row[f"weight_{name}"] for row in adaptive_history], label=name)
    axes[0].set_title("Unweighted adaptive component losses")
    axes[1].set_title("Actual weighted contributions $\\lambda_i L_i$")
    for ax in axes:
        ax.set_xlabel("L-BFGS closure evaluation")
        ax.set_ylabel("Loss contribution")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "adaptive_loss_contributions.png", dpi=180)
    plt.close(fig)

    summary = {
        "figures": sorted(path.name for path in FIGURES.glob("*.png")),
        "validation_cases": validation_ids.tolist(),
        "selected_contour_cases": list(selected_ids),
        "wake_case": "V06",
        "wake_stations_x_over_c": list(stations),
        "fixed_mean_case_field_error": float(np.mean(fixed_errors)),
        "adaptive_mean_case_field_error": float(np.mean(adaptive_errors)),
        "pressure_force_mae": {
            "fixed_Cl": float(np.mean(np.abs(fixed_forces[:, 0] - true_forces[:, 0]))),
            "fixed_Cd": float(np.mean(np.abs(fixed_forces[:, 1] - true_forces[:, 1]))),
            "adaptive_Cl": float(np.mean(np.abs(adaptive_forces[:, 0] - true_forces[:, 0]))),
            "adaptive_Cd": float(np.mean(np.abs(adaptive_forces[:, 1] - true_forces[:, 1]))),
        },
    }
    (OUT / "extended_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
