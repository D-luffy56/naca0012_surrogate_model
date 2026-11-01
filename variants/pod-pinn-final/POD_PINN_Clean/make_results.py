from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
POD_FILE = ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz"
OPERATOR_FILE = ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz"
MODEL_DIR = ROOT / "artifacts" / "models"
RESULTS_DIR = ROOT / "artifacts" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
WIDTHS = (32, 64, 128)
FIELD_NAMES = ("u", "w", "p", "nut")


def load_native(case_id: str) -> tuple[np.ndarray, dict]:
    with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
        fields = archive["fields"][:, :4].astype(np.float64)
        metadata = json.loads(archive["metadata"].item())
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    fields /= np.array((u_inf, u_inf, u_inf**2, u_inf * chord))
    return fields, metadata


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for width in WIDTHS:
        path = MODEL_DIR / f"pod_pinn_2x{width}_rank25_metrics.json"
        values = json.loads(path.read_text())
        # Correct the sign convention in already-written pressure-only coefficients.
        for case_values in values["validation_pressure_coefficients"].values():
            for key in tuple(case_values):
                case_values[key] *= -1.0
        path.write_text(json.dumps(values, indent=2) + "\n")
        metrics[width] = values

    comparison_rows = []
    for width, values in metrics.items():
        errors = values["validation_field_relative_l2"]
        comparison_rows.append(
            {
                "width": width,
                **{f"validation_{field}_relative_l2": errors[field] for field in FIELD_NAMES},
                "validation_mean_field_relative_l2": float(np.mean(list(errors.values()))),
                "Cl_pressure_MAE": values["validation_pressure_coefficient_mae"]["Cl_pressure"],
                "Cd_pressure_MAE": values["validation_pressure_coefficient_mae"]["Cd_pressure"],
                "elapsed_seconds": values["elapsed_seconds"],
            }
        )
    best_width = min(comparison_rows, key=lambda row: row["validation_mean_field_relative_l2"])[
        "width"
    ]
    with (RESULTS_DIR / "architecture_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    field_errors = np.array(
        [[metrics[width]["validation_field_relative_l2"][field] for field in FIELD_NAMES] for width in WIDTHS]
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    positions = np.arange(len(WIDTHS))
    bar_width = 0.19
    for index, field in enumerate(FIELD_NAMES):
        ax.bar(
            positions + (index - 1.5) * bar_width,
            100.0 * field_errors[:, index],
            width=bar_width,
            label=field,
        )
    ax.set_xticks(positions, [f"{width}×{width}" for width in WIDTHS])
    ax.set_ylabel("Validation relative $L_2$ error [%]")
    ax.set_xlabel("Two-hidden-layer architecture")
    ax.set_title("Fixed POD–PINN architecture comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, frameon=False)
    fig.savefig(FIGURES_DIR / "architecture_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for width in WIDTHS:
        history = json.loads(
            (MODEL_DIR / f"pod_pinn_2x{width}_rank25_history.json").read_text()
        )
        ax.semilogy(
            [record["closure"] for record in history],
            [record["total"] for record in history],
            label=f"{width}×{width}",
        )
    ax.set_xlabel("L-BFGS closure evaluation")
    ax.set_ylabel("Total objective")
    ax.set_title("Training convergence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "training_convergence.png", dpi=180)
    plt.close(fig)

    with np.load(POD_FILE, allow_pickle=False) as pod:
        native_xz = pod["xz"].astype(np.float64)
        weights = pod["weights"].astype(np.float64)
        mean = pod["mean"].astype(np.float64)
        field_rms = pod["field_rms"].astype(np.float64)
        basis = pod["basis"].astype(np.float64)
        validation_ids = pod["validation_ids"].astype(str)
    with np.load(OPERATOR_FILE, allow_pickle=False) as operator:
        x = operator["x"].astype(np.float64)
        z = operator["z"].astype(np.float64)
        valid_mask = operator["valid_mask"].astype(bool)
        grid_mean = operator["grid_mean"].astype(np.float64)
        grid_modes = operator["grid_modes"].astype(np.float64)
    with np.load(
        MODEL_DIR / f"pod_pinn_2x{best_width}_rank25_predictions.npz", allow_pickle=False
    ) as predictions:
        predicted_coefficients = predictions["validation_coefficients"].astype(np.float64)

    sqrt_weights = np.sqrt(weights)
    predicted_native = mean[None, :, :] + (
        (predicted_coefficients @ basis).reshape(len(validation_ids), len(weights), 4)
        / sqrt_weights[None, :, None]
    ) * field_rms[None, None, :]
    references = np.stack([load_native(case_id)[0] for case_id in validation_ids])
    case_rows = []
    for case_index, case_id in enumerate(validation_ids):
        metadata = load_native(case_id)[1]
        numerator = np.sum(
            weights[:, None] * (predicted_native[case_index] - references[case_index]) ** 2,
            axis=0,
        )
        denominator = np.sum(weights[:, None] * references[case_index] ** 2, axis=0)
        errors = np.sqrt(numerator / np.maximum(denominator, 1e-30))
        case_rows.append(
            {
                "case_id": case_id,
                "Re": metadata["Re"],
                "AoA_deg": metadata["AoA_deg"],
                **{f"{field}_relative_l2": float(value) for field, value in zip(FIELD_NAMES, errors)},
            }
        )
    with (RESULTS_DIR / "best_model_validation_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_rows[0].keys())
        writer.writeheader()
        writer.writerows(case_rows)

    # Show a representative high-incidence held-out case.
    plot_index = int(np.argmax([row["AoA_deg"] for row in case_rows]))
    plot_id = validation_ids[plot_index]
    reference_native, plot_metadata = load_native(plot_id)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((xx.ravel(), zz.ravel()))
    reference_grid = LinearNDInterpolator(
        Delaunay(native_xz), reference_native, fill_value=np.nan
    )(query).reshape(len(z), len(x), 4)
    prediction_grid = grid_mean + np.einsum(
        "r,rzxf->zxf", predicted_coefficients[plot_index], grid_modes
    )
    reference_grid[~valid_mask] = np.nan
    prediction_grid[~valid_mask] = np.nan

    labels = (r"$u/U_\infty$", r"$w/U_\infty$", r"$p/U_\infty^2$", r"$\nu_t/(U_\infty c)$")
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 10.0), sharex=True, sharey=True, constrained_layout=True)
    extent = (x[0], x[-1], z[0], z[-1])
    for field_index, label in enumerate(labels):
        reference = reference_grid[:, :, field_index]
        prediction = prediction_grid[:, :, field_index]
        error = prediction - reference
        lower, upper = np.nanpercentile(np.concatenate((reference.ravel(), prediction.ravel())), (1, 99))
        error_limit = np.nanpercentile(np.abs(error), 99)
        images = (
            axes[field_index, 0].imshow(reference, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[field_index, 1].imshow(prediction, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[field_index, 2].imshow(error, origin="lower", extent=extent, aspect="auto", vmin=-error_limit, vmax=error_limit, cmap="coolwarm"),
        )
        for column, image in enumerate(images):
            fig.colorbar(image, ax=axes[field_index, column], fraction=0.025, pad=0.02)
            axes[field_index, column].set_ylabel(f"z/c\n{label}")
    for column, title in enumerate(("CFD / Delaunay", f"POD–PINN {best_width}×{best_width}", "Prediction − CFD")):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("x/c")
    fig.suptitle(
        f"Held-out {plot_id}: Re={plot_metadata['Re']:.2g}, AoA={plot_metadata['AoA_deg']}°",
        fontsize=13,
    )
    fig.savefig(FIGURES_DIR / "best_model_heldout_contours.png", dpi=180)
    plt.close(fig)

    summary = {
        "best_architecture": [2, best_width, best_width, 25],
        "selection_metric": "lowest mean validation relative L2 across u, w, p, nut",
        "architecture_comparison": comparison_rows,
        "best_model_metrics": metrics[best_width],
        "operator_validation": json.loads(
            (ROOT / "artifacts" / "operator" / "operator_validation.json").read_text()
        ),
        "pod_summary": json.loads(
            (ROOT / "artifacts" / "pod" / "pod_summary.json").read_text()
        ),
    }
    (RESULTS_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "best_width": best_width,
                "comparison": comparison_rows,
                "representative_case": plot_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
