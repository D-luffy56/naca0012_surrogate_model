from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
MODELS = ROOT / "artifacts" / "models"
OUT = ROOT / "artifacts" / "self_adaptive_results"
FIGURES = OUT / "figures"
FIELDS = ("u", "w", "p", "nut")


def load_case(case_id: str) -> tuple[np.ndarray, dict]:
    with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
        fields = archive["fields"][:, :4].astype(np.float64)
        metadata = json.loads(archive["metadata"].item())
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    fields /= np.array((u_inf, u_inf, u_inf**2, u_inf * chord))
    return fields, metadata


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fixed = json.loads((MODELS / "pod_pinn_2x64_rank25_metrics.json").read_text())
    adaptive = json.loads(
        (MODELS / "pod_pinn_2x64_rank25_self_adaptive_metrics.json").read_text()
    )
    history = json.loads(
        (MODELS / "pod_pinn_2x64_rank25_self_adaptive_history.json").read_text()
    )

    closures = np.array([record["closure"] for record in history])
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for name in ("data", "continuity", "momentum", "boundary"):
        ax.semilogy(
            closures,
            [record[f"weight_{name}"] for record in history],
            label=name,
        )
    ax.axhline(np.exp(5.0), color="black", linestyle="--", linewidth=1, label="stability ceiling")
    ax.set_xlabel("L-BFGS closure evaluation")
    ax.set_ylabel("Learned positive loss weight")
    ax.set_title("Self-adaptive loss-weight evolution")
    ax.grid(alpha=0.25)
    ax.legend(ncols=3, frameon=False)
    fig.savefig(FIGURES / "adaptive_weight_evolution.png", dpi=180)
    plt.close(fig)

    fixed_errors = 100.0 * np.array(
        [fixed["validation_field_relative_l2"][field] for field in FIELDS]
    )
    adaptive_errors = 100.0 * np.array(
        [adaptive["validation_field_relative_l2"][field] for field in FIELDS]
    )
    positions = np.arange(len(FIELDS))
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.bar(positions - 0.19, fixed_errors, width=0.38, label="Fixed 1:1:1:1")
    ax.bar(positions + 0.19, adaptive_errors, width=0.38, label="Self-adaptive")
    ax.set_xticks(positions, FIELDS)
    ax.set_ylabel("Validation relative $L_2$ error [%]")
    ax.set_title("64×64 POD–PINN: fixed vs self-adaptive weighting")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "fixed_vs_adaptive_field_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for name in ("data", "continuity", "momentum", "boundary"):
        ax.semilogy(closures, [record[name] for record in history], label=name)
    ax.set_xlabel("L-BFGS closure evaluation")
    ax.set_ylabel("Unweighted component loss")
    ax.set_title("Self-adaptive model component losses")
    ax.grid(alpha=0.25)
    ax.legend(ncols=4, frameon=False)
    fig.savefig(FIGURES / "adaptive_component_losses.png", dpi=180)
    plt.close(fig)

    with np.load(ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz", allow_pickle=False) as pod:
        native_xz = pod["xz"].astype(np.float64)
        validation_ids = pod["validation_ids"].astype(str)
    with np.load(
        ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz", allow_pickle=False
    ) as operator:
        x = operator["x"].astype(np.float64)
        z = operator["z"].astype(np.float64)
        valid_mask = operator["valid_mask"].astype(bool)
        grid_mean = operator["grid_mean"].astype(np.float64)
        grid_modes = operator["grid_modes"].astype(np.float64)
    with np.load(
        MODELS / "pod_pinn_2x64_rank25_self_adaptive_predictions.npz", allow_pickle=False
    ) as predictions:
        coefficients = predictions["validation_coefficients"].astype(np.float64)

    metadata_list = [load_case(case_id)[1] for case_id in validation_ids]
    plot_index = int(np.argmax([metadata["AoA_deg"] for metadata in metadata_list]))
    plot_id = validation_ids[plot_index]
    reference_native, metadata = load_case(plot_id)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((xx.ravel(), zz.ravel()))
    reference_grid = LinearNDInterpolator(
        Delaunay(native_xz), reference_native, fill_value=np.nan
    )(query).reshape(len(z), len(x), 4)
    prediction_grid = grid_mean + np.einsum("r,rzxf->zxf", coefficients[plot_index], grid_modes)
    reference_grid[~valid_mask] = np.nan
    prediction_grid[~valid_mask] = np.nan

    labels = (r"$u/U_\infty$", r"$w/U_\infty$", r"$p/U_\infty^2$", r"$\nu_t/(U_\infty c)$")
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 10.0), sharex=True, sharey=True, constrained_layout=True)
    extent = (x[0], x[-1], z[0], z[-1])
    for field_index, label in enumerate(labels):
        reference = reference_grid[:, :, field_index]
        prediction = prediction_grid[:, :, field_index]
        error = prediction - reference
        lower, upper = np.nanpercentile(
            np.concatenate((reference.ravel(), prediction.ravel())), (1, 99)
        )
        error_limit = np.nanpercentile(np.abs(error), 99)
        images = (
            axes[field_index, 0].imshow(reference, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[field_index, 1].imshow(prediction, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper, cmap="viridis"),
            axes[field_index, 2].imshow(error, origin="lower", extent=extent, aspect="auto", vmin=-error_limit, vmax=error_limit, cmap="coolwarm"),
        )
        for column, image in enumerate(images):
            fig.colorbar(image, ax=axes[field_index, column], fraction=0.025, pad=0.02)
            axes[field_index, column].set_ylabel(f"z/c\n{label}")
    for column, title in enumerate(("CFD / Delaunay", "Self-adaptive POD–PINN", "Prediction − CFD")):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("x/c")
    fig.suptitle(
        f"Held-out {plot_id}: Re={metadata['Re']:.2g}, AoA={metadata['AoA_deg']}°",
        fontsize=13,
    )
    fig.savefig(FIGURES / "self_adaptive_heldout_contours.png", dpi=180)
    plt.close(fig)

    fixed_mean = float(np.mean(list(fixed["validation_field_relative_l2"].values())))
    adaptive_mean = float(np.mean(list(adaptive["validation_field_relative_l2"].values())))
    comparison = {
        "fixed_64x64": fixed,
        "self_adaptive_64x64": adaptive,
        "mean_validation_field_relative_l2": {
            "fixed": fixed_mean,
            "self_adaptive": adaptive_mean,
            "relative_improvement": (fixed_mean - adaptive_mean) / fixed_mean,
        },
        "caution": (
            "Continuity, momentum, and boundary weights reached the imposed exp(5) "
            "stability ceiling; the result is a bounded adaptive solution, not an "
            "unconstrained optimum."
        ),
    }
    (OUT / "self_adaptive_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison["mean_validation_field_relative_l2"], indent=2))


if __name__ == "__main__":
    main()
