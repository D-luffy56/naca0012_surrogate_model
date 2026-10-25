"""Reproduce POD-GPR validation and create presentation-ready tables and chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from evaluate_fresh_openfoam_tests import basis_data, scale_inputs_like_sklearn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gpr_validation"
SNAPSHOTS = ROOT / "outputs/native_sst/native_sst_snapshots.npz"
FRESH = ROOT / "outputs/fresh_test_cfd_v2"
FIELDS = ("u", "w", "p", "k", "omega", "nut")


def error(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {field: float(np.linalg.norm(prediction[:, i, mask] - truth[:, i, mask]) /
                         max(np.linalg.norm(truth[:, i, mask]), 1e-12))
            for i, field in enumerate(FIELDS)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SNAPSHOTS) as saved:
        raw = saved["fields"].copy()
        speed = saved["uinf_mps"].copy()
        reynolds = saved["reynolds"].copy()
        aoa = saved["aoa_deg"].copy()
        ids = saved["ids"].copy()
        points = saved["points_xz"].copy()
        wall_distance = saved["wall_distance"].copy()
        count = int(saved["train_count"])
    if count != 40 or len(raw) != 47:
        raise ValueError("Expected 40 training and seven original validation cases")
    units = np.stack((speed, speed, speed**2, speed**2, speed, speed), axis=1)
    q = raw / units[:, :, None]
    basis = basis_data(ROOT / "outputs/sst_pod_model")
    scaled = (q[:count] / basis["scale"][None, :, None]).reshape(count, -1)
    coefficients = (scaled - basis["mean"]) @ basis["modes"].T
    y = coefficients.astype(np.float32, copy=True)
    y -= basis["coefficient_mean"]
    y /= basis["coefficient_scale"]
    conditions = np.column_stack((reynolds, aoa))
    x_train = scale_inputs_like_sklearn(conditions[:count], basis)
    x_validation = scale_inputs_like_sklearn(conditions[count:], basis)
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
               Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)),
        alpha=1e-8, normalize_y=False, n_restarts_optimizer=1, random_state=42,
    ).fit(x_train, y)
    predicted_coef = model.predict(x_validation) * basis["coefficient_scale"] + basis["coefficient_mean"]
    prediction = ((basis["mean"] + predicted_coef @ basis["modes"])
                  .reshape(7, 6, -1) * basis["scale"][None, :, None])
    validation_truth = q[count:]
    regions = {
        "all": np.ones(len(points), dtype=bool),
        "near_wall_0_to_0.05_chord": wall_distance <= 0.05,
        "wake_1_to_3_chord": (points[:, 0] >= 1) & (points[:, 0] <= 3)
                             & (np.abs(points[:, 1]) <= 0.5),
    }
    aggregate = {region: error(validation_truth, prediction, mask)
                 for region, mask in regions.items()}
    rows = []
    for i, case_id in enumerate(ids[count:]):
        for region, mask in regions.items():
            rows.append({"id": str(case_id), "Re": float(reynolds[count+i]),
                         "AoA_deg": float(aoa[count+i]), "region": region,
                         **error(validation_truth[i:i+1], prediction[i:i+1], mask)})
    per_case = pd.DataFrame(rows)
    per_case.to_csv(OUT / "original_validation_by_case.csv", index=False)

    fresh = pd.read_csv(FRESH / "comparison_by_case.csv")
    fresh = fresh[fresh["model"].isin(("gpr_matern", "mlp_64_data"))].copy()
    if set(fresh["id"]) != {f"N{i:02d}" for i in range(1, 8)}:
        raise ValueError("Fresh comparison is missing cases")
    fresh.to_csv(OUT / "fresh_comparison_by_case.csv", index=False)
    fresh_summary = json.loads((FRESH / "comparison_summary.json").read_text())
    if not all(row["residual_control_converged"] for row in fresh_summary["quality"]):
        raise ValueError("Fresh CFD solver residuals did not converge")

    # Visualize case-level variation rather than only the pooled seven-case score.
    subset = fresh[fresh["region"] == "all"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    case_ids = [f"N{i:02d}" for i in range(1, 8)]
    positions = np.arange(len(case_ids))
    for ax, field in zip(axes, ("u", "w", "p")):
        gpr_values = subset[subset["model"] == "gpr_matern"].set_index("id").loc[case_ids, field].to_numpy() * 100
        mlp_values = subset[subset["model"] == "mlp_64_data"].set_index("id").loc[case_ids, field].to_numpy() * 100
        ax.bar(positions - 0.18, gpr_values, width=0.36, label="POD-GPR", color="#176B8A")
        ax.bar(positions + 0.18, mlp_values, width=0.36, label="POD-MLP 64-64", color="#D68C32")
        ax.set_ylabel(f"{field} L2 error (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xticks(positions, case_ids)
    axes[-1].set_xlabel("Fresh CFD case")
    fig.suptitle("Native-mesh field error by fresh CFD case")
    fig.savefig(OUT / "fresh_case_field_errors.png", dpi=180)
    plt.close(fig)

    summary = {
        "configuration": {
            "training_cases": 40, "original_validation_cases": 7,
            "fresh_cfd_cases": 7, "fields": FIELDS, "pod_modes": 16,
            "native_mesh_points": len(points), "kernel_fitted": str(model.kernel_),
            "inputs": ["Re", "AoA_deg"],
            "preprocessing": "Case-specific Uinf normalization; field scaling; POD and input/coefficient scaling fitted on training cases only",
            "pressure_convention": "OpenFOAM kinematic pressure p/rho; nondimensionalized by Uinf squared",
        },
        "original_validation_aggregate_relative_l2": aggregate,
        "fresh_aggregate_relative_l2": {
            model_name: fresh_summary["aggregate_relative_l2"][model_name]
            for model_name in ("gpr_matern", "mlp_64_data")
        },
        "fresh_quality": fresh_summary["quality"],
        "limitations": [
            "Fresh CFD cases met solver residual control, but all failed the stricter last-50-force-sample stability criterion.",
            "The fresh comparison's data-only MLP was refitted and did not exactly reproduce the older saved baseline predictions.",
            "The seven fresh cases have already been inspected and must not be reused as an untouched selection set.",
            "GPR is data-only; it has no SST physics loss or wall condition enforcement.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# POD-GPR validation for the AE646 digital twin", "",
        "## Model and protocol", "",
        "The model maps Reynolds number and angle of attack to 16 POD coefficients using a Matérn 5/2 Gaussian process. "
        "The coefficients reconstruct six native-mesh fields: u, w, kinematic pressure, k, omega, and eddy viscosity. "
        "It was fitted on 40 steady incompressible k-omega SST RANS CFD cases. The POD basis and scalers use only those 40 cases.", "",
        "The original seven validation cases and seven later fresh CFD cases are reported separately. "
        "Relative L2 is the norm of prediction minus CFD divided by the norm of CFD, pooled over the named cases and mesh points. "
        "These are field errors, not classification accuracy. Pressure is OpenFOAM kinematic pressure.", "",
        "## Aggregate field error", "",
        "| Case group and model | u | w | pressure |", "| --- | ---: | ---: | ---: |",
    ]
    for label, values in (
        ("Original validation (7), POD-GPR", aggregate["all"]),
        ("Fresh CFD (7), POD-GPR", fresh_summary["aggregate_relative_l2"]["gpr_matern"]["all"]),
        ("Fresh CFD (7), data-only POD-MLP 64-64", fresh_summary["aggregate_relative_l2"]["mlp_64_data"]["all"]),
    ):
        lines.append(f"| {label} | {values['u']:.3%} | {values['w']:.3%} | {values['p']:.3%} |")
    lines += ["", "## GPR error for each fresh CFD case", "",
              "| Case | Re | AoA | u | w | pressure |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in fresh[(fresh["model"] == "gpr_matern") & (fresh["region"] == "all")].itertuples():
        lines.append(f"| {row.id} | {row.Re:.0f} | {row.AoA_deg:g}° | {row.u:.3%} | {row.w:.3%} | {row.p:.3%} |")
    lines += ["", "## GPR error for each original validation case", "",
              "| Case | Re | AoA | u | w | pressure |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_case[per_case["region"] == "all"].itertuples():
        lines.append(f"| {row.id} | {row.Re:.0f} | {row.AoA_deg:g}° | {row.u:.3%} | {row.w:.3%} | {row.p:.3%} |")
    lines += ["", "## Interpretation", "",
              "On the seven fresh CFD cases pooled together, GPR had lower u and pressure error; "
              "the data-only MLP had lower w error. Individual cases can have different winners. "
              "The GPR has no physics loss; this comparison measures prediction against CFD, not equation residuals.", "",
              "The 40-case five-fold cross-validation found the MLP better overall than GPR when each fold trained on 32 cases. "
              "The full-40 fresh-test ranking therefore should not be presented as a universal GPR advantage.", "",
              "## Limits", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += ["", "## Presentation assets", "",
              "- `fresh_case_field_errors.png`: per-case u, w, and pressure error comparison.",
              "- `original_validation_by_case.csv`: GPR errors for seven original validation cases, including wall and wake regions.",
              "- `fresh_comparison_by_case.csv`: GPR and data-only MLP errors for seven fresh CFD cases and three regions.",
              "- `summary.json`: machine-readable configuration and aggregate results.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "original_validation": aggregate["all"],
                      "fresh_gpr": fresh_summary["aggregate_relative_l2"]["gpr_matern"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
