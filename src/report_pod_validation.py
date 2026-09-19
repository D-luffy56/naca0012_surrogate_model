"""Create POD/MLP validation contours and a CFD force-history review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIELDS = ("u", "w", "p")


def windows_path(value: str) -> Path:
    return Path("D:/" + value.removeprefix("/mnt/d/")) if value.startswith("/mnt/d/") else Path(value)


def restore_grid(values: np.ndarray, fluid_mask: np.ndarray) -> np.ndarray:
    grid = np.full(fluid_mask.size, np.nan, dtype=float)
    grid[fluid_mask.ravel()] = values
    return grid.reshape(fluid_mask.shape)


def read_force_history(case_path: Path) -> pd.DataFrame:
    path = case_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    return pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=("time", "Cd", "Cd_f", "Cd_r", "Cl", "Cl_f", "Cl_r", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs_f", "Cs_r"))


def relative_span(values: pd.Series) -> float:
    denominator = max(abs(float(values.mean())), 1e-4)
    return float((values.max() - values.min()) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.pod_dir / "reports"
    output.mkdir(exist_ok=True)

    basis = np.load(args.pod_dir / "pod_basis.npz")
    validation = np.load(args.pod_dir / "validation_snapshots.npz", allow_pickle=True)
    prediction = np.load(args.pod_dir / "validation_predictions.npz", allow_pickle=True)
    cases = pd.read_csv(args.pod_dir / "validation_cases.csv")
    x, z, mask = basis["x"], basis["z"], basis["fluid_mask"]
    truth, predicted = validation["fields"], prediction["fields"]

    for i, case_id in enumerate(validation["ids"]):
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        fig.suptitle(f"{case_id}: CFD vs POD-MLP reconstruction", fontsize=15)
        for field_index, field in enumerate(FIELDS):
            actual_grid = restore_grid(truth[i, field_index], mask)
            predicted_grid = restore_grid(predicted[i, field_index], mask)
            error_grid = predicted_grid - actual_grid
            bound = max(abs(np.nanmin(actual_grid)), abs(np.nanmax(actual_grid)), abs(np.nanmin(predicted_grid)), abs(np.nanmax(predicted_grid)))
            error_bound = max(abs(np.nanmin(error_grid)), abs(np.nanmax(error_grid)))
            for axis, data, title, cmap, vmin, vmax in (
                (axes[field_index, 0], actual_grid, "CFD", "viridis", -bound, bound),
                (axes[field_index, 1], predicted_grid, "POD-MLP", "viridis", -bound, bound),
                (axes[field_index, 2], error_grid, "Prediction − CFD", "coolwarm", -error_bound, error_bound),
            ):
                image = axis.pcolormesh(x, z, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_aspect("equal")
                axis.set_title(f"{field}: {title}")
                axis.set_xlabel("x/c")
                axis.set_ylabel("z/c")
                fig.colorbar(image, ax=axis, shrink=0.82)
        fig.savefig(output / f"validation_{case_id}.png", dpi=180)
        plt.close(fig)

    if "true_coefficients" in prediction.files:
        true_coefficients = prediction["true_coefficients"]
        predicted_coefficients = prediction["predicted_coefficients"]
        modes = true_coefficients.shape[1]
        fig, axes = plt.subplots(1, modes, figsize=(4 * modes, 3.8), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for mode, axis in enumerate(axes):
            values = np.r_[true_coefficients[:, mode], predicted_coefficients[:, mode]]
            lower, upper = values.min(), values.max()
            padding = max((upper - lower) * 0.08, 1e-6)
            axis.scatter(true_coefficients[:, mode], predicted_coefficients[:, mode], color="tab:blue")
            axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--", lw=1)
            axis.set_title(f"POD coefficient a{mode + 1}")
            axis.set_xlabel("CFD coefficient")
            axis.set_ylabel("MLP coefficient")
            axis.grid(alpha=0.3)
        fig.savefig(output / "pod_coefficients_true_vs_predicted.png", dpi=180)
        plt.close(fig)

    manifest = pd.read_csv(args.training_manifest)
    rows = []
    for _, case in manifest.iterrows():
        history = read_force_history(windows_path(case["case_path"]))
        tail = history.tail(min(20, len(history)))
        rows.append({
            "id": case["id"], "Re": case["Re"], "AoA_deg": case["AoA_deg"],
            "Cl_tail_mean": tail.Cl.mean(), "Cd_tail_mean": tail.Cd.mean(),
            "Cl_tail_relative_span": relative_span(tail.Cl),
            "Cd_tail_relative_span": relative_span(tail.Cd),
            "tail_samples": len(tail),
        })
    force = pd.DataFrame(rows)
    force["tail_stable_0p5pct"] = (force.Cl_tail_relative_span <= 0.005) & (force.Cd_tail_relative_span <= 0.005)
    force.to_csv(output / "force_tail_review.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for reynolds, group in force.groupby("Re"):
        axes[0].plot(group.AoA_deg, group.Cl_tail_relative_span * 100, marker="o", label=f"Re={reynolds / 1e6:g}M")
        axes[1].plot(group.AoA_deg, group.Cd_tail_relative_span * 100, marker="o", label=f"Re={reynolds / 1e6:g}M")
    for axis, label in zip(axes, ("Cl tail span / mean (%)", "Cd tail span / mean (%)")):
        axis.axhline(0.5, color="tab:red", ls="--", lw=1, label="0.5% review threshold")
        axis.set_xlabel("Angle of attack (deg)")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "force_tail_stability.png", dpi=180)
    print(f"Validation contour plots: {len(cases)}")
    print(f"Tail-stable cases at 0.5% span threshold: {force.tail_stable_0p5pct.sum()} / {len(force)}")


if __name__ == "__main__":
    main()
