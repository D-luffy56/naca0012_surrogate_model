"""Write a compact comparison for all seven held-out native-mesh cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path)
    args = parser.parse_args()
    model = args.model_dir
    summary = json.loads((model / "summary.json").read_text())
    cases = pd.read_csv(model / "validation_by_case.csv")
    if len(cases) != 7:
        raise ValueError("The report requires all seven held-out cases")
    names = ("u", "w", "p", "k", "omega", "nut")
    lines = [
        "# Native-mesh SST POD validation",
        "",
        f"Training cases: {summary['training_cases']}; validation cases: {summary['validation_cases']}; "
        f"native mesh points: {summary['native_points']}; airfoil-wall points: {summary['wall_points']}.",
        "",
        "The baseline and SST-trained models use the same six-field POD basis and network size.",
        "Relative L2 errors below are aggregated over the seven validation cases.",
        "",
        "| Field | Data-only baseline | SST-regularized | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        baseline = summary["validation_baseline_relative_l2"][name]
        sst = summary["validation_sst_relative_l2"][name]
        lines.append(f"| {name} | {baseline:.3%} | {sst:.3%} | {(sst-baseline):+.3%} |")
    common_grid = model / "legacy_grid_comparison.json"
    if common_grid.is_file():
        comparison = json.loads(common_grid.read_text())
        lines += ["", "## Comparison on the original baseline grid", "",
                  "All three predictions below use the same held-out CFD values "
                  "and the original 192 × 128 sampling grid.", "",
                  "| Field | Original 3-field baseline | Native 6-field data-only | Native SST-regularized |",
                  "| --- | ---: | ---: | ---: |"]
        for name in ("u", "w", "p"):
            lines.append(f"| {name} | {comparison['legacy_original_grid'][name]:.3%} | "
                         f"{comparison['native_data_only_on_original_grid'][name]:.3%} | "
                         f"{comparison['native_sst_on_original_grid'][name]:.3%} |")
    if args.snapshots is not None:
        with np.load(args.snapshots) as saved, np.load(model / "validation_predictions.npz") as prediction, \
             np.load(model / "sst_pod_basis.npz") as basis:
            start = int(saved["train_count"])
            truth = saved["wall_fields"][start:]
            speed = saved["uinf_mps"][start:]
            units = np.stack((speed, speed, speed**2, speed**2, speed, speed), axis=1)
            truth = truth / units[:, :, None]
            wall_index = saved["wall_index"]
            lines += ["", "## Airfoil wall errors", "",
                      "Wall RMSE divided by each field's training standard deviation, "
                      "against the separate CFD boundary exports:", "",
                      "| Field | Data-only baseline | SST-regularized |",
                      "| --- | ---: | ---: |"]
            for field, name in enumerate(names):
                baseline_error = np.sqrt(np.mean((
                    prediction["baseline"][:, field][:, wall_index] - truth[:, field])**2)) / basis["scale"][field]
                sst_error = np.sqrt(np.mean((
                    prediction["sst"][:, field][:, wall_index] - truth[:, field])**2)) / basis["scale"][field]
                lines.append(f"| {name} | {baseline_error:.4g} | {sst_error:.4g} |")
    lines += ["", "## Per-case field errors", "",
              "| Case | Re | AoA | Baseline u | SST u | Baseline w | SST w | Baseline p | SST p |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in cases.itertuples():
        lines.append(f"| {row.id} | {row.Re:.0f} | {row.AoA_deg:g}° | "
                     f"{row.baseline_u_relative_l2:.3%} | {row.sst_u_relative_l2:.3%} | "
                     f"{row.baseline_w_relative_l2:.3%} | {row.sst_w_relative_l2:.3%} | "
                     f"{row.baseline_p_relative_l2:.3%} | {row.sst_p_relative_l2:.3%} |")
    lines += ["", "## Interpretation", "", summary["limitation"],
              "", "The original baseline and native-mesh models use different POD bases "
              "and state definitions. The common-grid comparison holds the evaluation "
              "points fixed, but differences may reflect the representation as well as training.", ""]
    (model / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(model / "validation_report.md")


if __name__ == "__main__":
    main()
