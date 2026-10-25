"""Five-fold development-only comparison of native six-field POD regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


FIELDS = ("u", "w", "p", "k", "omega", "nut")


def pod_fit(q, train_indices, rank):
    scale = np.maximum(q[train_indices].std(axis=(0, 2)), 1e-7)
    scaled = (q / scale[None, :, None]).reshape(len(q), -1)
    mean = scaled[train_indices].mean(axis=0)
    centered = scaled[train_indices] - mean
    gram = centered @ centered.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    rank = min(rank, len(train_indices) - 1)
    modes = (vectors[:, order[:rank]].T @ centered) / np.sqrt(
        np.maximum(values[order[:rank]], 1e-10))[:, None]
    coefficients = (scaled - mean) @ modes.T
    return mean, modes, scale, coefficients


def reconstruct(coefficients, mean, modes, scale, points):
    return (mean + coefficients @ modes).reshape(-1, 6, points) * scale[None, :, None]


def relative_l2(truth, estimate, mask):
    return {field: float(np.linalg.norm(estimate[:, index, mask] - truth[:, index, mask]) /
                         max(np.linalg.norm(truth[:, index, mask]), 1e-12))
            for index, field in enumerate(FIELDS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()
    with np.load(args.snapshots) as saved:
        raw = saved["fields"].copy()
        speed = saved["uinf_mps"].copy()
        reynolds = saved["reynolds"].copy()
        angle = saved["aoa_deg"].copy()
        points = saved["points_xz"].copy()
        wall_distance = saved["wall_distance"].copy()
        count = int(saved["train_count"])
        case_ids = saved["ids"].copy()
    if count != 40 or len(raw) != 47:
        raise ValueError("Expected 40 development and seven previously inspected cases")
    if args.modes < 1:
        raise ValueError("POD rank must be positive")
    units = np.stack((speed, speed, speed**2, speed**2, speed, speed), axis=1)
    q = raw / units[:, :, None]
    x = np.column_stack((reynolds, angle))
    regions = {
        "all": np.ones(len(points), bool),
        "near_wall_0_to_0.05_chord": wall_distance <= 0.05,
        "wake_1_to_3_chord": (points[:, 0] >= 1) & (points[:, 0] <= 3)
                                  & (np.abs(points[:, 1]) <= 0.5),
    }
    rows = []
    oof = {"gpr": np.zeros_like(q[:count]), "mlp": np.zeros_like(q[:count]),
           "pod_projection": np.zeros_like(q[:count])}
    for fold, (train, validation) in enumerate(KFold(n_splits=5, shuffle=True,
                                                      random_state=42).split(np.arange(count)), 1):
        mean, modes, field_scale, coefficients = pod_fit(q, train, args.modes)
        x_scaler = StandardScaler().fit(x[train])
        y_scaler = StandardScaler().fit(coefficients[train])
        x_train, x_val = x_scaler.transform(x[train]), x_scaler.transform(x[validation])
        y_train = y_scaler.transform(coefficients[train])
        oof["pod_projection"][validation] = reconstruct(
            coefficients[validation], mean, modes, field_scale, q.shape[2])
        models = {
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                                solver="lbfgs", max_iter=5000, random_state=42),
            "gpr": GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
                       Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                       + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)),
                alpha=1e-8, normalize_y=False, n_restarts_optimizer=1,
                random_state=42),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            prediction = reconstruct(y_scaler.inverse_transform(model.predict(x_val)),
                                     mean, modes, field_scale, q.shape[2])
            oof[name][validation] = prediction
            for region, mask in regions.items():
                errors = relative_l2(q[validation], prediction, mask)
                for field, error in errors.items():
                    rows.append({"fold": fold, "model": name, "region": region,
                                 "field": field, "relative_l2": error})
        print(f"Completed development fold {fold}/5", flush=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "fold_errors.csv", index=False)
    summary = {
        "protocol": "Five shuffled folds among 40 development cases; POD and scalers refit on each fold's 32 training cases; seven existing validation cases excluded",
        "modes": args.modes,
        "region_points": {name: int(mask.sum()) for name, mask in regions.items()},
        "out_of_fold_relative_l2": {
            model: {region: relative_l2(q[:count], estimate, mask)
                    for region, mask in regions.items()}
            for model, estimate in oof.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
