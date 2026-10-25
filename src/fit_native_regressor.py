"""Refit a data-only native-mesh POD regressor from the saved 40-case snapshots.

The saved SST POD basis supplies the same 16 modes and scalers used in the
fresh-CFD comparison. This script does not use validation cases for fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import meshio
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor


FIELDS = ("u", "w", "p", "k", "omega", "nut")


def fit(snapshots: Path, basis_path: Path, kind: str, output: Path,
        fresh_dir: Path | None = None) -> None:
    with np.load(snapshots, allow_pickle=False) as data, np.load(basis_path, allow_pickle=False) as basis:
        ntrain = int(data["train_count"])
        if ntrain != 40 or len(data["fields"]) != 47:
            raise ValueError("Expected 40 training and seven validation cases")
        speed = data["uinf_mps"]
        units = np.stack((speed, speed, speed**2, speed**2, speed, speed), axis=1)
        truth = data["fields"] / units[:, :, None]
        conditions = np.column_stack((data["reynolds"], data["aoa_deg"]))
        ids = data["ids"][ntrain:].copy()
        points = data["points_xz"].copy()
        mean, modes, scale = basis["mean"].copy(), basis["modes"].copy(), basis["scale"].copy()
        x_mean, x_scale = basis["x_mean"].copy(), basis["x_scale"].copy()
        c_mean, c_scale = basis["coefficient_mean"].copy(), basis["coefficient_scale"].copy()

    state = (truth / scale[None, :, None]).reshape(47, -1)
    coefficients = (state - mean) @ modes.T
    x = np.asarray(conditions, dtype=np.float32).copy()
    x -= x_mean
    x /= x_scale
    y = coefficients[:ntrain].astype(np.float32, copy=True)
    y -= c_mean
    y /= c_scale
    if kind == "gpr":
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
                   Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                   + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)),
            alpha=1e-8, normalize_y=False, n_restarts_optimizer=1, random_state=42)
    else:
        hidden = (64, 64) if kind == "mlp64" else (32, 32)
        model = MLPRegressor(hidden_layer_sizes=hidden,
                             activation="relu" if kind == "mlp64" else "tanh",
                             solver="lbfgs", alpha=1e-4, max_iter=5000, random_state=42)
    model.fit(x[:ntrain], y)
    pred_coef = model.predict(x[ntrain:]) * c_scale + c_mean
    estimate = (mean + pred_coef @ modes).reshape(7, 6, -1) * scale[None, :, None]
    errors = {field: float(np.linalg.norm(estimate[:, i] - truth[ntrain:, i]) /
                           np.linalg.norm(truth[ntrain:, i]))
              for i, field in enumerate(FIELDS)}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "regressor.joblib")
    np.savez_compressed(output / "validation_predictions.npz", ids=ids, fields=estimate,
                        true_fields=truth[ntrain:])
    summary = {"kind": kind, "training_cases": 40, "validation_cases": 7,
               "representation": "native mesh, six nondimensional fields, 16 POD modes",
               "data_source": "native_sst_snapshots.npz; basis from saved SST training run",
               "fit_note": "Refitted estimator; original sklearn data-only estimator was not saved",
               "validation_relative_l2": errors}
    if kind == "gpr":
        summary["fitted_kernel"] = str(model.kernel_)
    if fresh_dir is not None:
        specs = pd.read_csv(fresh_dir / "frozen_cases.csv")
        fresh_truth = []
        for row in specs.itertuples():
            exports = list((fresh_dir / row.id / "VTK").rglob("internal.vtu"))
            if len(exports) != 1:
                raise ValueError(f"Expected one VTK export for {row.id}")
            mesh = meshio.read(exports[0])
            current, index = np.unique(mesh.points[:, [0, 2]], axis=0, return_index=True)
            if current.shape != points.shape or not np.allclose(current, points, atol=1e-7, rtol=0):
                raise ValueError(f"Native mesh differs for {row.id}")
            velocity = mesh.point_data["U"][index]
            fields = np.stack((velocity[:, 0], velocity[:, 2],
                               *(mesh.point_data[name][index] for name in ("p", "k", "omega", "nut"))))
            uinf = row.Re * 1.56e-5
            fresh_truth.append(fields / np.array((uinf, uinf, uinf**2, uinf**2, uinf, uinf))[:, None])
        fresh_truth = np.stack(fresh_truth)
        fresh_x = specs[["Re", "AoA_deg"]].to_numpy(dtype=np.float32)
        fresh_x -= x_mean
        fresh_x /= x_scale
        fresh_coef = model.predict(fresh_x) * c_scale + c_mean
        fresh_estimate = (mean + fresh_coef @ modes).reshape(len(specs), 6, -1) * scale[None, :, None]
        summary["fresh_relative_l2"] = {
            field: float(np.linalg.norm(fresh_estimate[:, i] - fresh_truth[:, i]) /
                         np.linalg.norm(fresh_truth[:, i]))
            for i, field in enumerate(FIELDS)}
        np.savez_compressed(output / "fresh_predictions.npz", ids=specs.id.to_numpy(dtype=str),
                            fields=fresh_estimate, true_fields=fresh_truth)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--kind", choices=("mlp64", "mlp32", "gpr"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fresh-dir", type=Path)
    a = p.parse_args()
    fit(a.snapshots, a.basis, a.kind, a.output_dir, a.fresh_dir)
