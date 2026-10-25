"""Compare a fixed SST POD-MLP weight sweep with data-only POD-GPR."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from build_sst_snapshots import extract_values
from train_sst_pod import predict_sst_case


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs/physics_weight_sweep"
SNAPSHOTS = ROOT / "outputs/native_sst/native_sst_snapshots.npz"
FRESH = ROOT / "outputs/fresh_test_cfd_v2"
FIELDS = ("u", "w", "p", "k", "omega", "nut")
WEIGHTS = ("w0", "w1e-5", "w1e-4", "w1e-3")


def errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {name: float(100 * np.linalg.norm(prediction[:, i] - truth[:, i]) /
                        max(np.linalg.norm(truth[:, i]), 1e-12))
            for i, name in enumerate(FIELDS)}


def main() -> None:
    with np.load(SNAPSHOTS) as saved:
        raw = saved["fields"].copy()
        ids = saved["ids"].copy()
        re = saved["reynolds"].copy()
        aoa = saved["aoa_deg"].copy()
        speed = saved["uinf_mps"].copy()
        points = saved["points_xz"].copy()
    units = np.stack((speed, speed, speed**2, speed**2, speed, speed), axis=1)
    q = raw / units[:, :, None]
    baseline_dir = SWEEP / "w0"
    with np.load(baseline_dir / "sst_pod_basis.npz") as basis:
        basis = {key: basis[key].copy() for key in basis.files}
    for label in WEIGHTS:
        with np.load(SWEEP / label / "sst_pod_basis.npz") as candidate:
            for key in ("mean", "modes", "scale", "x_mean", "x_scale",
                        "coefficient_mean", "coefficient_scale"):
                if not np.allclose(basis[key], candidate[key], rtol=1e-6, atol=1e-7):
                    raise ValueError(f"{label} uses a different POD representation: {key}")

    # Match the existing native-mesh Matérn GPR comparator.
    scaled = (q[:40] / basis["scale"][None, :, None]).reshape(40, -1)
    coefficient = (scaled - basis["mean"]) @ basis["modes"].T
    standardized_y = ((coefficient - basis["coefficient_mean"]) /
                      basis["coefficient_scale"]).astype(np.float32)
    conditions = np.column_stack((re, aoa)).astype(np.float32)
    standardized_x = (conditions - basis["x_mean"]) / basis["x_scale"]
    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
               Matern(length_scale=np.ones(2), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)),
        alpha=1e-8, normalize_y=False, n_restarts_optimizer=1, random_state=42,
    ).fit(standardized_x[:40], standardized_y)

    def gpr_predict(conditions_to_predict: np.ndarray) -> np.ndarray:
        x = (conditions_to_predict.astype(np.float32) - basis["x_mean"]) / basis["x_scale"]
        c = gpr.predict(x) * basis["coefficient_scale"] + basis["coefficient_mean"]
        return ((basis["mean"] + c @ basis["modes"]).reshape(len(x), 6, -1)
                * basis["scale"][None, :, None])

    validation = {"pod_gpr_data_only": gpr_predict(conditions[40:])}
    for label in WEIGHTS:
        with np.load(SWEEP / label / "validation_predictions.npz") as saved:
            if not np.array_equal(saved["ids"], ids[40:]):
                raise ValueError(f"{label}: validation case IDs differ")
            validation[label] = saved["sst"].copy()

    specs = pd.read_csv(FRESH / "frozen_cases.csv")
    fresh_conditions = specs[["Re", "AoA_deg"]].to_numpy()
    fresh_truth = []
    for row in specs.itertuples():
        exports = list((FRESH / row.id / "VTK").rglob("internal.vtu"))
        if len(exports) != 1:
            raise ValueError(f"{row.id}: expected one final CFD export")
        mesh = meshio.read(exports[0])
        native, unique_index = np.unique(mesh.points[:, [0, 2]], axis=0, return_index=True)
        if native.shape != points.shape or not np.allclose(native, points, rtol=0, atol=1e-7):
            raise ValueError(f"{row.id}: mesh differs")
        uinf = row.Re * 1.56e-5
        case_units = np.array((uinf, uinf, uinf**2, uinf**2, uinf, uinf))
        fresh_truth.append(extract_values(mesh.point_data, unique_index) / case_units[:, None])
    fresh_truth = np.stack(fresh_truth)
    fresh = {"pod_gpr_data_only": gpr_predict(fresh_conditions)}
    for label in WEIGHTS:
        case_fields = []
        for row in specs.itertuples():
            dimensional = predict_sst_case(SWEEP / label, row.Re, row.AoA_deg)["fields"]
            uinf = row.Re * 1.56e-5
            case_units = np.array((uinf, uinf, uinf**2, uinf**2, uinf, uinf))
            case_fields.append(dimensional / case_units[:, None])
        fresh[label] = np.stack(case_fields)

    rows = []
    for split, truth, predictions in (("original_validation_7", q[40:], validation),
                                      ("fresh_cfd_7", fresh_truth, fresh)):
        for model, prediction in predictions.items():
            row = {"split": split, "model": model, **errors(truth, prediction)}
            rows.append(row)
            print(row, flush=True)
    output = SWEEP / "comparison.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    (SWEEP / "comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
