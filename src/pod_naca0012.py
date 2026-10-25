"""Build a POD basis and MLP surrogate from fixed-geometry OpenFOAM VTK exports.

Inputs are the verified training/validation manifests produced by the CFD batch
runner. The CFD mesh is common to every case; fields are sampled onto a shared
2D x-z grid before POD. The solid NACA 0012 region is excluded from the state
vector, so no artificial values inside the airfoil influence the SVD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion
from scipy.spatial import Delaunay
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


FIELDS = ("u", "w", "p")  # OpenFOAM 2D plane is x-z; U=(u, v_thin, w).


def windows_path(value: str) -> Path:
    """Convert a manifest's WSL /mnt/d path into a Windows path when needed."""
    if value.startswith("/mnt/d/"):
        return Path("D:/" + value.removeprefix("/mnt/d/"))
    return Path(value)


def latest_internal_vtu(row: pd.Series) -> Path:
    case = windows_path(row.case_path)
    candidate = case / "VTK" / f"{row.id}_{row.latest_time}" / "internal.vtu"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing CFD field export: {candidate}")
    return candidate


def naca0012_polygon(points: int = 401) -> np.ndarray:
    """Return a closed NACA 0012 x-z boundary in chord coordinates."""
    x = (1.0 - np.cos(np.linspace(0.0, np.pi, points))) / 2.0
    thickness = 5 * 0.12 * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return np.vstack((np.c_[x, thickness], np.c_[x[::-1], -thickness[::-1]]))


def interpolation_plan(vtu_path: Path, grid_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reusable triangle interpolation weights from the common CFD mesh."""
    mesh = meshio.read(vtu_path)
    # The 2D OpenFOAM mesh is extruded in y. Collapse duplicate x-z points.
    xy = mesh.points[:, [0, 2]]
    unique_xy, unique_indices = np.unique(xy, axis=0, return_index=True)
    triangulation = Delaunay(unique_xy)
    simplices = triangulation.find_simplex(grid_xy)
    if np.any(simplices < 0):
        raise ValueError("Chosen POD grid extends outside the CFD mesh.")
    transforms = triangulation.transform[simplices]
    delta = grid_xy - transforms[:, 2]
    barycentric = np.einsum("nij,nj->ni", transforms[:, :2], delta)
    weights = np.c_[barycentric, 1.0 - barycentric.sum(axis=1)]
    vertices = triangulation.simplices[simplices]
    return unique_indices, vertices, weights


def sample_case(
    vtu_path: Path,
    unique_indices: np.ndarray,
    vertices: np.ndarray,
    weights: np.ndarray,
    include_nut: bool = False,
) -> np.ndarray:
    mesh = meshio.read(vtu_path)
    velocity = mesh.point_data["U"][unique_indices]
    pressure = mesh.point_data["p"][unique_indices]
    raw_fields = (velocity[:, 0], velocity[:, 2], pressure)
    if include_nut:
        raw_fields += (mesh.point_data["nut"][unique_indices],)
    return np.stack([(raw[vertices] * weights).sum(axis=1) for raw in raw_fields])


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "Re", "AoA_deg", "latest_time", "case_path", "converged", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame[(frame.converged.astype(str).str.lower() == "true") & (frame.status == "outputs_verified")].copy()
    if frame.empty:
        raise ValueError(f"No verified converged cases in {path}")
    return frame.sort_values("id").reset_index(drop=True)


def error_by_field(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Relative L2 errors over cases and grid points, separately per field."""
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[1] != len(FIELDS):
        raise ValueError("Expected matching arrays with shape (cases, fields, fluid_points)")
    return {
        field: float(np.linalg.norm(prediction[:, i] - reference[:, i]) / np.linalg.norm(reference[:, i]))
        for i, field in enumerate(FIELDS)
    }


def train_physics_informed(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    train_fields: np.ndarray,
    validation_fields: np.ndarray | None,
    train_nut: np.ndarray,
    validation_nut: np.ndarray | None,
    baseline_validation: np.ndarray | None,
    coefficients: np.ndarray,
    mean_state: np.ndarray,
    pod_modes: np.ndarray,
    field_scale: np.ndarray,
    baseline_model: MLPRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    x: np.ndarray,
    z: np.ndarray,
    fluid_mask: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> dict[str, object]:
    """Fine-tune POD coefficient networks with gridded steady RANS residuals.

    `p` is the kinematic pressure exported by simpleFoam. Its momentum equation
    uses the deviatoric effective-viscosity stress, so no separate 2k/3 gradient
    is added here. This constrains mean-flow RANS only, not the full SST transport
    equations. The Cartesian grid is a subdomain of the CFD mesh: its outer-edge
    loss is a CFD-sampled Dirichlet condition, not a freestream condition. The
    actual airfoil wall is not represented by grid nodes and is not penalized.
    """
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("--physics-informed requires PyTorch; install the project requirements") from exc

    torch.manual_seed(args.physics_seed)
    torch.set_num_threads(args.physics_threads)
    n_fluid = int(fluid_mask.sum())
    if train_nut.shape != (len(train), n_fluid):
        raise ValueError("Turbulent viscosity snapshots do not match the POD grid")
    pde_mask = binary_erosion(fluid_mask, iterations=2, border_value=0)
    if not pde_mask.any():
        raise ValueError("No fluid cells have a valid two-cell PDE stencil")
    edge = np.zeros_like(fluid_mask, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge_index = np.flatnonzero(edge.ravel()[fluid_mask.ravel()])
    if not len(edge_index):
        raise ValueError("The POD subdomain has no fluid edge points")

    # A separate POD decoder supplies nut for new conditions. Its rank is capped
    # because only 40 CFD cases are available; predictions are clipped at zero.
    nut_scale = float(train_nut.std()) or 1.0
    scaled_nut = train_nut / nut_scale
    mean_nut = scaled_nut.mean(axis=0)
    nut_left, nut_singular, nut_modes_all = np.linalg.svd(scaled_nut - mean_nut, full_matrices=False)
    del nut_left
    nut_energy = np.cumsum(nut_singular**2) / np.sum(nut_singular**2)
    nut_rank = min(args.nut_modes, len(nut_singular))
    nut_modes = nut_modes_all[:nut_rank]
    nut_coefficients = (scaled_nut - mean_nut) @ nut_modes.T
    nut_scaler = StandardScaler().fit(nut_coefficients)
    train_inputs = x_scaler.transform(train[["Re", "AoA_deg"]])
    nut_model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs",
        max_iter=5000, random_state=args.physics_seed,
    ).fit(train_inputs, nut_scaler.transform(nut_coefficients))

    class CoefficientNet(nn.Module):
        def __init__(self, outputs: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs)
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.layers(inputs)

        def load_sklearn(self, source: MLPRegressor) -> None:
            with torch.no_grad():
                for index, layer in enumerate((self.layers[0], self.layers[2], self.layers[4])):
                    layer.weight.copy_(torch.from_numpy(source.coefs_[index].T).float())
                    layer.bias.copy_(torch.from_numpy(source.intercepts_[index]).float())

    flow_net = CoefficientNet(pod_modes.shape[0])
    flow_net.load_sklearn(baseline_model)
    nut_net = CoefficientNet(nut_rank)
    nut_net.load_sklearn(nut_model)
    parameters = list(flow_net.parameters()) + list(nut_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.physics_lr)

    to_tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)
    input_t = to_tensor(train_inputs)
    target_coef_t = to_tensor(y_scaler.transform(coefficients))
    target_nut_coef_t = to_tensor(nut_scaler.transform(nut_coefficients))
    field_t = to_tensor(train_fields)
    nut_t = to_tensor(train_nut)
    uinf_t = to_tensor(train.Uinf_mps.to_numpy())
    re_t = to_tensor(train.Re.to_numpy())
    mode_t, mean_t, scale_t = map(to_tensor, (pod_modes, mean_state, field_scale))
    nut_mode_t, nut_mean_t = map(to_tensor, (nut_modes, mean_nut))
    flow_coef_mean_t, flow_coef_scale_t = map(to_tensor, (y_scaler.mean_, y_scaler.scale_))
    nut_coef_mean_t, nut_coef_scale_t = map(to_tensor, (nut_scaler.mean_, nut_scaler.scale_))
    fluid_index_t = torch.from_numpy(np.flatnonzero(fluid_mask.ravel()).copy()).long()
    pde_mask_t = torch.from_numpy(pde_mask.copy())
    edge_index_t = torch.from_numpy(edge_index.copy()).long()
    nx, nz = len(x), len(z)
    dx, dz = float(x[1] - x[0]), float(z[1] - z[0])

    def reconstruct(flow_std: torch.Tensor, nut_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow_coef = flow_std * flow_coef_scale_t + flow_coef_mean_t
        q = (mean_t + flow_coef @ mode_t).reshape(-1, len(FIELDS), n_fluid) * scale_t[None, :, None]
        nut_coef = nut_std * nut_coef_scale_t + nut_coef_mean_t
        turbulent_viscosity = (nut_mean_t + nut_coef @ nut_mode_t) * nut_scale
        return q, turbulent_viscosity.clamp_min(0.0)

    def nondimensional_fields(q: torch.Tensor, uinf: torch.Tensor) -> torch.Tensor:
        units = torch.stack((uinf, uinf, uinf.square()), dim=1).unsqueeze(-1)
        return q / units

    def ddx(v: torch.Tensor) -> torch.Tensor:
        return (torch.roll(v, -1, -1) - torch.roll(v, 1, -1)) / (2.0 * dx)

    def ddz(v: torch.Tensor) -> torch.Tensor:
        return (torch.roll(v, -1, -2) - torch.roll(v, 1, -2)) / (2.0 * dz)

    def rans_losses(
        q: torch.Tensor, turbulent_viscosity: torch.Tensor, uinf: torch.Tensor, reynolds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Every operation remains differentiable with respect to POD coefficients.
        values = torch.cat((q, turbulent_viscosity[:, None, :]), dim=1)
        full = torch.zeros((len(q), 4, nx * nz), dtype=q.dtype).index_copy(2, fluid_index_t, values)
        full = full.reshape(-1, 4, nz, nx)
        speed = uinf[:, None, None]
        u, w, p = full[:, 0] / speed, full[:, 1] / speed, full[:, 2] / speed.square()
        nu_effective = 1.0 / reynolds[:, None, None] + full[:, 3] / speed  # chord = 1 m
        ux, uz, wx, wz = ddx(u), ddz(u), ddx(w), ddz(w)
        divergence = ux + wz
        tau_xx = nu_effective * (2.0 * ux - 2.0 * divergence / 3.0)
        tau_zz = nu_effective * (2.0 * wz - 2.0 * divergence / 3.0)
        tau_xz = nu_effective * (uz + wx)
        rx = u * ux + w * uz + ddx(p) - ddx(tau_xx) - ddz(tau_xz)
        rz = u * wx + w * wz + ddz(p) - ddx(tau_xz) - ddz(tau_zz)
        return divergence[:, pde_mask_t].square().mean(), (
            rx[:, pde_mask_t].square().mean() + rz[:, pde_mask_t].square().mean()
        ) / 2.0

    # A uniform flow must satisfy both gridded residuals, including near masks.
    uniform = torch.zeros((1, len(FIELDS), n_fluid), dtype=torch.float32)
    uniform[:, 0] = 1.0
    uniform_continuity, uniform_momentum = rans_losses(
        uniform, torch.zeros((1, n_fluid)), torch.ones(1), torch.full((1,), 1e6)
    )
    if max(float(uniform_continuity), float(uniform_momentum)) > 1e-10:
        raise AssertionError("The gridded RANS stencil does not preserve uniform flow")

    def loss_parts(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        flow_std, nut_std = flow_net(input_t[indices]), nut_net(input_t[indices])
        q, turbulent_viscosity = reconstruct(flow_std, nut_std)
        q_star = nondimensional_fields(q, uinf_t[indices])
        truth_star = nondimensional_fields(field_t[indices], uinf_t[indices])
        continuity, momentum = rans_losses(q, turbulent_viscosity, uinf_t[indices], re_t[indices])
        return {
            "data": F.mse_loss(flow_std, target_coef_t[indices]) + F.mse_loss(q_star, truth_star),
            "nut_data": F.mse_loss(nut_std, target_nut_coef_t[indices]),
            "continuity": continuity,
            "momentum": momentum,
            # These are CFD-sampled Dirichlet values on the Cartesian subdomain edge.
            "boundary": F.mse_loss(q_star[:, :, edge_index_t], truth_star[:, :, edge_index_t]),
        }

    weights = {
        "data": 1.0,
        "nut_data": args.lambda_nut_data,
        "continuity": args.lambda_continuity,
        "momentum": args.lambda_momentum,
        "boundary": args.lambda_boundary,
    }
    history = []
    for epoch in range(1, args.physics_epochs + 1):
        permutation = torch.randperm(len(train))
        sums = {key: 0.0 for key in weights}
        for indices in permutation.split(args.physics_batch_size):
            optimizer.zero_grad()
            parts = loss_parts(indices)
            total = sum(weights[key] * value for key, value in parts.items())
            total.backward()
            optimizer.step()
            for key, value in parts.items():
                sums[key] += float(value.detach()) * len(indices)
        record = {"epoch": epoch, **{key: value / len(train) for key, value in sums.items()}}
        record["total"] = sum(weights[key] * record[key] for key in weights)
        history.append(record)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.physics_epochs:
            print(f"physics epoch {epoch}/{args.physics_epochs}: " + ", ".join(
                f"{key}={record[key]:.3g}" for key in ("data", "continuity", "momentum", "boundary")
            ), flush=True)
    pd.DataFrame(history).to_csv(output / "physics_loss_history.csv", index=False)

    def predict(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        inputs = to_tensor(x_scaler.transform(frame[["Re", "AoA_deg"]]))
        with torch.no_grad():
            q, turbulent_viscosity = reconstruct(flow_net(inputs), nut_net(inputs))
        return q.numpy(), turbulent_viscosity.numpy()

    train_prediction, train_nut_prediction = predict(train)

    def residual_metrics(q: np.ndarray, nut: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
        with torch.no_grad():
            continuity, momentum = rans_losses(
                to_tensor(q), to_tensor(nut), to_tensor(frame.Uinf_mps.to_numpy()),
                to_tensor(frame.Re.to_numpy()),
            )
        return {"continuity_mse": float(continuity), "momentum_mse": float(momentum)}

    baseline_train_coefficients = y_scaler.inverse_transform(baseline_model.predict(train_inputs))
    baseline_train_fields = (
        mean_state + baseline_train_coefficients @ pod_modes
    ).reshape(train_fields.shape) * field_scale[None, :, None]
    np.savez_compressed(
        output / "physics_nut_basis.npz", mean=mean_nut, modes=nut_modes,
        scale=nut_scale, singular_values=nut_singular, energy=nut_energy,
    )
    torch.save({
        "flow_state_dict": flow_net.state_dict(), "nut_state_dict": nut_net.state_dict(),
        "x_mean": to_tensor(x_scaler.mean_), "x_scale": to_tensor(x_scaler.scale_),
        "flow_coefficient_mean": to_tensor(y_scaler.mean_),
        "flow_coefficient_scale": to_tensor(y_scaler.scale_),
        "nut_coefficient_mean": to_tensor(nut_scaler.mean_),
        "nut_coefficient_scale": to_tensor(nut_scaler.scale_),
        "nut_modes": nut_rank,
    }, output / "physics_pod_mlp.pt")
    result: dict[str, object] = {
        "description": "POD coefficient MLP with steady incompressible eddy-viscosity RANS mean-flow regularization",
        "not_enforced": "k-omega SST transport equations and exact airfoil-wall boundary values",
        "pressure_convention": "simpleFoam exported kinematic p; isotropic 2k/3 not added separately",
        "nut_modes": nut_rank,
        "nut_retained_energy": float(nut_energy[nut_rank - 1]),
        "physics_stencil_points": int(pde_mask.sum()),
        "subdomain_edge_points": int(len(edge_index)),
        "weights": weights,
        "epochs": args.physics_epochs,
        "training_relative_l2": error_by_field(train_fields, train_prediction),
        "training_nut_relative_l2": float(np.linalg.norm(train_nut_prediction - train_nut) / np.linalg.norm(train_nut)),
        "training_gridded_rans_residual_mse": {
            "cfd": residual_metrics(train_fields, train_nut, train),
            "baseline_with_predicted_nut": residual_metrics(baseline_train_fields, train_nut_prediction, train),
            "physics_informed": residual_metrics(train_prediction, train_nut_prediction, train),
        },
    }
    if validation is not None:
        assert validation_fields is not None and validation_nut is not None and baseline_validation is not None
        predicted, predicted_nut = predict(validation)
        np.savez_compressed(
            output / "physics_validation_predictions.npz", fields=predicted,
            nut=predicted_nut, ids=validation.id.to_numpy(dtype=str),
        )
        result["validation_relative_l2"] = error_by_field(validation_fields, predicted)
        result["validation_nut_relative_l2"] = float(
            np.linalg.norm(predicted_nut - validation_nut) / np.linalg.norm(validation_nut)
        )
        result["validation_gridded_rans_residual_mse"] = {
            "cfd": residual_metrics(validation_fields, validation_nut, validation),
            "baseline_with_predicted_nut": residual_metrics(baseline_validation, predicted_nut, validation),
            "physics_informed": residual_metrics(predicted, predicted_nut, validation),
        }
        rows = []
        for case_index, row in validation.iterrows():
            case_true = validation_fields[case_index:case_index + 1]
            case_pred = predicted[case_index:case_index + 1]
            case_base = baseline_validation[case_index:case_index + 1]
            case_error, base_error = error_by_field(case_true, case_pred), error_by_field(case_true, case_base)
            case_frame = validation.iloc[case_index:case_index + 1]
            cfd_residual = residual_metrics(case_true, validation_nut[case_index:case_index + 1], case_frame)
            base_residual = residual_metrics(case_base, predicted_nut[case_index:case_index + 1], case_frame)
            physics_residual = residual_metrics(case_pred, predicted_nut[case_index:case_index + 1], case_frame)
            rows.append({
                "id": row.id, "Re": row.Re, "AoA_deg": row.AoA_deg,
                **{f"baseline_{key}_relative_l2": value for key, value in base_error.items()},
                **{f"physics_{key}_relative_l2": value for key, value in case_error.items()},
                "physics_nut_relative_l2": float(
                    np.linalg.norm(predicted_nut[case_index] - validation_nut[case_index])
                    / np.linalg.norm(validation_nut[case_index])
                ),
                **{f"cfd_{key}": value for key, value in cfd_residual.items()},
                **{f"baseline_{key}": value for key, value in base_residual.items()},
                **{f"physics_{key}": value for key, value in physics_residual.items()},
            })
        pd.DataFrame(rows).to_csv(output / "physics_validation_by_case.csv", index=False)
    return result


def predict_physics_case(model_dir: Path, reynolds: float, angle_deg: float) -> dict[str, np.ndarray]:
    """Predict a new flow and nut field using saved physics-informed POD weights."""
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("Physics-informed inference requires PyTorch") from exc
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")
    checkpoint = torch.load(model_dir / "physics_pod_mlp.pt", map_location="cpu", weights_only=True)

    def forward_saved(state: dict[str, torch.Tensor], values: torch.Tensor) -> torch.Tensor:
        for index in (0, 2, 4):
            values = F.linear(values, state[f"layers.{index}.weight"], state[f"layers.{index}.bias"])
            if index != 4:
                values = F.relu(values)
        return values

    conditions = torch.tensor([[reynolds, angle_deg]], dtype=torch.float32)
    inputs = (conditions - checkpoint["x_mean"]) / checkpoint["x_scale"]
    with torch.no_grad():
        flow_std = forward_saved(checkpoint["flow_state_dict"], inputs)
        nut_std = forward_saved(checkpoint["nut_state_dict"], inputs)
        flow_coefficients = (
            flow_std * checkpoint["flow_coefficient_scale"] + checkpoint["flow_coefficient_mean"]
        ).numpy()[0]
        nut_coefficients = (
            nut_std * checkpoint["nut_coefficient_scale"] + checkpoint["nut_coefficient_mean"]
        ).numpy()[0]
    with np.load(model_dir / "pod_basis.npz") as basis, np.load(model_dir / "physics_nut_basis.npz") as nut_basis:
        fields = ((basis["mean_state"] + flow_coefficients @ basis["modes"])
                  .reshape(len(FIELDS), -1) * basis["field_scale"][:, None])
        nut = np.maximum(0.0, (nut_basis["mean"] + nut_coefficients @ nut_basis["modes"])
                         * float(nut_basis["scale"]))
        return {
            "fields": fields, "nut": nut,
            "x": basis["x"].copy(), "z": basis["z"].copy(),
            "fluid_mask": basis["fluid_mask"].copy(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.999)
    parser.add_argument("--physics-informed", action="store_true", help="Fine-tune a POD MLP with steady RANS losses")
    parser.add_argument("--physics-epochs", type=int, default=100)
    parser.add_argument("--physics-batch-size", type=int, default=8)
    parser.add_argument("--physics-lr", type=float, default=1e-4)
    parser.add_argument("--physics-seed", type=int, default=42)
    parser.add_argument("--physics-threads", type=int, default=4)
    parser.add_argument("--nut-modes", type=int, default=8)
    parser.add_argument("--lambda-nut-data", type=float, default=1.0)
    parser.add_argument("--lambda-continuity", type=float, default=0.01)
    parser.add_argument("--lambda-momentum", type=float, default=0.01)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)
    args = parser.parse_args()
    if args.physics_batch_size < 1 or args.physics_epochs < 1 or args.nut_modes < 1 or args.physics_threads < 1:
        parser.error("physics epochs, batch size, threads, and nut modes must be positive")
    if min(args.lambda_nut_data, args.lambda_continuity, args.lambda_momentum, args.lambda_boundary) < 0:
        parser.error("loss weights must be nonnegative")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train = load_manifest(args.training_manifest)
    validation = load_manifest(args.validation_manifest) if args.validation_manifest else None

    x = np.linspace(-2.0, 4.0, args.nx)
    z = np.linspace(-1.5, 1.5, args.nz)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    grid_xy = np.c_[xx.ravel(), zz.ravel()]
    solid_mask = MplPath(naca0012_polygon()).contains_points(grid_xy)
    fluid_mask = ~solid_mask

    unique_indices, vertices, weights = interpolation_plan(latest_internal_vtu(train.iloc[0]), grid_xy)

    def fields_for(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        result, nut_result = [], []
        for _, row in frame.iterrows():
            sampled = sample_case(
                latest_internal_vtu(row), unique_indices, vertices, weights,
                include_nut=args.physics_informed,
            )
            result.append(sampled[:len(FIELDS), fluid_mask])
            if args.physics_informed:
                nut_result.append(sampled[len(FIELDS), fluid_mask])
        return np.asarray(result, dtype=np.float32), (
            np.asarray(nut_result, dtype=np.float32) if args.physics_informed else None
        )

    train_fields, train_nut = fields_for(train)
    validation_fields, validation_nut = fields_for(validation) if validation is not None else (None, None)

    # Per-field scale avoids pressure magnitude dominating the velocity POD modes.
    field_scale = train_fields.reshape(len(train), len(FIELDS), -1).std(axis=(0, 2))
    field_scale[field_scale == 0] = 1.0
    scaled = train_fields / field_scale[None, :, None]
    state = scaled.reshape(len(train), -1)
    mean_state = state.mean(axis=0)
    centred = state - mean_state
    modes, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    rank = int(np.searchsorted(energy, args.energy) + 1)
    pod_modes = right_vectors[:rank]
    coefficients = centred @ pod_modes.T
    train_reconstruction = (mean_state + coefficients @ pod_modes).reshape(train_fields.shape) * field_scale[None, :, None]

    np.savez_compressed(
        output / "pod_basis.npz",
        x=x,
        z=z,
        fluid_mask=fluid_mask.reshape(zz.shape),
        field_scale=field_scale,
        mean_state=mean_state,
        modes=pod_modes,
        singular_values=singular_values,
        energy=energy,
        coefficients=coefficients,
    )
    np.savez_compressed(output / "training_snapshots.npz", fields=train_fields, ids=train.id.to_numpy(dtype=str))
    train.to_csv(output / "training_cases.csv", index=False)

    # The MLP maps operating conditions to POD coefficients. This is the
    # architecture specified in the AE646 project brief: 2 → 64 → 64 → r.
    # MLPRegressor minimizes mean-squared error on the standardized coefficients.
    x_scaler = StandardScaler().fit(train[["Re", "AoA_deg"]])
    y_scaler = StandardScaler().fit(coefficients)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs",
        max_iter=5000, random_state=42,
    ).fit(x_scaler.transform(train[["Re", "AoA_deg"]]), y_scaler.transform(coefficients))
    joblib.dump({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}, output / "pod_mlp.joblib")

    train_prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(train[["Re", "AoA_deg"]])))
    train_mlp_fields = (mean_state + train_prediction @ pod_modes).reshape(train_fields.shape) * field_scale[None, :, None]
    summary: dict[str, object] = {
        "training_cases": int(len(train)),
        "validation_cases": int(0 if validation is None else len(validation)),
        "grid_shape": [args.nz, args.nx],
        "state_fields": list(FIELDS),
        "pod_energy_threshold": args.energy,
        "retained_modes": rank,
        "retained_energy": float(energy[rank - 1]),
        "training_pod_relative_l2": error_by_field(train_fields, train_reconstruction),
        "training_mlp_relative_l2": error_by_field(train_fields, train_mlp_fields),
        "mlp_architecture": [2, 64, 64, rank],
        "mlp_activation": "ReLU",
        "mlp_loss": "MSE on standardized POD coefficients",
        "training_coefficient_mse": float(np.mean((train_prediction - coefficients) ** 2)),
    }

    predicted_fields = None
    if validation is not None:
        validation.to_csv(output / "validation_cases.csv", index=False)
        np.savez_compressed(output / "validation_snapshots.npz", fields=validation_fields, ids=validation.id.to_numpy(dtype=str))
        predicted_coefficients = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(validation[["Re", "AoA_deg"]]))
        )
        validation_state = (validation_fields / field_scale[None, :, None]).reshape(len(validation), -1)
        validation_coefficients = (validation_state - mean_state) @ pod_modes.T
        predicted_fields = (mean_state + predicted_coefficients @ pod_modes).reshape(validation_fields.shape)
        predicted_fields *= field_scale[None, :, None]
        summary["validation_mlp_relative_l2"] = error_by_field(validation_fields, predicted_fields)
        summary["validation_coefficient_mse"] = float(np.mean((predicted_coefficients - validation_coefficients) ** 2))
        np.savez_compressed(
            output / "validation_predictions.npz",
            fields=predicted_fields,
            ids=validation.id.to_numpy(dtype=str),
            true_coefficients=validation_coefficients,
            predicted_coefficients=predicted_coefficients,
        )

    if args.physics_informed:
        assert train_nut is not None
        summary["physics_informed"] = train_physics_informed(
            train=train, validation=validation, train_fields=train_fields,
            validation_fields=validation_fields, train_nut=train_nut,
            validation_nut=validation_nut, baseline_validation=predicted_fields,
            coefficients=coefficients, mean_state=mean_state, pod_modes=pod_modes,
            field_scale=field_scale, baseline_model=model, x_scaler=x_scaler,
            y_scaler=y_scaler, x=x, z=z, fluid_mask=fluid_mask.reshape(zz.shape),
            args=args, output=output,
        )

    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(energy) + 1), energy, marker="o", ms=3)
    plt.axhline(args.energy, color="tab:red", ls="--", label=f"target = {args.energy:.3%}")
    plt.axvline(rank, color="tab:green", ls="--", label=f"retained modes = {rank}")
    plt.xlabel("POD modes retained")
    plt.ylabel("Cumulative energy")
    plt.ylim(0.0, 1.01)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "pod_energy.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
