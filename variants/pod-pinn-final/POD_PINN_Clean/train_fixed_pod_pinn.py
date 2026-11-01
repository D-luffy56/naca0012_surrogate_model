from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
POD_FILE = ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz"
OPERATOR_FILE = ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz"
MODEL_DIR = ROOT / "artifacts" / "models"

ARCHITECTURES = (32, 64, 128)
L2_WEIGHT = 1.0e-4
MAX_LBFGS_ITERATIONS = 500
LOSS_MULTIPLIERS = {"data": 1.0, "continuity": 1.0, "momentum": 1.0, "boundary": 1.0}
DTYPE = torch.float64


class CoefficientNetwork(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, rank),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=DTYPE)


def reconstruct(expansion: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return expansion[:, 0][None, :, :] + torch.einsum(
        "br,nrf->bnf", coefficients, expansion[:, 1:]
    )


def residuals(
    packed: torch.Tensor, coefficients: torch.Tensor, reynolds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Each four-column group is [u, w, p, nut].
    reconstructed = reconstruct(packed, coefficients)
    values = reconstructed[:, :, 0:4]
    derivative_x = reconstructed[:, :, 4:8]
    derivative_z = reconstructed[:, :, 8:12]
    derivative_xx = reconstructed[:, :, 12:16]
    derivative_xz = reconstructed[:, :, 16:20]
    derivative_zz = reconstructed[:, :, 20:24]

    u, w, _, nut = values.unbind(dim=-1)
    ux, wx, pressure_x, nut_x = derivative_x.unbind(dim=-1)
    uz, wz, pressure_z, nut_z = derivative_z.unbind(dim=-1)
    uxx, wxx = derivative_xx[:, :, 0], derivative_xx[:, :, 1]
    uxz, wxz = derivative_xz[:, :, 0], derivative_xz[:, :, 1]
    uzz, wzz = derivative_zz[:, :, 0], derivative_zz[:, :, 1]

    divergence = ux + wz
    effective_viscosity = 1.0 / reynolds[:, None] + torch.clamp(nut, min=0.0)
    a = 2.0 * ux - (2.0 / 3.0) * divergence
    b = uz + wx
    c = 2.0 * wz - (2.0 / 3.0) * divergence
    a_x = 2.0 * uxx - (2.0 / 3.0) * (uxx + wxz)
    b_z = uzz + wxz
    b_x = uxz + wxx
    c_z = 2.0 * wzz - (2.0 / 3.0) * (uxz + wzz)
    stress_divergence_x = (
        nut_x * a + effective_viscosity * a_x + nut_z * b + effective_viscosity * b_z
    )
    stress_divergence_z = (
        nut_x * b + effective_viscosity * b_x + nut_z * c + effective_viscosity * c_z
    )
    momentum_x = u * ux + w * uz + pressure_x - stress_divergence_x
    momentum_z = u * wx + w * wz + pressure_z - stress_divergence_z
    return divergence, momentum_x, momentum_z, nut


def relative_l2(reference: np.ndarray, estimate: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = np.sum(weights[None, :, None] * (estimate - reference) ** 2, axis=(0, 1))
    denominator = np.sum(weights[None, :, None] * reference**2, axis=(0, 1))
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def load_native_cases(case_ids: np.ndarray) -> np.ndarray:
    result = []
    for case_id in case_ids:
        with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
            fields = archive["fields"][:, :4].astype(np.float64)
            metadata = json.loads(archive["metadata"].item())
        u_inf = float(metadata["Uinf_mps"])
        chord = float(metadata["chord_m"])
        result.append(fields / np.array((u_inf, u_inf, u_inf**2, u_inf * chord)))
    return np.stack(result)


def pressure_coefficients(
    boundary: np.ndarray,
    parameters: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    patch: np.ndarray,
) -> np.ndarray:
    wall = patch < 2
    # OpenFOAM wall normals point out of the fluid domain (into the body).
    # The corresponding pressure force on the body is +p*n_fluid.
    force_xz = 2.0 * np.sum(
        boundary[:, wall, 2, None] * normals[None, wall, :] * areas[None, wall, None], axis=1
    )
    alpha = np.deg2rad(parameters[:, 1])
    drag_direction = np.column_stack((np.cos(alpha), np.sin(alpha)))
    lift_direction = np.column_stack((-np.sin(alpha), np.cos(alpha)))
    cd_pressure = np.sum(force_xz * drag_direction, axis=1)
    cl_pressure = np.sum(force_xz * lift_direction, axis=1)
    return np.column_stack((cl_pressure, cd_pressure))


def train(width: int) -> dict:
    torch.manual_seed(1729 + width)
    np.random.seed(1729 + width)
    torch.set_num_threads(min(9, torch.get_num_threads()))

    with np.load(OPERATOR_FILE, allow_pickle=False) as operator:
        packed_np = np.concatenate(
            (
                operator["expansion_values"],
                operator["expansion_dx"],
                operator["expansion_dz"],
                operator["expansion_dxx"],
                operator["expansion_dxz"],
                operator["expansion_dzz"],
            ),
            axis=-1,
        ).astype(np.float64)
        train_coefficients_np = operator["train_coefficients"].astype(np.float64)
        validation_coefficients_np = operator["validation_coefficients"].astype(np.float64)
        coefficient_mean_np = operator["coefficient_mean"].astype(np.float64)
        coefficient_scale_np = operator["coefficient_scale"].astype(np.float64)
        train_parameters_np = operator["train_parameters"].astype(np.float64)
        validation_parameters_np = operator["validation_parameters"].astype(np.float64)
        parameter_center_np = operator["parameter_center"].astype(np.float64)
        parameter_scale_np = operator["parameter_scale"].astype(np.float64)
        train_boundary_np = operator["train_boundary"].astype(np.float64)
        validation_boundary_np = operator["validation_boundary"].astype(np.float64)
        boundary_intercept_np = operator["boundary_affine_intercept"].astype(np.float64)
        boundary_modes_np = operator["boundary_affine_modes"].astype(np.float64)
        boundary_scales_np = operator["boundary_scales"].astype(np.float64)
        boundary_areas = operator["boundary_areas"].astype(np.float64)
        boundary_patch = operator["boundary_patch"].astype(np.int64)
        train_ids = operator["train_ids"].astype(str)
        validation_ids = operator["validation_ids"].astype(str)
    with np.load(DATA / "geometry.npz", allow_pickle=False) as geometry:
        boundary_normals = geometry["boundary_normals"].astype(np.float64)

    rank = train_coefficients_np.shape[1]
    packed = tensor(packed_np)
    coefficient_mean = tensor(coefficient_mean_np)
    coefficient_scale = tensor(coefficient_scale_np)
    train_inputs = tensor((train_parameters_np - parameter_center_np) / parameter_scale_np)
    validation_inputs = tensor(
        (validation_parameters_np - parameter_center_np) / parameter_scale_np
    )
    train_targets = tensor(
        (train_coefficients_np - coefficient_mean_np) / coefficient_scale_np
    )
    validation_targets = tensor(
        (validation_coefficients_np - coefficient_mean_np) / coefficient_scale_np
    )
    reynolds_train = tensor(train_parameters_np[:, 0])
    reynolds_validation = tensor(validation_parameters_np[:, 0])
    train_boundary = tensor(train_boundary_np)
    validation_boundary = tensor(validation_boundary_np)
    boundary_intercept = tensor(boundary_intercept_np)
    boundary_modes = tensor(boundary_modes_np)
    boundary_scales = tensor(boundary_scales_np)

    model = CoefficientNetwork(width, rank).to(dtype=DTYPE)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=MAX_LBFGS_ITERATIONS,
        max_eval=int(MAX_LBFGS_ITERATIONS * 1.25),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        standardized = model(train_inputs)
        coefficients = coefficient_mean[None, :] + coefficient_scale[None, :] * standardized
        divergence, momentum_x, momentum_z, _ = residuals(
            packed, coefficients, reynolds_train
        )
        predicted_boundary = boundary_intercept[None, :, :] + torch.einsum(
            "br,rnf->bnf", coefficients, boundary_modes
        )
        data_loss = torch.mean((standardized - train_targets) ** 2)
        continuity_loss = torch.mean(divergence**2)
        momentum_loss = 0.5 * torch.mean(momentum_x**2 + momentum_z**2)
        boundary_loss = torch.mean(
            ((predicted_boundary - train_boundary) / boundary_scales[None, None, :]) ** 2
        )
        l2_loss = sum(torch.sum(parameter**2) for parameter in model.parameters())
        total = (
            LOSS_MULTIPLIERS["data"] * data_loss
            + LOSS_MULTIPLIERS["continuity"] * continuity_loss
            + LOSS_MULTIPLIERS["momentum"] * momentum_loss
            + LOSS_MULTIPLIERS["boundary"] * boundary_loss
            + L2_WEIGHT * l2_loss
        )
        total.backward()
        record = {
            "closure": len(history) + 1,
            "total": float(total.detach()),
            "data": float(data_loss.detach()),
            "continuity": float(continuity_loss.detach()),
            "momentum": float(momentum_loss.detach()),
            "boundary": float(boundary_loss.detach()),
            "l2_unweighted": float(l2_loss.detach()),
        }
        history.append(record)
        if len(history) == 1 or len(history) % 25 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"width={width} closure={len(history)} total={record['total']:.6e} "
                f"data={record['data']:.3e} cont={record['continuity']:.3e} "
                f"mom={record['momentum']:.3e} bc={record['boundary']:.3e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        return total

    optimizer.step(closure)
    elapsed_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        standardized_train = model(train_inputs)
        standardized_validation = model(validation_inputs)
        predicted_train_coefficients = (
            coefficient_mean[None, :] + coefficient_scale[None, :] * standardized_train
        )
        predicted_validation_coefficients = (
            coefficient_mean[None, :] + coefficient_scale[None, :] * standardized_validation
        )
        train_div, train_mx, train_mz, train_nut = residuals(
            packed, predicted_train_coefficients, reynolds_train
        )
        val_div, val_mx, val_mz, val_nut = residuals(
            packed, predicted_validation_coefficients, reynolds_validation
        )
        predicted_train_boundary = boundary_intercept[None, :, :] + torch.einsum(
            "br,rnf->bnf", predicted_train_coefficients, boundary_modes
        )
        predicted_validation_boundary = boundary_intercept[None, :, :] + torch.einsum(
            "br,rnf->bnf", predicted_validation_coefficients, boundary_modes
        )

    predicted_train_coefficients_np = predicted_train_coefficients.numpy()
    predicted_validation_coefficients_np = predicted_validation_coefficients.numpy()
    predicted_train_boundary_np = predicted_train_boundary.numpy()
    predicted_validation_boundary_np = predicted_validation_boundary.numpy()

    with np.load(POD_FILE, allow_pickle=False) as pod:
        native_mean = pod["mean"].astype(np.float64)
        native_weights = pod["weights"].astype(np.float64)
        field_rms = pod["field_rms"].astype(np.float64)
        basis = pod["basis"].astype(np.float64)
    sqrt_weights = np.sqrt(native_weights)

    def native_reconstruction(coefficients: np.ndarray) -> np.ndarray:
        normalized = (
            (coefficients @ basis).reshape(len(coefficients), len(native_weights), 4)
            / sqrt_weights[None, :, None]
        )
        return native_mean[None, :, :] + normalized * field_rms[None, None, :]

    train_native_reference = load_native_cases(train_ids)
    validation_native_reference = load_native_cases(validation_ids)
    train_native_prediction = native_reconstruction(predicted_train_coefficients_np)
    validation_native_prediction = native_reconstruction(predicted_validation_coefficients_np)
    train_field_error = relative_l2(
        train_native_reference, train_native_prediction, native_weights
    )
    validation_field_error = relative_l2(
        validation_native_reference, validation_native_prediction, native_weights
    )

    area_weights = boundary_areas / boundary_areas.sum()
    train_boundary_error = relative_l2(
        train_boundary_np, predicted_train_boundary_np, area_weights
    )
    validation_boundary_error = relative_l2(
        validation_boundary_np, predicted_validation_boundary_np, area_weights
    )
    true_pressure_coefficients = pressure_coefficients(
        validation_boundary_np,
        validation_parameters_np,
        boundary_areas,
        boundary_normals,
        boundary_patch,
    )
    predicted_pressure_coefficients = pressure_coefficients(
        predicted_validation_boundary_np,
        validation_parameters_np,
        boundary_areas,
        boundary_normals,
        boundary_patch,
    )
    pressure_coefficient_mae = np.mean(
        np.abs(predicted_pressure_coefficients - true_pressure_coefficients), axis=0
    )

    def residual_summary(div: torch.Tensor, mx: torch.Tensor, mz: torch.Tensor) -> dict:
        return {
            "continuity_rms": float(torch.sqrt(torch.mean(div**2))),
            "momentum_x_rms": float(torch.sqrt(torch.mean(mx**2))),
            "momentum_z_rms": float(torch.sqrt(torch.mean(mz**2))),
        }

    metrics = {
        "architecture": [2, width, width, rank],
        "activation": "tanh",
        "optimizer": "L-BFGS",
        "l2_weight": L2_WEIGHT,
        "loss_multipliers": LOSS_MULTIPLIERS,
        "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
        "closure_evaluations": len(history),
        "elapsed_seconds": elapsed_seconds,
        "final_losses": history[-1],
        "train_standardized_coefficient_rmse": float(
            torch.sqrt(torch.mean((standardized_train - train_targets) ** 2))
        ),
        "validation_standardized_coefficient_rmse": float(
            torch.sqrt(torch.mean((standardized_validation - validation_targets) ** 2))
        ),
        "train_field_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, train_field_error))
        ),
        "validation_field_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, validation_field_error))
        ),
        "train_boundary_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, train_boundary_error))
        ),
        "validation_boundary_relative_l2": dict(
            zip(("u", "w", "p", "nut"), map(float, validation_boundary_error))
        ),
        "train_physics_residual": residual_summary(train_div, train_mx, train_mz),
        "validation_physics_residual": residual_summary(val_div, val_mx, val_mz),
        "train_negative_nut_fraction": float(torch.mean((train_nut < 0).to(DTYPE))),
        "validation_negative_nut_fraction": float(torch.mean((val_nut < 0).to(DTYPE))),
        "validation_pressure_coefficient_mae": {
            "Cl_pressure": float(pressure_coefficient_mae[0]),
            "Cd_pressure": float(pressure_coefficient_mae[1]),
        },
        "validation_pressure_coefficients": {
            case_id: {
                "Cl_pressure_true": float(truth[0]),
                "Cd_pressure_true": float(truth[1]),
                "Cl_pressure_predicted": float(prediction[0]),
                "Cd_pressure_predicted": float(prediction[1]),
            }
            for case_id, truth, prediction in zip(
                validation_ids, true_pressure_coefficients, predicted_pressure_coefficients
            )
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pod_pinn_2x{width}_rank25"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": metrics["architecture"],
            "parameter_center": parameter_center_np,
            "parameter_scale": parameter_scale_np,
            "coefficient_mean": coefficient_mean_np,
            "coefficient_scale": coefficient_scale_np,
        },
        MODEL_DIR / f"{stem}.pt",
    )
    (MODEL_DIR / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (MODEL_DIR / f"{stem}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(
        MODEL_DIR / f"{stem}_predictions.npz",
        train_coefficients=predicted_train_coefficients_np,
        validation_coefficients=predicted_validation_coefficients_np,
        train_boundary=predicted_train_boundary_np.astype(np.float32),
        validation_boundary=predicted_validation_boundary_np.astype(np.float32),
        validation_pressure_coefficients=predicted_pressure_coefficients,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=ARCHITECTURES, action="append")
    args = parser.parse_args()
    widths = args.width or list(ARCHITECTURES)
    for width in widths:
        train(width)


if __name__ == "__main__":
    main()
