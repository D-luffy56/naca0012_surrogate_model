from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_fixed_pod_pinn import (
    CoefficientNetwork,
    load_native_cases,
    pressure_coefficients,
    reconstruct,
    relative_l2,
    residuals,
    tensor,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
POD_FILE = ROOT / "artifacts" / "pod" / "pod_basis_rank25.npz"
OPERATOR_FILE = ROOT / "artifacts" / "operator" / "pod_pinn_operator_rank25.npz"
MODEL_DIR = ROOT / "artifacts" / "models"

WIDTH = 64
RANK = 25
L2_WEIGHT = 1.0e-4
MAX_LBFGS_ITERATIONS = 500
DTYPE = torch.float64
LOSS_NAMES = ("data", "continuity", "momentum", "boundary")


class BoundedUncertaintyWeights(nn.Module):
    """Positive self-adaptive global loss weights with a stable log range.

    The learned log variances s_i enter the objective as
    exp(-s_i) L_i + s_i. The tanh map limits s_i to [-5, 5], preventing
    numerical collapse while retaining a 2.2e4 dynamic range in weights.
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(count, dtype=DTYPE))

    def log_variances(self) -> torch.Tensor:
        return 5.0 * torch.tanh(self.raw / 5.0)

    def weights(self) -> torch.Tensor:
        return torch.exp(-self.log_variances())

    def objective(self, losses: torch.Tensor) -> torch.Tensor:
        s = self.log_variances()
        return torch.sum(torch.exp(-s) * losses + s)


def main() -> None:
    torch.manual_seed(1729 + WIDTH)
    np.random.seed(1729 + WIDTH)
    torch.set_num_threads(min(9, torch.get_num_threads()))

    with np.load(OPERATOR_FILE, allow_pickle=False) as operator:
        packed_np = np.concatenate(
            tuple(
                operator[name]
                for name in (
                    "expansion_values",
                    "expansion_dx",
                    "expansion_dz",
                    "expansion_dxx",
                    "expansion_dxz",
                    "expansion_dzz",
                )
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

    model = CoefficientNetwork(WIDTH, RANK).to(dtype=DTYPE)
    adaptive = BoundedUncertaintyWeights(len(LOSS_NAMES)).to(dtype=DTYPE)
    optimizer = torch.optim.LBFGS(
        list(model.parameters()) + list(adaptive.parameters()),
        lr=1.0,
        max_iter=MAX_LBFGS_ITERATIONS,
        max_eval=int(MAX_LBFGS_ITERATIONS * 1.25),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    def evaluate_losses(inputs: torch.Tensor, targets: torch.Tensor, reynolds: torch.Tensor,
                        reference_boundary: torch.Tensor):
        standardized = model(inputs)
        coefficients = coefficient_mean[None, :] + coefficient_scale[None, :] * standardized
        divergence, momentum_x, momentum_z, nut = residuals(packed, coefficients, reynolds)
        predicted_boundary = boundary_intercept[None, :, :] + torch.einsum(
            "br,rnf->bnf", coefficients, boundary_modes
        )
        components = torch.stack(
            (
                torch.mean((standardized - targets) ** 2),
                torch.mean(divergence**2),
                0.5 * torch.mean(momentum_x**2 + momentum_z**2),
                torch.mean(
                    ((predicted_boundary - reference_boundary) / boundary_scales[None, None, :])
                    ** 2
                ),
            )
        )
        return standardized, coefficients, predicted_boundary, divergence, momentum_x, momentum_z, nut, components

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        outputs = evaluate_losses(train_inputs, train_targets, reynolds_train, train_boundary)
        components = outputs[-1]
        l2_loss = sum(torch.sum(parameter**2) for parameter in model.parameters())
        total = adaptive.objective(components) + L2_WEIGHT * l2_loss
        total.backward()
        current_weights = adaptive.weights().detach().cpu().numpy()
        record = {
            "closure": len(history) + 1,
            "total": float(total.detach()),
            **{name: float(value.detach()) for name, value in zip(LOSS_NAMES, components)},
            **{f"weight_{name}": float(value) for name, value in zip(LOSS_NAMES, current_weights)},
            "l2_unweighted": float(l2_loss.detach()),
        }
        history.append(record)
        if len(history) == 1 or len(history) % 25 == 0:
            print(
                f"closure={len(history)} total={record['total']:.5e} "
                f"losses={[f'{record[n]:.2e}' for n in LOSS_NAMES]} "
                f"weights={[f'{record[f'weight_{n}']:.2f}' for n in LOSS_NAMES]} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        return total

    optimizer.step(closure)
    elapsed_seconds = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        train_outputs = evaluate_losses(train_inputs, train_targets, reynolds_train, train_boundary)
        validation_outputs = evaluate_losses(
            validation_inputs, validation_targets, reynolds_validation, validation_boundary
        )
    (
        standardized_train,
        predicted_train_coefficients,
        predicted_train_boundary,
        train_div,
        train_mx,
        train_mz,
        train_nut,
        train_components,
    ) = train_outputs
    (
        standardized_validation,
        predicted_validation_coefficients,
        predicted_validation_boundary,
        val_div,
        val_mx,
        val_mz,
        val_nut,
        validation_components,
    ) = validation_outputs

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
    train_field_error = relative_l2(
        train_native_reference, native_reconstruction(predicted_train_coefficients_np), native_weights
    )
    validation_field_error = relative_l2(
        validation_native_reference,
        native_reconstruction(predicted_validation_coefficients_np),
        native_weights,
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
    pressure_mae = np.mean(
        np.abs(predicted_pressure_coefficients - true_pressure_coefficients), axis=0
    )

    def residual_summary(div: torch.Tensor, mx: torch.Tensor, mz: torch.Tensor) -> dict:
        return {
            "continuity_rms": float(torch.sqrt(torch.mean(div**2))),
            "momentum_x_rms": float(torch.sqrt(torch.mean(mx**2))),
            "momentum_z_rms": float(torch.sqrt(torch.mean(mz**2))),
        }

    learned_weights = adaptive.weights().detach().numpy()
    learned_log_variances = adaptive.log_variances().detach().numpy()
    metrics = {
        "architecture": [2, WIDTH, WIDTH, RANK],
        "activation": "tanh",
        "optimizer": "L-BFGS",
        "weighting_method": "bounded homoscedastic uncertainty weighting",
        "adaptive_objective": "sum(exp(-s_i) * L_i + s_i), s_i in [-5, 5]",
        "initial_loss_weights": dict.fromkeys(LOSS_NAMES, 1.0),
        "learned_loss_weights": dict(zip(LOSS_NAMES, map(float, learned_weights))),
        "learned_log_variances": dict(zip(LOSS_NAMES, map(float, learned_log_variances))),
        "l2_weight": L2_WEIGHT,
        "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
        "closure_evaluations": len(history),
        "elapsed_seconds": elapsed_seconds,
        "final_train_component_losses": dict(
            zip(LOSS_NAMES, map(float, train_components.detach().numpy()))
        ),
        "validation_component_losses": dict(
            zip(LOSS_NAMES, map(float, validation_components.detach().numpy()))
        ),
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
            "Cl_pressure": float(pressure_mae[0]),
            "Cd_pressure": float(pressure_mae[1]),
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = "pod_pinn_2x64_rank25_self_adaptive"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "adaptive_weight_state_dict": adaptive.state_dict(),
            "architecture": metrics["architecture"],
            "parameter_center": parameter_center_np,
            "parameter_scale": parameter_scale_np,
            "coefficient_mean": coefficient_mean_np,
            "coefficient_scale": coefficient_scale_np,
            "learned_loss_weights": learned_weights,
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


if __name__ == "__main__":
    main()
