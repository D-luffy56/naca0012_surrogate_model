"""Train a coefficient-query DeepONet with the existing steady SST residuals.

The physics residual uses a differentiable graph least-squares discretization.
It approximates, but does not reproduce, OpenFOAM's finite-volume operators or
its nonlinear omega wall function. Wall values are supervised from CFD exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from coefficient_deeponet import CoefficientDeepONet, fit_coefficients, weight_penalty
from sst_deeponet_physics import make_residuals


def graph_derivatives(mesh_path: Path, points: np.ndarray, wall_distance: np.ndarray):
    """First derivatives use only actual mesh-cell neighbours, never across solid."""
    mesh = meshio.read(mesh_path)
    native, inverse = np.unique(mesh.points[:, [0, 2]], axis=0, return_inverse=True)
    if native.shape != points.shape or not np.allclose(native, points, rtol=0, atol=1e-7):
        raise ValueError('Derivative mesh does not match snapshot coordinates/order')
    neighbours = [set() for _ in points]
    for block in mesh.cells:
        for cell in block.data:
            ids = np.unique(inverse[cell])
            for index in ids:
                neighbours[index].update(int(other) for other in ids if other != index)
    rows, cols, gx, gz, good = [], [], [], [], np.zeros(len(points), bool)
    for index, first in enumerate(neighbours):
        second = set(first)
        for other in first:
            second.update(neighbours[other])
        second.discard(index)
        ids = np.fromiter(second, dtype=np.int32)
        if len(ids) < 6:
            continue
        delta = points[ids] - points[index]
        radius = np.linalg.norm(delta, axis=1)
        scale = np.median(radius)
        if scale <= 0:
            continue
        A = delta / scale
        weight = 1 / np.maximum(radius / scale, 0.2)
        weighted = A * weight[:, None]
        if np.linalg.cond(weighted.T @ weighted) > 1e5:
            continue
        fit = np.linalg.pinv(weighted) * weight[None, :] / scale
        for axis, values in enumerate((fit[0], fit[1])):
            target = gx if axis == 0 else gz
            target.extend(values.tolist())
        rows.extend([index] * (len(ids) + 1))
        cols.extend(ids.tolist() + [index])
        gx.append(-fit[0].sum())
        gz.append(-fit[1].sum())
        good[index] = wall_distance[index] > 2 * scale
    shape = (len(points), len(points))
    Dx = csr_matrix((gx, (rows, cols)), shape=shape, dtype=np.float32)
    Dz = csr_matrix((gz, (rows, cols)), shape=shape, dtype=np.float32)
    # A second graph derivative is only used where both neighbouring stencils exist.
    valid = good & (np.asarray(Dx.getnnz(axis=1)) > 0) & (np.asarray(Dz.getnnz(axis=1)) > 0)
    return Dx, Dz, valid


def as_torch_sparse(matrix: csr_matrix) -> torch.Tensor:
    matrix = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor(np.stack((matrix.row, matrix.col)), dtype=torch.long),
        torch.tensor(matrix.data, dtype=torch.float32), matrix.shape, check_invariants=True,
    ).coalesce()



def relative_errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    names = ("u", "w", "p", "k", "omega", "nut")
    return {name: float(np.linalg.norm(prediction[:, i] - truth[:, i]) /
                        max(np.linalg.norm(truth[:, i]), 1e-12)) for i, name in enumerate(names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physics-weight", type=float, default=1e-4)
    parser.add_argument("--wall-weight", type=float, default=0.1)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--hidden-units", type=int, nargs=2, default=(32, 32))
    parser.add_argument("--activation", choices=("relu", "tanh"), default="tanh")
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="lbfgs")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument('--latent', type=int, default=32)
    parser.add_argument('--adam-steps', type=int, default=1500)
    parser.add_argument('--lbfgs-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    with np.load(args.snapshots) as saved:
        raw = saved["fields"].copy()
        wall_raw = saved["wall_fields"].copy()
        points = saved["points_xz"].copy()
        wall_index = saved["wall_index"].copy()
        wall_distance = saved["wall_distance"].copy()
        train_count = int(saved["train_count"])
        uinf = saved["uinf_mps"].copy()
        reynolds = saved["reynolds"].copy()
        aoa = saved["aoa_deg"].copy()
        ids = saved["ids"].copy()
    if train_count != 40 or len(raw) - train_count != 7:
        raise ValueError("Expected 40 training and seven validation cases")
    if min(args.epochs, args.batch_size, args.modes) < 1:
        raise ValueError("Epochs, batch size and modes must be positive")
    if min(args.hidden_units) < 1 or args.l2 < 0 or args.latent < 1 or min(args.adam_steps, args.lbfgs_steps, args.physics_weight, args.wall_weight) < 0:
        raise ValueError("Invalid architecture, step count, or loss weight")
    mesh_path = args.mesh_path
    Dx, Dz, valid = graph_derivatives(mesh_path, points, wall_distance)
    if valid.sum() < 1000:
        raise ValueError("Too few valid native-mesh derivative stencils")
    dx, dz = as_torch_sparse(Dx), as_torch_sparse(Dz)
    valid_t = torch.from_numpy(valid)
    wall_t = torch.from_numpy(wall_index.astype(np.int64))
    n = len(points)

    # Chord is one metre. OpenFOAM p is kinematic pressure; omega is 1/s.
    units = np.stack((uinf, uinf, uinf**2, uinf**2, uinf, uinf), axis=1)
    q = raw / units[:, :, None]
    wall_q = wall_raw / units[:, :, None]
    field_scale = q[:train_count].std(axis=(0, 2))
    field_scale = np.maximum(field_scale, 1e-7)
    scaled = (q / field_scale[None, :, None]).reshape(len(q), -1)
    mean = scaled[:train_count].mean(axis=0)
    centered = scaled[:train_count] - mean
    # Method of snapshots keeps SVD small: 40 x 40 instead of 40 x 231660.
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    rank = min(args.modes, train_count - 1)
    modes = (eigenvectors[:, order[:rank]].T @ centered) / np.sqrt(eigenvalues[:rank, None].clip(1e-10))
    targets = centered @ modes.T
    x = np.stack((reynolds, aoa), axis=1)
    x_scale = StandardScaler().fit(x[:train_count])
    y_scale = StandardScaler().fit(targets)
    net, pretrain_history = fit_coefficients(
        x_scale.transform(x[:train_count]), y_scale.transform(targets),
        args.hidden_units, args.activation, args.latent, args.seed,
        args.adam_steps, args.lbfgs_steps, args.l2)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output / 'deeponet_data.pt')
    pd.DataFrame(pretrain_history).to_csv(output / 'pretrain_history.csv', index=False)
    with torch.no_grad():
        base_std = net(torch.tensor(x_scale.transform(x), dtype=torch.float32)).numpy()
    base_coef = y_scale.inverse_transform(base_std)
    base = ((mean + base_coef @ modes).reshape(len(q), 6, n) * field_scale[None, :, None])
    optimizer = (torch.optim.LBFGS(net.parameters(), lr=0.1, max_iter=1,
                                  history_size=20, line_search_fn=None)
                 if args.optimizer == "lbfgs" else
                 torch.optim.Adam(net.parameters(), lr=1e-4))
    input_t = torch.tensor(x_scale.transform(x), dtype=torch.float32)
    target_t = torch.tensor(y_scale.transform(targets), dtype=torch.float32)
    truth_t = torch.tensor(q, dtype=torch.float32)
    wall_truth_t = torch.tensor(wall_q, dtype=torch.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    modes_t = torch.tensor(modes, dtype=torch.float32)
    scale_t = torch.tensor(field_scale, dtype=torch.float32)
    coef_mean_t = torch.tensor(y_scale.mean_, dtype=torch.float32)
    coef_scale_t = torch.tensor(y_scale.scale_, dtype=torch.float32)
    re_t = torch.tensor(reynolds, dtype=torch.float32)

    def decode(standardized):
        coef = standardized * coef_scale_t + coef_mean_t
        return (mean_t + coef @ modes_t).reshape(-1, 6, n) * scale_t[None, :, None]

    residuals = make_residuals(dx, dz, wall_distance)

    # CFD-consistent reference scales prevent omega from dominating simply by units.
    with torch.no_grad():
        reference = residuals(truth_t[:train_count], re_t[:train_count])
    residual_scale = {key: float(value[:, valid_t].square().mean().sqrt().clamp_min(1e-5))
                      for key, value in reference.items()}
    weights = {"continuity": 1, "momentum_x": 1, "momentum_z": 1,
               "k_transport": 1, "omega_transport": 1, "nut_closure": 1}
    def batch_losses(indices):
        prediction_std = net(input_t[indices])
        predicted = decode(prediction_std)
        data = F.mse_loss(prediction_std, target_t[indices]) + F.mse_loss(
            predicted / scale_t[None, :, None], truth_t[indices] / scale_t[None, :, None])
        wall = F.mse_loss(predicted[:, :, wall_t] / scale_t[None, :, None],
                          wall_truth_t[indices] / scale_t[None, :, None])
        terms = residuals(predicted, re_t[indices])
        physics = sum(weights[key] * (value[:, valid_t] / residual_scale[key]).square().mean()
                      for key, value in terms.items()) / sum(weights.values())
        return data, wall, physics

    def l2_loss():
        # Match sklearn's alpha/(2*n_samples) convention; biases are excluded.
        return args.l2 / (2 * train_count) * weight_penalty(net)

    history = []
    for epoch in range(1, args.epochs + 1):
        totals = {key: 0.0 for key in ("data", "wall", "physics")}
        if args.optimizer == "lbfgs":
            # Fixed full-training-set objective, accumulated in chunks to limit memory.
            evaluations = 0
            def closure():
                nonlocal evaluations
                evaluations += 1
                optimizer.zero_grad()
                value = torch.zeros((), dtype=torch.float32)
                for indices in torch.arange(train_count).split(args.batch_size):
                    data, wall, physics = batch_losses(indices)
                    part = (data + args.wall_weight * wall + args.physics_weight * physics) * (len(indices) / train_count)
                    part.backward()
                    value = value + part.detach()
                regularizer = l2_loss()
                regularizer.backward()
                if epoch == 1 and evaluations == 1:
                    gradient_norm = float(torch.linalg.vector_norm(torch.cat([
                        parameter.grad.flatten() for parameter in net.parameters()])))
                    print({"initial_full_objective": float(value + regularizer.detach()),
                           "initial_gradient_norm": gradient_norm}, flush=True)
                return value + regularizer.detach()
            before = [parameter.detach().clone() for parameter in net.parameters()]
            optimizer.step(closure)
            if epoch == 1:
                change = float(torch.linalg.vector_norm(torch.cat([
                    (parameter.detach() - old).flatten()
                    for parameter, old in zip(net.parameters(), before)])))
                print({"lbfgs_evaluations": evaluations, "parameter_change_norm": change}, flush=True)
            with torch.no_grad():
                for indices in torch.arange(train_count).split(args.batch_size):
                    values = batch_losses(indices)
                    for key, value in zip(totals, values):
                        totals[key] += float(value) * len(indices)
        else:
            for indices in torch.randperm(train_count).split(args.batch_size):
                optimizer.zero_grad()
                data, wall, physics = batch_losses(indices)
                total = data + args.wall_weight * wall + args.physics_weight * physics + l2_loss()
                total.backward()
                optimizer.step()
                for key, value in (("data", data), ("wall", wall), ("physics", physics)):
                    totals[key] += float(value.detach()) * len(indices)
        record = {"epoch": epoch, **{key: value / train_count for key, value in totals.items()}}
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(record, flush=True)

    with torch.no_grad():
        predicted = torch.cat([decode(net(batch)) for batch in input_t.split(args.batch_size)]).numpy()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "sst_pod_basis.npz", mean=mean, modes=modes, scale=field_scale,
                        x_mean=x_scale.mean_, x_scale=x_scale.scale_,
                        coefficient_mean=y_scale.mean_, coefficient_scale=y_scale.scale_,
                        points_xz=points, wall_index=wall_index,
                        hidden_units=np.asarray(args.hidden_units), activation=args.activation, latent=args.latent, model_type='coefficient_deeponet')
    np.savez_compressed(output / "validation_predictions.npz", ids=ids[train_count:],
                        cfd=q[train_count:], baseline=base[train_count:],
                        sst=predicted[train_count:])
    torch.save(net.state_dict(), output / "deeponet_sst.pt")
    pd.DataFrame(history).to_csv(output / "loss_history.csv", index=False)
    rows = []
    for i in range(train_count, len(q)):
        row = {"id": str(ids[i]), "Re": float(reynolds[i]), "AoA_deg": float(aoa[i])}
        for label, values in (("baseline", base[i:i+1]), ("sst", predicted[i:i+1])):
            row.update({f"{label}_{name}_relative_l2": value for name, value in
                        relative_errors(q[i:i+1], values).items()})
            row[f"{label}_wall_mse"] = float(np.mean((values[:, :, wall_index] - wall_q[i:i+1])**2))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "validation_by_case.csv", index=False)
    summary = {
        "description": "Coefficient DeepONet with approximate steady SST graph residuals",
        "seed": args.seed, "latent": args.latent, "adam_steps": args.adam_steps, "lbfgs_steps": args.lbfgs_steps,
        "limitation": "Least-squares graph derivatives and CFD wall-value loss do not reproduce OpenFOAM finite-volume operators or nonlinear omega wall functions exactly.",
        "training_cases": train_count, "validation_cases": len(q) - train_count,
        "native_points": n, "wall_points": len(wall_index), "physics_points": int(valid.sum()),
        "retained_modes": rank, "epochs": args.epochs, "physics_weight": args.physics_weight,
        "hidden_units": list(args.hidden_units), "activation": args.activation,
        "optimizer": args.optimizer, "l2": args.l2,
        "wall_weight": args.wall_weight, "reference_residual_scale": residual_scale,
        "validation_baseline_relative_l2": relative_errors(q[train_count:], base[train_count:]),
        "validation_sst_relative_l2": relative_errors(q[train_count:], predicted[train_count:]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def predict_sst_case(model_dir: Path, reynolds: float, angle_deg: float, uinf_mps: float | None = None, physics=True):
    """Return dimensional native-mesh fields in u,w,p,k,omega,nut order."""
    if uinf_mps is None:
        # All project cases use chord=1 m and nu=1.56e-5 m^2/s.
        uinf_mps = reynolds * 1.56e-5
    if not np.isfinite([reynolds, angle_deg, uinf_mps]).all() or reynolds <= 0 or uinf_mps <= 0:
        raise ValueError("Reynolds number and freestream speed must be positive")
    with np.load(model_dir / "sst_pod_basis.npz") as basis:
        rank = len(basis["modes"])
        hidden_units = tuple(basis["hidden_units"].tolist()) if "hidden_units" in basis else (64, 64)
        activation = str(basis["activation"]) if "activation" in basis else "relu"
        net = CoefficientDeepONet(rank, hidden_units, activation, int(basis["latent"]))
        net.load_state_dict(torch.load(model_dir / ("deeponet_sst.pt" if physics else "deeponet_data.pt"), map_location="cpu", weights_only=True))
        conditions = np.array([reynolds, angle_deg], dtype=np.float32)
        standardized_input = (conditions - basis["x_mean"]) / basis["x_scale"]
        with torch.no_grad():
            standardized_coef = net(torch.tensor(standardized_input, dtype=torch.float32)).numpy()
        coef = standardized_coef * basis["coefficient_scale"] + basis["coefficient_mean"]
        units = np.array([uinf_mps, uinf_mps, uinf_mps**2, uinf_mps**2,
                          uinf_mps, uinf_mps], dtype=np.float32)
        fields = (basis["mean"] + coef @ basis["modes"]).reshape(6, -1)
        fields = fields * basis["scale"][:, None] * units[:, None]
        return {"coefficients": coef, "fields": fields, "points_xz": basis["points_xz"].copy(),
                "wall_index": basis["wall_index"].copy(),
                "field_names": ("u", "w", "p", "k", "omega", "nut")}


if __name__ == "__main__":
    main()
