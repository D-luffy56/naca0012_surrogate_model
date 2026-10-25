"""Train a native-mesh POD coefficient surrogate with steady SST residuals.

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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

try:
    from .build_sst_snapshots import verified_manifest, export_dir
except ImportError:  # Direct `python src/train_sst_pod.py` execution.
    from build_sst_snapshots import verified_manifest, export_dir


def graph_derivatives(mesh_path: Path, points: np.ndarray, wall_distance: np.ndarray):
    """First derivatives use only actual mesh-cell neighbours, never across solid."""
    mesh = meshio.read(mesh_path)
    _, inverse = np.unique(mesh.points[:, [0, 2]], axis=0, return_inverse=True)
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
        torch.tensor(matrix.data, dtype=torch.float32), matrix.shape,
    ).coalesce()


class CoefficientNet(nn.Module):
    def __init__(self, rank: int, hidden_units=(64, 64), activation="relu"):
        super().__init__()
        nonlinear = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        sizes = (2, *hidden_units, rank)
        self.layers = nn.Sequential(*[
            module for i in range(len(sizes) - 1)
            for module in ([nn.Linear(sizes[i], sizes[i + 1])] +
                           ([nonlinear()] if i < len(sizes) - 2 else []))
        ])

    def forward(self, x):
        return self.layers(x)

    def copy_sklearn(self, model: MLPRegressor):
        with torch.no_grad():
            layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
            for i, layer in enumerate(layers):
                layer.weight.copy_(torch.as_tensor(model.coefs_[i].T, dtype=torch.float32))
                layer.bias.copy_(torch.as_tensor(model.intercepts_[i], dtype=torch.float32))


def relative_errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    names = ("u", "w", "p", "k", "omega", "nut")
    return {name: float(np.linalg.norm(prediction[:, i] - truth[:, i]) /
                        max(np.linalg.norm(truth[:, i]), 1e-12)) for i, name in enumerate(names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physics-weight", type=float, default=1e-4)
    parser.add_argument("--wall-weight", type=float, default=0.1)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--hidden-units", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--diagnose-model-dir", type=Path,
                        help="Compare saved predictions with CFD using this training run's exact SST residual")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
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
    if min(args.hidden_units) < 1 or args.l2 < 0:
        raise ValueError("Hidden units must be positive and L2 must be nonnegative")
    train_manifest = verified_manifest(args.training_manifest)
    mesh_path = export_dir(train_manifest.iloc[0]) / "internal.vtu"
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
    baseline = MLPRegressor(hidden_layer_sizes=tuple(args.hidden_units), activation=args.activation,
                            solver="lbfgs", alpha=args.l2, max_iter=5000,
                            random_state=42).fit(x_scale.transform(x[:train_count]),
                                                                  y_scale.transform(targets))
    base_coef = y_scale.inverse_transform(baseline.predict(x_scale.transform(x)))
    base = ((mean + base_coef @ modes).reshape(len(q), 6, n) * field_scale[None, :, None])

    net = CoefficientNet(rank, args.hidden_units, args.activation)
    net.copy_sklearn(baseline)
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

    def derivative(v, matrix):
        return torch.sparse.mm(matrix, v.T).T

    def decode(standardized):
        coef = standardized * coef_scale_t + coef_mean_t
        return (mean_t + coef @ modes_t).reshape(-1, 6, n) * scale_t[None, :, None]

    def residuals(fields, re):
        # All terms are nondimensionalized with chord=1 and case Uinf.
        u, w, p, k, omega, nut = fields.unbind(1)
        k = k.clamp_min(1e-9)
        omega = omega.clamp_min(1e-8)
        nut = nut.clamp_min(0)
        nu = 1 / re[:, None]
        ux, uz, wx, wz = derivative(u, dx), derivative(u, dz), derivative(w, dx), derivative(w, dz)
        px, pz = derivative(p, dx), derivative(p, dz)
        kx, kz = derivative(k, dx), derivative(k, dz)
        ox, oz = derivative(omega, dx), derivative(omega, dz)
        div = ux + wz
        s2 = 2 * ux.square() + 2 * wz.square() + (uz + wx).square()
        g0 = (s2 - 2 / 3 * div.square()).clamp_min(0)
        y = torch.tensor(wall_distance, dtype=torch.float32)[None, :].clamp_min(1e-6)
        beta_star, a1, b1, c1 = 0.09, 0.31, 1.0, 10.0
        alpha_k1, alpha_k2 = 0.85, 1.0
        alpha_o1, alpha_o2 = 0.5, 0.856
        gamma1, gamma2 = 5 / 9, 0.44
        beta1, beta2 = 0.075, 0.0828
        cd = 2 * alpha_o2 * (kx * ox + kz * oz) / omega
        arg1 = torch.minimum(torch.maximum(torch.sqrt(k) / (beta_star * omega * y),
                          500 * nu / (y.square() * omega)),
                          4 * alpha_o2 * k / (cd.clamp_min(1e-10) * y.square()))
        f1 = torch.tanh(arg1.clamp_max(10).pow(4))
        arg2 = torch.maximum(2 * torch.sqrt(k) / (beta_star * omega * y),
                             500 * nu / (y.square() * omega))
        f2 = torch.tanh(arg2.clamp_max(100).square())
        blend = lambda first, second: f1 * first + (1 - f1) * second
        alpha_k = blend(alpha_k1, alpha_k2)
        alpha_o = blend(alpha_o1, alpha_o2)
        gamma, beta = blend(gamma1, gamma2), blend(beta1, beta2)
        closure = a1 * k / torch.maximum(a1 * omega, b1 * f2 * torch.sqrt(s2.clamp_min(0)))
        g = nut * g0
        pk = torch.minimum(g, c1 * beta_star * k * omega)
        g_by_nu = torch.minimum(g0, c1 / a1 * beta_star * omega *
                                torch.maximum(a1 * omega, b1 * f2 * torch.sqrt(s2.clamp_min(0))))
        effective = nu + nut
        tau_xx = effective * (2 * ux - 2 / 3 * div)
        tau_zz = effective * (2 * wz - 2 / 3 * div)
        tau_xz = effective * (uz + wx)
        rx = u * ux + w * uz + px - derivative(tau_xx, dx) - derivative(tau_xz, dz)
        rz = u * wx + w * wz + pz - derivative(tau_xz, dx) - derivative(tau_zz, dz)
        dk = nu + alpha_k * nut
        do = nu + alpha_o * nut
        rk = u * kx + w * kz - derivative(dk * kx, dx) - derivative(dk * kz, dz) \
             - pk + 2 / 3 * div * k + beta_star * omega * k
        ro = u * ox + w * oz - derivative(do * ox, dx) - derivative(do * oz, dz) \
             - gamma * g_by_nu + 2 / 3 * gamma * div * omega + beta * omega.square() \
             - 2 * (1 - f1) * alpha_o2 * (kx * ox + kz * oz) / omega
        return {
            "continuity": div, "momentum_x": rx, "momentum_z": rz,
            "k_transport": rk, "omega_transport": ro,
            "nut_closure": nut - closure,
        }

    # CFD-consistent reference scales prevent omega from dominating simply by units.
    with torch.no_grad():
        reference = residuals(truth_t[:train_count], re_t[:train_count])
    residual_scale = {key: float(value[:, valid_t].square().mean().sqrt().clamp_min(1e-5))
                      for key, value in reference.items()}
    if args.diagnose_model_dir is not None:
        model_dir = args.diagnose_model_dir
        with np.load(model_dir / "validation_predictions.npz") as saved:
            if not np.array_equal(saved["ids"], ids[train_count:]):
                raise ValueError("Saved validation predictions have different case IDs")
            predicted_sets = {name: saved[name].copy() for name in ("cfd", "baseline", "sst")}
        if not np.allclose(predicted_sets["cfd"], q[train_count:], rtol=1e-5, atol=1e-6):
            raise ValueError("Saved CFD targets differ from the native snapshots")
        regions = {
            "near_wall_0_to_0.05_chord": wall_distance <= 0.05,
            "wake_1_to_3_chord": (points[:, 0] >= 1) & (points[:, 0] <= 3)
                                & (np.abs(points[:, 1]) <= 0.5),
            "other_fluid": (wall_distance > 0.05) & ~((points[:, 0] >= 1)
                           & (points[:, 0] <= 3) & (np.abs(points[:, 1]) <= 0.5)),
        }
        rows = []
        aggregate = {"training_cfd_residual_scale": residual_scale,
                     "valid_physics_points": int(valid.sum()),
                     "excluded_physics_points": int((~valid).sum()),
                     "region_points": {key: int(mask.sum()) for key, mask in regions.items()},
                     "residual_rms": {}, "normalized_physics_score": {},
                     "regional_relative_l2": {}, "pod_projection_relative_l2": {}}
        with np.load(model_dir / "sst_pod_basis.npz") as saved:
            diagnostic_hidden = tuple(saved["hidden_units"].tolist()) if "hidden_units" in saved else (64, 64)
            diagnostic_activation = str(saved["activation"]) if "activation" in saved else "relu"
        net = CoefficientNet(rank, diagnostic_hidden, diagnostic_activation)
        net.load_state_dict(torch.load(model_dir / "sst_pod_mlp.pt", map_location="cpu", weights_only=True))
        with torch.no_grad():
            trained_fields = torch.cat([decode(net(batch)) for batch in input_t[:train_count].split(4)])
            training_fields = {
                "cfd": truth_t[:train_count],
                "baseline": torch.as_tensor(base[:train_count], dtype=torch.float32),
                "sst": trained_fields,
            }
            aggregate["training_normalized_physics_score"] = {}
            for model_name, model_fields in training_fields.items():
                train_terms = residuals(model_fields, re_t[:train_count])
                aggregate["training_normalized_physics_score"][model_name] = float(np.mean([
                    float((value[:, valid_t] / residual_scale[key]).square().mean())
                    for key, value in train_terms.items()
                ]))
        sample = torch.tensor([0, 13, 26, 39], dtype=torch.long)
        pred_std = net(input_t[sample])
        pred_fields = decode(pred_std)
        diagnostic_losses = {
            "data": F.mse_loss(pred_std, target_t[sample]) + F.mse_loss(
                pred_fields / scale_t[None, :, None],
                truth_t[sample] / scale_t[None, :, None]),
            "wall": args.wall_weight * F.mse_loss(
                pred_fields[:, :, wall_t] / scale_t[None, :, None],
                wall_truth_t[sample] / scale_t[None, :, None]),
        }
        diagnostic_terms = residuals(pred_fields, re_t[sample])
        diagnostic_losses["physics"] = args.physics_weight * sum(
            (value[:, valid_t] / residual_scale[key]).square().mean()
            for key, value in diagnostic_terms.items()
        ) / len(diagnostic_terms)
        gradients = {}
        for name, value in diagnostic_losses.items():
            parts = torch.autograd.grad(value, tuple(net.parameters()), retain_graph=True)
            gradients[name] = torch.cat([part.reshape(-1) for part in parts]).detach()
        aggregate["representative_batch_gradient"] = {
            "case_ids": [str(ids[index]) for index in sample],
            "weighted_loss": {key: float(value.detach()) for key, value in diagnostic_losses.items()},
            "norm": {key: float(value.norm()) for key, value in gradients.items()},
            "cosine_to_data": {
                key: float(F.cosine_similarity(value, gradients["data"], dim=0))
                for key, value in gradients.items() if key != "data"
            },
        }
        validation_truth = q[train_count:]
        for name, fields in predicted_sets.items():
            with torch.no_grad():
                terms = residuals(torch.as_tensor(fields, dtype=torch.float32), re_t[train_count:])
            aggregate["residual_rms"][name] = {
                key: float(value[:, valid_t].square().mean().sqrt()) for key, value in terms.items()
            }
            aggregate["normalized_physics_score"][name] = float(np.mean([
                (aggregate["residual_rms"][name][key] / residual_scale[key])**2
                for key in terms
            ]))
            aggregate["regional_relative_l2"][name] = {}
            for region, mask in regions.items():
                aggregate["regional_relative_l2"][name][region] = relative_errors(
                    validation_truth[:, :, mask], fields[:, :, mask]
                )
            for case, case_id in enumerate(ids[train_count:]):
                row = {"id": str(case_id), "model": name}
                for key, value in terms.items():
                    row[f"{key}_rms"] = float(value[case, valid_t].square().mean().sqrt())
                rows.append(row)
        standardized_validation = (validation_truth / field_scale[None, :, None]).reshape(
            len(validation_truth), -1
        )
        projected_coef = (standardized_validation - mean) @ modes.T
        projected = ((mean + projected_coef @ modes).reshape(len(validation_truth), 6, n)
                     * field_scale[None, :, None])
        aggregate["pod_projection_relative_l2"] = relative_errors(validation_truth, projected)
        history_path = model_dir / "loss_history.csv"
        if history_path.is_file():
            last = pd.read_csv(history_path).iloc[-1]
            aggregate["last_epoch_weighted_loss"] = {
                "data": float(last.data), "wall": float(args.wall_weight * last.wall),
                "physics": float(args.physics_weight * last.physics),
            }
        output = model_dir / "diagnostics"
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output / "validation_residuals_by_case.csv", index=False)
        (output / "summary.json").write_text(json.dumps(aggregate, indent=2))
        print(json.dumps(aggregate, indent=2), flush=True)
        return
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
        return args.l2 / (2 * train_count) * sum(
            layer.weight.square().sum() for layer in net.layers if isinstance(layer, nn.Linear))

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
                        hidden_units=np.asarray(args.hidden_units), activation=args.activation)
    np.savez_compressed(output / "validation_predictions.npz", ids=ids[train_count:],
                        cfd=q[train_count:], baseline=base[train_count:],
                        sst=predicted[train_count:])
    torch.save(net.state_dict(), output / "sst_pod_mlp.pt")
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
        "description": "Native-mesh POD coefficient network with approximate steady SST graph residuals",
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


def predict_sst_case(model_dir: Path, reynolds: float, angle_deg: float, uinf_mps: float | None = None):
    """Return dimensional native-mesh fields in u,w,p,k,omega,nut order."""
    if uinf_mps is None:
        # All project cases use chord=1 m and nu=1.56e-5 m^2/s.
        uinf_mps = reynolds * 1.56e-5
    if reynolds <= 0 or uinf_mps <= 0:
        raise ValueError("Reynolds number and freestream speed must be positive")
    with np.load(model_dir / "sst_pod_basis.npz") as basis:
        rank = len(basis["modes"])
        hidden_units = tuple(basis["hidden_units"].tolist()) if "hidden_units" in basis else (64, 64)
        activation = str(basis["activation"]) if "activation" in basis else "relu"
        net = CoefficientNet(rank, hidden_units, activation)
        net.load_state_dict(torch.load(model_dir / "sst_pod_mlp.pt", map_location="cpu", weights_only=True))
        conditions = np.array([reynolds, angle_deg], dtype=np.float32)
        standardized_input = (conditions - basis["x_mean"]) / basis["x_scale"]
        with torch.no_grad():
            standardized_coef = net(torch.tensor(standardized_input, dtype=torch.float32)).numpy()
        coef = standardized_coef * basis["coefficient_scale"] + basis["coefficient_mean"]
        units = np.array([uinf_mps, uinf_mps, uinf_mps**2, uinf_mps**2,
                          uinf_mps, uinf_mps], dtype=np.float32)
        fields = (basis["mean"] + coef @ basis["modes"]).reshape(6, -1)
        fields = fields * basis["scale"][:, None] * units[:, None]
        return {"fields": fields, "points_xz": basis["points_xz"].copy(),
                "wall_index": basis["wall_index"].copy(),
                "field_names": ("u", "w", "p", "k", "omega", "nut")}


if __name__ == "__main__":
    main()
