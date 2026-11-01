from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "NACA0012_PINN_Study" / "data" / "native"
OUT = ROOT / "artifacts" / "pod"
FIELD_NAMES = ("u", "w", "p", "nut")
FIXED_RANK = 25


def load_case(case_id: str) -> tuple[np.ndarray, dict]:
    with np.load(DATA / f"{case_id}.npz", allow_pickle=False) as archive:
        fields = archive["fields"][:, :4].astype(np.float64, copy=False)
        metadata = json.loads(archive["metadata"].item())
    return fields, metadata


def nondimensionalize(fields: np.ndarray, metadata: dict) -> np.ndarray:
    u_inf = float(metadata["Uinf_mps"])
    chord = float(metadata["chord_m"])
    scales = np.array([u_inf, u_inf, u_inf**2, u_inf * chord])
    return fields / scales


def relative_l2(reference: np.ndarray, estimate: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = np.sum(weights[None, :, None] * (estimate - reference) ** 2, axis=(0, 1))
    denominator = np.sum(weights[None, :, None] * reference**2, axis=(0, 1))
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    records = {record["id"]: record for record in manifest}
    train_ids = sorted(case_id for case_id in records if case_id.startswith("T"))
    validation_ids = sorted(case_id for case_id in records if case_id.startswith("V"))
    if len(train_ids) != 40 or len(validation_ids) != 8:
        raise RuntimeError("Expected exactly 40 training and 8 validation cases")
    if len({records[case_id]["mesh_sha256"] for case_id in records}) != 1:
        raise RuntimeError("POD requires one common mesh; multiple mesh hashes were found")

    with np.load(DATA / "geometry.npz", allow_pickle=False) as geometry:
        xz = geometry["xz"].astype(np.float64)
        volumes = geometry["volumes"].astype(np.float64)
    if not np.isfinite(xz).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise RuntimeError("Invalid native geometry or cell volumes")
    weights = volumes / volumes.sum()
    sqrt_weights = np.sqrt(weights)

    train = np.stack([nondimensionalize(*load_case(case_id)) for case_id in train_ids])
    validation = np.stack([nondimensionalize(*load_case(case_id)) for case_id in validation_ids])
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise RuntimeError("Non-finite field values found")

    mean = train.mean(axis=0)
    fluctuations = train - mean[None, :, :]
    field_rms = np.sqrt(
        np.sum(weights[None, :, None] * fluctuations**2, axis=(0, 1)) / len(train_ids)
    )
    if np.any(field_rms <= 0):
        raise RuntimeError(f"Non-positive field RMS scale: {field_rms}")

    normalized = fluctuations / field_rms[None, None, :]
    weighted_snapshots = (normalized * sqrt_weights[None, :, None]).reshape(len(train_ids), -1)
    u_case, singular_values, vt = np.linalg.svd(weighted_snapshots, full_matrices=False)
    energy = singular_values**2
    cumulative_energy = np.cumsum(energy) / energy.sum()
    numerical_rank = int(np.sum(singular_values > singular_values[0] * 1e-12))
    if FIXED_RANK > numerical_rank:
        raise RuntimeError(f"Fixed rank {FIXED_RANK} exceeds numerical rank {numerical_rank}")

    basis = vt[:FIXED_RANK]
    train_coefficients = weighted_snapshots @ basis.T
    train_reconstructed_weighted = train_coefficients @ basis
    train_reconstructed_normalized = (
        train_reconstructed_weighted.reshape(len(train_ids), len(xz), len(FIELD_NAMES))
        / sqrt_weights[None, :, None]
    )
    train_reconstructed = mean[None, :, :] + train_reconstructed_normalized * field_rms[None, None, :]

    validation_normalized = (validation - mean[None, :, :]) / field_rms[None, None, :]
    validation_weighted = (
        validation_normalized * sqrt_weights[None, :, None]
    ).reshape(len(validation_ids), -1)
    validation_coefficients = validation_weighted @ basis.T
    validation_reconstructed_weighted = validation_coefficients @ basis
    validation_reconstructed_normalized = (
        validation_reconstructed_weighted.reshape(len(validation_ids), len(xz), len(FIELD_NAMES))
        / sqrt_weights[None, :, None]
    )
    validation_reconstructed = (
        mean[None, :, :] + validation_reconstructed_normalized * field_rms[None, None, :]
    )

    train_error = relative_l2(train, train_reconstructed, weights)
    validation_error = relative_l2(validation, validation_reconstructed, weights)
    minimum_rank_999 = int(np.searchsorted(cumulative_energy, 0.999) + 1)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "pod_basis_rank25.npz",
        xz=xz,
        weights=weights,
        mean=mean,
        field_rms=field_rms,
        basis=basis,
        singular_values=singular_values,
        cumulative_energy=cumulative_energy,
        train_coefficients=train_coefficients,
        validation_coefficients=validation_coefficients,
        train_ids=np.array(train_ids),
        validation_ids=np.array(validation_ids),
    )
    summary = {
        "fixed_rank": FIXED_RANK,
        "numerical_rank": numerical_rank,
        "minimum_rank_for_99.9_percent_energy": minimum_rank_999,
        "energy_at_fixed_rank": float(cumulative_energy[FIXED_RANK - 1]),
        "field_rms_scales": dict(zip(FIELD_NAMES, map(float, field_rms))),
        "train_projection_relative_l2": dict(zip(FIELD_NAMES, map(float, train_error))),
        "validation_oracle_projection_relative_l2": dict(
            zip(FIELD_NAMES, map(float, validation_error))
        ),
        "mesh_sha256": records[train_ids[0]]["mesh_sha256"],
        "training_cases": train_ids,
        "validation_cases": validation_ids,
    }
    (OUT / "pod_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
