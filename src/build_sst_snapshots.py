"""Build wall-aware, native-mesh SST snapshots from verified CFD cases.

This is a data preparation step, not a trained PINN. The original CFD files are
read without modification. Each case is sampled on the same collapsed x-z mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


FIELD_NAMES = ("u", "w", "p", "k", "omega", "nut")
WALL_PATCHES = ("pressure", "suction")


def case_path(value: str) -> Path:
    if value.startswith("/mnt/d/"):
        return Path("D:/" + value.removeprefix("/mnt/d/"))
    return Path(value)


def verified_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "case_path", "latest_time", "Re", "AoA_deg", "Uinf_mps", "mesh_sha256"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame = frame[(frame.converged.astype(str).str.lower() == "true")
                  & (frame.status == "outputs_verified")].copy()
    if frame.empty:
        raise ValueError(f"No verified CFD cases in {path}")
    return frame.sort_values("id").reset_index(drop=True)


def export_dir(row: pd.Series) -> Path:
    folder = case_path(row.case_path) / "VTK" / f"{row.id}_{row.latest_time}"
    if not (folder / "internal.vtu").is_file():
        raise FileNotFoundError(folder / "internal.vtu")
    return folder


def extract_values(data: dict[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    required = {"U", "p", "k", "omega", "nut"}
    if missing := required - set(data):
        raise ValueError(f"CFD export is missing SST fields: {sorted(missing)}")
    velocity = np.asarray(data["U"])
    arrays = (velocity[:, 0], velocity[:, 2], *(np.asarray(data[key]) for key in ("p", "k", "omega", "nut")))
    result = np.stack(arrays).astype(np.float32)
    return result if indices is None else result[:, indices]


def build(train_manifest: Path, validation_manifest: Path, output: Path) -> dict:
    train = verified_manifest(train_manifest)
    validation = verified_manifest(validation_manifest)
    if set(train.id) & set(validation.id):
        raise ValueError("Training and validation case IDs overlap")
    combined = pd.concat((train, validation), ignore_index=True)
    if combined.mesh_sha256.nunique() != 1:
        raise ValueError("Cases do not share one CFD mesh")

    first = export_dir(train.iloc[0])
    mesh = meshio.read(first / "internal.vtu")
    points, unique_index = np.unique(mesh.points[:, [0, 2]], axis=0, return_index=True)
    wall_xyz = []
    for name in WALL_PATCHES:
        surface = pv.read(first / "boundary" / f"{name}.vtp")
        wall_xyz.append(np.asarray(surface.points)[:, [0, 2]])
    wall_points = np.unique(np.concatenate(wall_xyz), axis=0)
    wall_tree = cKDTree(wall_points)
    wall_distance, wall_nearest = wall_tree.query(points)
    # Match geometric nodes to exported wall nodes at float32 VTK precision.
    wall_index = np.flatnonzero(wall_distance < 1e-6)
    if len(wall_index) < 100:
        raise ValueError("Unable to identify the airfoil wall on the native mesh")
    wall_distance[wall_index] = 0.0

    snapshots = []
    wall_values = []
    for _, row in combined.iterrows():
        folder = export_dir(row)
        current = meshio.read(folder / "internal.vtu")
        if current.points.shape != mesh.points.shape or not np.allclose(
            current.points[:, [0, 2]], mesh.points[:, [0, 2]], rtol=0, atol=1e-7
        ):
            raise ValueError(f"Mesh coordinate mismatch for {row.id}")
        values = extract_values(current.point_data, unique_index)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite CFD values in {row.id}")
        boundary_coordinates = []
        boundary_fields = []
        for name in WALL_PATCHES:
            surface = pv.read(folder / "boundary" / f"{name}.vtp")
            boundary_coordinates.append(np.asarray(surface.points)[:, [0, 2]])
            boundary_fields.append(extract_values(surface.point_data))
        boundary_coordinates = np.concatenate(boundary_coordinates)
        boundary_fields = np.concatenate(boundary_fields, axis=1)
        gap, boundary_index = cKDTree(boundary_coordinates).query(points[wall_index])
        if np.max(gap) > 1e-6:
            raise ValueError(f"Wall surface does not match internal mesh for {row.id}")
        snapshots.append(values)
        wall_values.append(boundary_fields[:, boundary_index])
    fields = np.stack(snapshots)
    wall_fields = np.stack(wall_values)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "native_sst_snapshots.npz",
        fields=fields, points_xz=points.astype(np.float32),
        wall_points_xz=points[wall_index].astype(np.float32),
        wall_index=wall_index, wall_distance=wall_distance.astype(np.float32),
        nearest_wall_index=wall_nearest.astype(np.int32),
        wall_fields=wall_fields, field_names=np.array(FIELD_NAMES),
        ids=combined.id.to_numpy(dtype=str),
        reynolds=combined.Re.to_numpy(dtype=np.float32),
        aoa_deg=combined.AoA_deg.to_numpy(dtype=np.float32),
        uinf_mps=combined.Uinf_mps.to_numpy(dtype=np.float32),
        train_count=np.array(len(train)),
    )
    summary = {
        "training_cases": len(train), "validation_cases": len(validation),
        "native_xz_points": len(points), "airfoil_wall_points": len(wall_index),
        "fields": list(FIELD_NAMES), "mesh_sha256": str(combined.mesh_sha256.iloc[0]),
        "note": "Raw native-mesh CFD snapshots; no model has been trained by this step.",
    }
    (output / "native_sst_snapshot_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.training_manifest, args.validation_manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
