"""Download AirfRANS and create an auditable NACA 0012 case catalogue.

The script intentionally does not resample unstructured point clouds. That is
the next controlled preprocessing step, because a common grid and airfoil mask
must be reviewed before POD is calculated.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import airfrans as af
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "airfrans"
MANIFEST_ROOT = ROOT / "data" / "manifests"
KINEMATIC_VISCOSITY = 1.56e-5  # m^2/s, documented AirfRANS generation value
CHORD_M = 1.0
NACA0012_SUFFIX = re.compile(r"_0(?:\\.0+)?_0(?:\\.0+)?_12(?:\\.0+)?$")


def parse_case_name(name: str) -> tuple[float, float]:
    """Return inlet-speed magnitude and AoA from an AirfRANS simulation name."""
    parts = name.split("_")
    if len(parts) < 7 or parts[0] != "airFoil2D":
        raise ValueError(f"Unexpected AirfRANS case name: {name}")
    return float(parts[2]), float(parts[3])


def evenly_spaced_indices(count: int, requested: int) -> np.ndarray:
    """Return ordered, unique indices spanning an already AoA/Re-sorted table."""
    return np.unique(np.linspace(0, count - 1, min(count, requested), dtype=int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download before cataloguing")
    parser.add_argument("--target-cases", type=int, default=40)
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    if args.download:
        af.dataset.download(root=str(DATA_ROOT), file_name="Dataset", unzip=True, OpenFOAM=False)

    # Loading both sides of the documented 'full' split exposes the complete set.
    _, train_names = af.dataset.load(root=str(DATA_ROOT), task="full", train=True)
    _, test_names = af.dataset.load(root=str(DATA_ROOT), task="full", train=False)
    names = sorted(set(train_names) | set(test_names))
    naca_names = [name for name in names if NACA0012_SUFFIX.search(name)]
    if not naca_names:
        raise RuntimeError(
            "No NACA 0012 names found. Check the dataset root or download it with --download."
        )

    rows = []
    for name in naca_names:
        speed, aoa = parse_case_name(name)
        rows.append({
            "case_name": name,
            "airfoil": "NACA0012",
            "aoa_deg": aoa,
            "inlet_speed_m_s": speed,
            "reynolds_number": speed * CHORD_M / KINEMATIC_VISCOSITY,
            "source_split": "full",
        })
    catalog = pd.DataFrame(rows).sort_values(["aoa_deg", "reynolds_number", "case_name"])
    catalog.to_csv(MANIFEST_ROOT / "naca0012_catalog.csv", index=False)

    # This is a candidate set, not a train/test split. Review its domain coverage first.
    selected = catalog.iloc[evenly_spaced_indices(len(catalog), args.target_cases)].copy()
    selected.insert(0, "candidate_rank", range(1, len(selected) + 1))
    selected.to_csv(MANIFEST_ROOT / "naca0012_candidate_cases.csv", index=False)
    print(f"Catalogued {len(catalog)} NACA 0012 cases.")
    print(f"Wrote {len(selected)} candidate cases to {MANIFEST_ROOT}.")


if __name__ == "__main__":
    main()

