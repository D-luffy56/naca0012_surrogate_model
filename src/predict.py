"""Predict the baseline NACA 0012 flow field and companion CL/CD model.

Load joblib files only from trusted sources. The model artifacts are tied to the
checksums in data/manifests/artifacts.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def predict(reynolds: float, aoa_deg: float, model_dir: Path) -> dict[str, object]:
    if not 2_000_000 <= reynolds <= 6_000_000 or not -4 <= aoa_deg <= 14:
        raise ValueError("Inputs must be within the training domain: Re 2M–6M, AoA -4°–14°")
    basis = np.load(model_dir / "pod_basis.npz", allow_pickle=False)
    field_model = joblib.load(model_dir / "pod_mlp.joblib")
    aero_model = joblib.load(model_dir / "aero_coefficients_mlp.joblib")
    inputs = pd.DataFrame({"Re": [reynolds], "AoA_deg": [aoa_deg]})

    coefficients = field_model["y_scaler"].inverse_transform(
        field_model["model"].predict(field_model["x_scaler"].transform(inputs))
    )
    scaled_state = basis["mean_state"] + coefficients @ basis["modes"]
    field_values = scaled_state.reshape(3, -1) * basis["field_scale"][:, None]
    mask = basis["fluid_mask"]
    fields = np.full((3, *mask.shape), np.nan, dtype=float)
    fields[:, mask] = field_values

    cl_cd = aero_model["y_scaler"].inverse_transform(
        aero_model["model"].predict(aero_model["x_scaler"].transform(inputs))
    )[0]
    return {"x": basis["x"], "z": basis["z"], "fluid_mask": mask,
            "fields": fields, "coefficients": coefficients[0],
            "Cl": float(cl_cd[0]), "Cd": float(cl_cd[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=float, required=True, help="Reynolds number")
    parser.add_argument("--aoa", type=float, required=True, help="angle of attack in degrees")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "baseline-v1")
    parser.add_argument("--output", type=Path, help="optional NPZ file for x, z, mask, and u/w/p fields")
    args = parser.parse_args()
    result = predict(args.re, args.aoa, args.model_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, x=result["x"], z=result["z"],
                            fluid_mask=result["fluid_mask"], fields=result["fields"],
                            coefficients=result["coefficients"], Cl=result["Cl"], Cd=result["Cd"])
    print(json.dumps({"Re": args.re, "AoA_deg": args.aoa, "Cl": result["Cl"],
                      "Cd": result["Cd"], "grid_shape": list(result["fields"].shape[1:]),
                      "field_order": ["u", "w", "p"], "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
