"""Retrain the published companion CL/CD MLP from committed CFD coefficient tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "aerodynamic_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "aero_retrained")
    args = parser.parse_args()
    train = pd.read_csv(TABLES / "training_aero_coefficients.csv")
    valid = pd.read_csv(TABLES / "validation_engineering_metrics.csv")
    if len(train) != 40 or len(valid) != 7:
        raise ValueError("Expected the published 40 training and 7 validation cases")
    inputs = ["Re", "AoA_deg"]
    targets = ["Cl_cfd", "Cd_cfd"]
    x_scaler = StandardScaler().fit(train[inputs])
    y_scaler = StandardScaler().fit(train[targets])
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                         solver="lbfgs", max_iter=5000, random_state=42)
    model.fit(x_scaler.transform(train[inputs]), y_scaler.transform(train[targets]))
    predicted = y_scaler.inverse_transform(model.predict(x_scaler.transform(valid[inputs])))
    result = {
        "validation_cases": len(valid),
        "Cl_mse": float(np.mean((predicted[:, 0] - valid["Cl_cfd"]) ** 2)),
        "Cd_mse": float(np.mean((predicted[:, 1] - valid["Cd_cfd"]) ** 2)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler},
                args.output_dir / "aero_coefficients_mlp.joblib")
    (args.output_dir / "validation_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
