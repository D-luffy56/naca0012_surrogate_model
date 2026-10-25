"""Verify published baseline artifacts and regenerate per-case field errors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "baseline-v1"


def main() -> None:
    manifest = json.loads((ROOT / "data" / "manifests" / "artifacts.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Artifact checksum or size mismatch: {path}")

    truth = np.load(PROCESSED / "validation_snapshots.npz", allow_pickle=False)
    pred = np.load(PROCESSED / "validation_predictions.npz", allow_pickle=False)
    ids = truth["ids"].astype(str)
    if not np.array_equal(ids, pred["ids"].astype(str)):
        raise ValueError("Validation IDs are misaligned")
    if truth["fields"].shape != pred["fields"].shape:
        raise ValueError("Validation field shapes differ")
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    for field_index, field in enumerate(("u", "w", "p")):
        actual = truth["fields"][:, field_index]
        estimate = pred["fields"][:, field_index]
        error = float(np.linalg.norm(estimate - actual) / np.linalg.norm(actual))
        if not np.isclose(error, summary["validation_mlp_relative_l2"][field], atol=1e-7):
            raise ValueError(f"Published {field} error differs from processed arrays")
    rows = []
    for index, case_id in enumerate(ids):
        row = {"id": case_id}
        for field_index, field in enumerate(("u", "w", "p")):
            actual = truth["fields"][index, field_index]
            estimate = pred["fields"][index, field_index]
            row[f"{field}_relative_l2"] = float(np.linalg.norm(estimate - actual) / np.linalg.norm(actual))
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = ROOT / "results" / "baseline-v1" / "validation_field_errors.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Verified {len(manifest['files'])} artifacts and wrote {len(frame)} case errors to {output}")


if __name__ == "__main__":
    main()
