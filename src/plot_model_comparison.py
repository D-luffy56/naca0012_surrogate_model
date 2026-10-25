"""Plot historical native-mesh model errors from the committed comparison CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/model_comparison/all_field_errors.csv"
OUTPUT = ROOT / "results/model_comparison/historical_native_accuracy.png"
ORDER = (
    "native POD-MLP 64-64",
    "native SST POD-MLP 64-64",
    "selected 32-32 tanh data-only",
    "selected 32-32 tanh SST",
    "native POD-GPR",
)
LABELS = ("64–64 MLP", "64–64 SST", "32–32 tanh", "32–32 tanh SST", "GPR")
FIELDS = ("u", "w", "p")


def main() -> None:
    data = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, group, title in zip(axes, ("original validation 7", "fresh CFD 7"),
                                 ("Original validation cases", "Later CFD cases")):
        subset = data[(data.evaluation_set == group) &
                      (data.provenance.isin(("historical saved prediction / report",
                                           "historical fresh comparison")))]
        values = np.array([[float(subset.loc[subset.model == model, f"{field}_l2_percent"].iloc[0])
                            for field in FIELDS] for model in ORDER])
        image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=15 if group.startswith("original") else 7)
        ax.set_xticks(range(3), ["u", "w", "p"])
        ax.set_yticks(range(5), LABELS)
        ax.set_title(title)
        ax.set_xlabel("Field")
        for row in range(5):
            for col in range(3):
                ax.text(col, row, f"{values[row, col]:.2f}%", ha="center", va="center",
                        color="white" if values[row, col] > (7 if group.startswith("original") else 3.5)
                        else "#172029", fontsize=9)
        fig.colorbar(image, ax=ax, label="Relative L2 error (%)", shrink=0.85)
    fig.suptitle("NACA 0012 native-mesh POD surrogate accuracy", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
