"""Optional plots of the measured IQD correlations, without model dependencies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import PREDICTORS


def plot_correlations(summary: dict, path: str | Path) -> None:
    dims, bits = summary["dims"], summary["bits"]
    cells = {(cell["dim"], cell["bits"]): cell for cell in summary["per_setting"]}
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")
    for name, ax in zip(PREDICTORS, axes.flat):
        values = np.full((len(dims), len(bits)), np.nan)
        for i, dim in enumerate(dims):
            for j, bit in enumerate(bits):
                rho = cells[(dim, bit)]["measures"][name]["spearman_rho"]
                if rho is not None:
                    values[i, j] = rho
        shown = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_title(name)
        ax.set_xticks(range(len(bits)), bits)
        ax.set_yticks(range(len(dims)), dims)
        ax.set_xlabel("Quantization precision (bits / dimension)")
        ax.set_ylabel("Embedding dimensions")
        for i, j in np.argwhere(np.isfinite(values)):
            rho = values[i, j]
            ax.text(j, i, f"{rho:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(rho) > 0.65 else "black")
    figure.colorbar(shown, ax=axes.ravel().tolist(), shrink=0.7, label="Spearman ρ with rank degradation")
    dataset = summary.get("manifest", {}).get("dataset", "explicit-GT embedding pool")
    figure.suptitle(f"IQD predictor correlations · {dataset}\n"
                   f"{summary['n_queries']} queries · fixed symmetric grid · gray = undefined correlation")
    figure.savefig(path, dpi=160)
    plt.close(figure)
