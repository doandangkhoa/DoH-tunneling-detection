"""Publication figures for the DoH evaluation reports."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

COLORS = {"xgboost": "#5144d5", "isolation_forest": "#087f70", "cnn1d": "#bd4d32"}
NAMES = {"xgboost": "XGBoost", "isolation_forest": "Isolation Forest", "cnn1d": "1D-CNN"}


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_figures(metrics, sources, curves, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for model, curve in curves.groupby("model", sort=False):
        ap = metrics.loc[model, "ap"]
        ax.step(curve.recall, curve.precision, where="pre", color=COLORS[model],
                linewidth=2, label=f"{NAMES[model]} (AP {ap:.2%})")
    prevalence = metrics.iloc[0].tunnel / metrics.iloc[0].n
    ax.axhline(prevalence, color="#65718a", linestyle="--", label=f"Tunnel prevalence ({prevalence:.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall on the shared test set",
           xlim=(0, 1.02), ylim=(0, 1.04))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, directory / "pr_comparison.png")

    for model, row in metrics.iterrows():
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, f"{count:,}", ha="center", va="center", fontsize=16,
                    color="white" if count > matrix.max() * .55 else "#172447")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Tunnel"],
               yticklabels=["Benign", "Tunnel"], xlabel="Predicted label", ylabel="Actual label",
               title=f"{NAMES[model]} · Confusion matrix")
        save(fig, directory / f"confusion_matrix_{model}.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, label, title in zip(axes, (0, 1), ("False positives by browser", "Tunnel recall by tool")):
        rows = sources[sources.label.eq(label)]
        names = rows.source.unique()
        positions = np.arange(len(names))
        for offset, model in enumerate(metrics.index):
            values = rows[rows.model.eq(model)].set_index("source").loc[names, "rate"]
            bars = ax.bar(positions + (offset - 1) * .25, values, width=.23, color=COLORS[model], label=NAMES[model])
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=8, padding=3)
        ax.set(xticks=positions, xticklabels=names, title=title, ylim=(0, 1.19))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper center", fontsize=9)
    save(fig, directory / "source_comparison.png")
