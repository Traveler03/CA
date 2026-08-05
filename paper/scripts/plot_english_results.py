import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


methods = ["Zero", "CoT", "X-CoT", "Self-Tr.", "SoT", "CA-only", "CA-Mem"]
models = ["Base", "Direct-SFT", "CA-SFT"]

global_mmlu = {
    "Base": [78.15, 78.32, 78.88, 77.11, 79.32, 78.69, 79.16],
    "Direct-SFT": [78.02, 78.16, 78.55, 77.85, 79.02, 78.31, 78.79],
    "CA-SFT": [78.81, 79.02, 80.18, 79.80, 80.76, 80.02, 80.58],
}

mgsm = {
    "Base": [27.2, 27.6, 28.4, 27.2, 28.8, 27.6, 28.4],
    "Direct-SFT": [24.8, 25.2, 32.4, 31.6, 33.2, 31.2, 32.8],
    "CA-SFT": [31.2, 31.6, 37.2, 35.6, 38.0, 36.4, 37.6],
}

gains = {
    "Global-MMLU": {"Base": 0.47, "Direct-SFT": 0.48, "CA-SFT": 0.56},
    "MGSM": {"Base": 0.80, "Direct-SFT": 1.60, "CA-SFT": 1.20},
}


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
    }
)

colors = {
    "Base": "#4c78a8",
    "Direct-SFT": "#f58518",
    "CA-SFT": "#54a24b",
}

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
width = 0.22
x = np.arange(len(methods))


def plot_panel(ax, data, title, ylim):
    for idx, model in enumerate(models):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            data[model],
            width=width,
            color=colors[model],
            label=model,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    gain_lines = [
        (f"Base: +{gains[title]['Base']:.2f}", colors["Base"]),
        (f"Direct-SFT: +{gains[title]['Direct-SFT']:.2f}", colors["Direct-SFT"]),
        (f"CA-SFT: +{gains[title]['CA-SFT']:.2f}", colors["CA-SFT"]),
    ]
    ax.text(
        0.02,
        0.98,
        "CA-Mem gain over CA-only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "linewidth": 0.6,
            "alpha": 0.95,
        },
        zorder=5,
    )
    for idx, (label, color) in enumerate(gain_lines):
        ax.text(
            0.04,
            0.92 - idx * 0.06,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            color=color,
            zorder=5,
        )


plot_panel(axes[0], global_mmlu, "Global-MMLU", (76.8, 81.35))
plot_panel(axes[1], mgsm, "MGSM", (24.0, 40.2))

axes[0].set_ylabel("Accuracy")
axes[1].set_ylabel("Accuracy")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(rect=(0, 0, 1, 0.98))

output_dir = Path("latex/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "english_results_bars.pdf", bbox_inches="tight")
fig.savefig(output_dir / "english_results_bars.png", dpi=300, bbox_inches="tight")
