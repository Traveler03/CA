import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


labels = ["CA-only", "0.5", "0.6", "0.7", "0.8", "0.9"]
gamma_positions = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
coverage = [76.91, 65.38, 57.24, 40.83, 21.68]
macro_avg = [61.94, 64.67, 65.17, 65.55, 66.24, 65.38]


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

coverage_color = "#2a6f97"
macro_color = "#c75146"
best_color = "#9e2f25"
x = np.arange(len(labels))
coverage_x = x[1:]

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(6.7, 3.55),
    sharex=False,
    gridspec_kw={"width_ratios": [1.0, 1.05]},
)

ax1.plot(
    coverage_x,
    coverage,
    marker="o",
    linewidth=2.3,
    markersize=6.5,
    color=coverage_color,
    linestyle="-",
)
ax1.fill_between(coverage_x, coverage, 0, color=coverage_color, alpha=0.08)
ax1.set_title("Coverage")
ax1.set_xlabel(r"Threshold $\gamma$")
ax1.set_ylabel("Coverage (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlim(-0.2, 5.2)
ax1.set_ylim(0, 80)
ax1.set_yticks([0, 20, 40, 60, 80])
ax1.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.32)
ax1.set_axisbelow(True)

ax2.plot(
    x,
    macro_avg,
    marker="s",
    linewidth=2.4,
    markersize=7,
    color=macro_color,
)
ax2.fill_between(x, macro_avg, 61.7, color=macro_color, alpha=0.08)
ax2.scatter([4], [66.24], color=best_color, s=65, zorder=5)
ax2.set_title("Macro Avg.")
ax2.set_xlabel(r"Threshold $\gamma$")
ax2.set_ylabel("Macro Avg.")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlim(-0.2, 5.2)
ax2.set_ylim(61.8, 66.5)
ax2.set_yticks([62.0, 63.0, 64.0, 65.0, 66.0])
ax2.axvline(4, color=best_color, linestyle=":", linewidth=0.9, alpha=0.55)
ax2.axhline(66.24, color=best_color, linestyle=":", linewidth=0.9, alpha=0.45)
ax2.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.32)
ax2.set_axisbelow(True)

ax2.annotate(
    "best",
    xy=(4, 66.24),
    xytext=(2.75, 66.30),
    textcoords="data",
    fontsize=8.2,
    color=best_color,
    arrowprops={"arrowstyle": "->", "color": best_color, "lw": 0.8},
    bbox={
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.9,
    },
)

fig.tight_layout()

output_dir = Path("latex/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "threshold_sensitivity_curve.pdf", bbox_inches="tight")
fig.savefig(output_dir / "threshold_sensitivity_curve.png", dpi=300, bbox_inches="tight")
