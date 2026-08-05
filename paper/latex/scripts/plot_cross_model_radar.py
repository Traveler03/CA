import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

labels = ["bn", "sw", "te", "ne", "hi"]
data = {
    "Llama": [3.08, 3.17, 3.65, 3.12, 2.86],
    "Ministral": [3.44, 3.79, 3.17, 3.34, 2.91],
    "Gemma": [2.35, 2.48, 2.15, 2.32, 2.08],
}

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8.5,
    }
)

colors = {
    "Llama": "#2a6f97",
    "Ministral": "#c75146",
    "Gemma": "#4c9f38",
}

fig, ax = plt.subplots(figsize=(5.4, 4.15), subplot_kw=dict(polar=True))

for name, values in data.items():
    values = values + values[:1]
    ax.plot(angles, values, linewidth=2.2, color=colors[name], label=name)
    ax.fill(angles, values, color=colors[name], alpha=0.08)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 4.5)
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["1", "2", "3", "4"], color="#666666")
ax.grid(color="#b0b0b0", alpha=0.45, linewidth=0.6)
ax.spines["polar"].set_linewidth(1.1)
ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.08), frameon=False, handlelength=2.6)

output_dir = Path(__file__).resolve().parents[1] / "figures"
output_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_dir / "cross_model_memory_gain_radar.pdf", bbox_inches="tight")
plt.savefig(output_dir / "cross_model_memory_gain_radar.png", dpi=300, bbox_inches="tight")
