import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


subjects = [
    "Biology",
    "Business",
    "Chemistry",
    "Computer Science",
    "Economics",
    "Engineering",
    "Health",
    "History",
    "Law",
    "Math",
    "Other",
    "Philosophy",
    "Physics",
    "Psychology",
]

languages = ["bn", "sw", "te", "ne", "hi"]

counts = np.array(
    [
        [340, 348, 342, 344, 344],
        [323, 327, 319, 321, 316],
        [351, 351, 353, 353, 353],
        [329, 326, 327, 331, 329],
        [341, 334, 342, 336, 342],
        [345, 348, 347, 340, 349],
        [328, 328, 324, 325, 326],
        [316, 313, 323, 319, 316],
        [335, 329, 336, 335, 333],
        [352, 358, 357, 352, 357],
        [323, 325, 321, 326, 324],
        [322, 317, 317, 320, 318],
        [347, 356, 346, 349, 346],
        [348, 340, 346, 349, 347],
    ]
)

totals = counts.sum(axis=1)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    }
)

fig = plt.figure(figsize=(12.6, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.5], wspace=0.24)

ax_bar = fig.add_subplot(gs[0, 0])
ax_heat = fig.add_subplot(gs[0, 1])

y = np.arange(len(subjects))
bar_colors = plt.cm.YlGnBu(np.linspace(0.35, 0.9, len(subjects)))
ax_bar.barh(y, totals, color=bar_colors, edgecolor="white", linewidth=0.7)
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(subjects)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Retained Source Questions")
ax_bar.set_title("Subject Totals")
ax_bar.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
ax_bar.set_axisbelow(True)
ax_bar.set_xlim(0, 2300)

for idx, value in enumerate(totals):
    ax_bar.text(value + 20, idx, f"{value:,}", va="center", ha="left", fontsize=8.3)

im = ax_heat.imshow(counts, cmap="YlOrRd", aspect="auto", vmin=310, vmax=360)
ax_heat.set_xticks(np.arange(len(languages)))
ax_heat.set_xticklabels([f"{lang}\n(4700)" for lang in languages])
ax_heat.set_yticks(np.arange(len(subjects)))
ax_heat.set_yticklabels(subjects)
ax_heat.set_title("Language-by-Subject Balance")

for row in range(counts.shape[0]):
    for col in range(counts.shape[1]):
        value = counts[row, col]
        text_color = "white" if value >= 360 else "#2b2b2b"
        ax_heat.text(col, row, str(value), ha="center", va="center", fontsize=8, color=text_color)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
cbar.set_label("Questions per Subject")

fig.suptitle(
    "Final SFT source pool after teacher-correct and judge-based filtering",
    fontsize=12,
    y=0.98,
)
fig.text(
    0.5,
    0.02,
    "23,500 retained source questions in total; each language contributes 4,700 questions, "
    "and each question is expanded into concept generation and concept-grounded answering tasks.",
    ha="center",
    fontsize=9,
)

output_dir = Path("latex/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "sft_subject_distribution.pdf", bbox_inches="tight")
fig.savefig(output_dir / "sft_subject_distribution.png", dpi=300, bbox_inches="tight")
