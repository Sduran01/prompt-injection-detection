
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results")
OUT.mkdir(exist_ok=True)

datasets = ["Safe Guard", "Deepset", "Gandalf"]

# Values from check_final_checkpoint.py
recall = [0.9968, 0.3837, 0.9349]
lower = [0.9919, 0.3212, 0.9199]
upper = [1.0000, 0.4436, 0.9489]

errors = [
    [r - lo for r, lo in zip(recall, lower)],
    [hi - r for r, hi in zip(recall, upper)],
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    datasets,
    recall,
    yerr=errors,
    capsize=6,
    edgecolor="black",
    alpha=0.8
)

ax.set_ylabel("Recall")
ax.set_xlabel("Evaluation dataset")
ax.set_title(
    "Cross-Dataset Prompt-Injection Detection\n"
    "TF-IDF + Logistic Regression"
)
ax.set_ylim(0, 1.1)

for bar, value in zip(bars, recall):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.07,
        f"{value:.2%}",
        ha="center"
    )

fig.tight_layout()
fig.savefig(OUT / "final_comparison.png", dpi=200)
plt.close(fig)

print("Saved results/final_comparison.png")
