import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd

PROGRAMS = [
    ("linear",                        "Linear\n(CPU)",                          "#e74c3c"),
    ("naive_iterative",               "Naive Iterative\n(GPU)",                 "#e67e22"),
    ("separable_iterative",           "Separable Iterative\n(GPU)",             "#3498db"),
    ("separable_recursive_deriche",   "Separable Recursive\nDeriche (GPU)",     "#2ecc71"),
    ("separable_recursive_van_vliet", "Separable Recursive\nVan Vliet (GPU)",   "#9b59b6"),
]

# Real sigma outputs produced by the actual programs on 3000x3000
SIGMA_IMAGES = [
    (1,  "img/sigma/output_3000_s1.jpg"),
    (3,  "img/sigma/output_3000_s3.jpg"),
    (5,  "img/sigma/output_3000_s5.jpg"),
    (10, "img/sigma/output_3000_s10.jpg"),
    (20, "img/sigma/output_3000_s20.jpg"),
    (30, "img/sigma/output_3000_s30.jpg"),
]

# Real per-algorithm outputs at sigma=10, 3000x3000
CORRECTNESS_IMAGES = {
    "linear":                        "img/correctness/output_linear_3000_s10.jpg",
    "naive_iterative":               "img/correctness/output_naive_iterative_3000_s10.jpg",
    "separable_iterative":           "img/correctness/output_separable_iterative_3000_s10.jpg",
    "separable_recursive_deriche":   "img/correctness/output_separable_recursive_deriche_3000_s10.jpg",
    "separable_recursive_van_vliet": "img/correctness/output_separable_recursive_van_vliet_3000_s10.jpg",
}

def load(path):
    return np.array(Image.open(path))


# ── Graph 4: Original vs real sigma progression (3000×3000) ───────────────────
original = load("img/input_3000.jpg")

n = len(SIGMA_IMAGES) + 1
fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.6))

axes[0].imshow(original)
axes[0].set_title("Original", fontsize=11, fontweight="bold", pad=6)
axes[0].axis("off")

for ax, (sigma, path) in zip(axes[1:], SIGMA_IMAGES):
    ax.imshow(load(path))
    ax.set_title(f"σ = {sigma}", fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")

fig.suptitle("Effect of Gaussian Blur at Different Sigma Values (3000×3000, real program output)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("img/sigma_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved img/sigma_progression.png")


# ── Graph 5: All programs at sigma=10 — real outputs, different speed ─────────
DEMO_SIGMA = 10
df = pd.read_csv("benchmark.csv")
times = df[(df["sigma"] == DEMO_SIGMA) & (df["image_size"] == 3000)].set_index("program")["time_ms"]

n_prog = len(PROGRAMS)
fig = plt.figure(figsize=(3.2 * (n_prog + 1), 4.4))
gs = gridspec.GridSpec(1, n_prog + 1, figure=fig, wspace=0.05)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(original)
ax0.set_title("Original", fontsize=10, fontweight="bold", pad=6)
ax0.axis("off")
rect = mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax0.transAxes,
                                boxstyle="round,pad=0.03", linewidth=2,
                                edgecolor="#555", facecolor="none", clip_on=False)
ax0.add_patch(rect)

for i, (prog, label, color) in enumerate(PROGRAMS):
    ax = fig.add_subplot(gs[i + 1])
    ax.imshow(load(CORRECTNESS_IMAGES[prog]))
    t = times.get(prog, float("nan"))
    ax.set_title(f"{label}\n{t:.1f} ms", fontsize=9, fontweight="bold",
                 pad=6, color=color)
    ax.axis("off")
    rect = mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                    boxstyle="round,pad=0.03", linewidth=2.5,
                                    edgecolor=color, facecolor="none", clip_on=False)
    ax.add_patch(rect)

fig.suptitle(f"All Algorithms at σ = {DEMO_SIGMA}, 3000×3000 — Identical Output, Vastly Different Speed",
             fontsize=12, fontweight="bold", y=1.02)
plt.savefig("img/programs_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved img/programs_comparison.png")


print("\nDone — 2 image demo slides saved to img/")
