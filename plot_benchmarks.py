import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("benchmark.csv")

COLORS = {
    "linear":                      "#e74c3c",
    "naive_iterative":             "#e67e22",
    "separable_iterative":         "#3498db",
    "separable_recursive_deriche": "#2ecc71",
    "separable_recursive_van_vliet": "#9b59b6",
}
LABELS = {
    "linear":                        "Linear (CPU)",
    "naive_iterative":               "Naive Iterative (GPU)",
    "separable_iterative":           "Separable Iterative (GPU)",
    "separable_recursive_deriche":   "Separable Recursive Deriche (GPU)",
    "separable_recursive_van_vliet": "Separable Recursive Van Vliet (GPU)",
}
SIZES = [500, 1000, 2000, 3000]
PROGRAMS = list(COLORS.keys())


# ── Graph 1: Time vs Sigma, one subplot per image size ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for ax, size in zip(axes, SIZES):
    sub = df[df["image_size"] == size]
    for prog in PROGRAMS:
        d = sub[sub["program"] == prog].sort_values("sigma")
        ax.plot(d["sigma"], d["time_ms"], label=LABELS[prog],
                color=COLORS[prog], linewidth=2, marker="o", markersize=3)
    ax.set_title(f"Image size: {size}×{size}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True)
fig.suptitle("Gaussian Blur: Execution Time vs Sigma\n(by image size)", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("img/time_vs_sigma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved img/time_vs_sigma.png")


# ── Graph 2: Speedup over linear baseline, one subplot per image size ─────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for ax, size in zip(axes, SIZES):
    sub = df[df["image_size"] == size]
    linear_times = sub[sub["program"] == "linear"].set_index("sigma")["time_ms"]
    for prog in [p for p in PROGRAMS if p != "linear"]:
        d = sub[sub["program"] == prog].sort_values("sigma").copy()
        d["speedup"] = d["sigma"].map(linear_times) / d["time_ms"]
        ax.plot(d["sigma"], d["speedup"], label=LABELS[prog],
                color=COLORS[prog], linewidth=2, marker="o", markersize=3)
    ax.axhline(1, color=COLORS["linear"], linestyle="--", linewidth=1.5,
               label="Linear baseline (1×)")
    ax.set_title(f"Image size: {size}×{size}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Speedup (×)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True)
fig.suptitle("Gaussian Blur: Speedup vs Sigma (relative to Linear CPU)\n(by image size)",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("img/speedup_vs_sigma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved img/speedup_vs_sigma.png")


# ── Graph 3: Time vs Image Size at sigma=15 ────────────────────────────────────
FIXED_SIGMA = 15
sub = df[df["sigma"] == FIXED_SIGMA]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for prog in PROGRAMS:
    d = sub[sub["program"] == prog].sort_values("image_size")
    ax1.plot(d["image_size"], d["time_ms"], label=LABELS[prog],
             color=COLORS[prog], linewidth=2, marker="s", markersize=6)
ax1.set_title(f"All algorithms (σ = {FIXED_SIGMA})", fontsize=12, fontweight="bold")
ax1.set_xlabel("Image size (px)")
ax1.set_ylabel("Time (ms)")
ax1.set_xticks(SIZES)
ax1.grid(True, alpha=0.3)

for prog in [p for p in PROGRAMS if p != "linear"]:
    d = sub[sub["program"] == prog].sort_values("image_size")
    ax2.plot(d["image_size"], d["time_ms"], label=LABELS[prog],
             color=COLORS[prog], linewidth=2, marker="s", markersize=6)
ax2.set_title(f"GPU algorithms only — zoomed (σ = {FIXED_SIGMA})", fontsize=12, fontweight="bold")
ax2.set_xlabel("Image size (px)")
ax2.set_ylabel("Time (ms)")
ax2.set_xticks(SIZES)
ax2.grid(True, alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.08), frameon=True)
fig.suptitle(f"Execution Time vs Image Size (σ = {FIXED_SIGMA})", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig("img/time_vs_imagesize.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved img/time_vs_imagesize.png")


print("\nDone — 3 graphs saved to img/")
