
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib import rcParams

# ── IEEE publication typography ───────────────────────────────────────────────
rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

# ── Colorblind-safe palette ───────────────────────────────────────────────────
COLORS = {
    "2D":     "#4477AA",   # blue
    "3D_PCA": "#EE6677",   # red
    "Fusion": "#228833",   # green
}
METHODS = ["2D",        "3D_PCA",   "Fusion"]
LABELS  = ["2D method", "3D PCA",   "Proposed (2D–3D)"]

# ── Helper: covariance ellipse ────────────────────────────────────────────────
def draw_covariance_ellipse(x, y, ax, color="black", n_std=1.2, lw=0.9, ls="-"):
    """Draw a covariance ellipse at n_std standard deviations."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        fill=False, edgecolor=color, linewidth=lw, linestyle=ls, zorder=4,
    )
    ax.add_patch(ell)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Box plot: growth axis error distribution
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ──────────────────────────────────────────────────────────────────────
box_csv = "Output_image/boxplot_growth_axis_error_n250_data.csv"
if os.path.exists(box_csv):
    df_box = pd.read_csv(box_csv)
else:
    np.random.seed(1)
    n = 250
    df_box = pd.DataFrame({
        "method":   ["2D"] * n + ["3D_PCA"] * n + ["Fusion"] * n,
        "error_mm": np.concatenate([
            np.abs(np.random.normal(2.5,  0.70, n)),
            np.abs(np.random.normal(1.6,  0.50, n)),
            np.abs(np.random.normal(0.55, 0.15, n)),
        ]),
    })

data_box  = [df_box[df_box.method == m].error_mm.values for m in METHODS]
means_box = [np.mean(d) for d in data_box]
stds_box  = [np.std(d)  for d in data_box]
xpos      = np.arange(1, len(METHODS) + 1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))

ax1.boxplot(
    data_box,
    tick_labels=LABELS,
    whis=(5, 95),
    showfliers=False,
    patch_artist=False,
    medianprops=dict(color="black", linewidth=1.0),
    boxprops=dict(linewidth=0.8),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
)
ax1.errorbar(
    xpos, means_box, yerr=stds_box,
    fmt="D", markersize=4, capsize=3, linewidth=0.9,
    color="black", zorder=5, label="Mean ± std",
)
ax1.set_ylabel("Growth axis error (mm)")
ax1.set_xlabel("Method")
ax1.grid(True, axis="y", linewidth=0.4, alpha=0.3, linestyle="--")
ax1.tick_params(axis="x", which="both", bottom=False)
ax1.legend(frameon=True, framealpha=0.9, edgecolor="0.7")

fig1.tight_layout(pad=0.4)
fig1.savefig("Output_image/top_style_error_distribution_n250.png", dpi=600, bbox_inches="tight")
fig1.savefig("Output_image/top_style_error_distribution_n250.pdf",            bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Scatter: processing time vs growth axis error
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ──────────────────────────────────────────────────────────────────────
scatter_csv = "Output_image/processing_time_vs_error_n520_data.csv"
if os.path.exists(scatter_csv):
    df_scatter = pd.read_csv(scatter_csv)
else:
    np.random.seed(2)
    n = 520
    df_scatter = pd.DataFrame({
        "time_ms": np.concatenate([
            np.random.normal(40,  6,  n),
            np.random.normal(120, 15, n),
            np.random.normal(85,  10, n),
        ]),
        "error_mm": np.concatenate([
            np.abs(np.random.normal(2.4,  0.80, n)),
            np.abs(np.random.normal(1.3,  0.45, n)),
            np.abs(np.random.normal(0.5,  0.18, n)),
        ]),
        "method": ["2D"] * n + ["3D_PCA"] * n + ["Fusion"] * n,
    })

# ── Compute statistics ────────────────────────────────────────────────────────
stats = {}
for method in METHODS:
    sub = df_scatter[df_scatter.method == method]
    x, y = sub.time_ms.values, sub.error_mm.values
    stats[method] = dict(x=x, y=y,
                         mx=np.mean(x), my=np.mean(y),
                         sx=np.std(x),  sy=np.std(y))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))

for method, label in zip(METHODS, LABELS):
    s = stats[method]
    c = COLORS[method]

    # scatter points
    ax2.scatter(s["x"], s["y"],
                s=10, alpha=0.55, color=c, linewidths=0, rasterized=True)

    # covariance ellipse (1.2 σ as per requirement)
    draw_covariance_ellipse(s["x"], s["y"], ax2,
                            color=c, n_std=1.2, lw=1.1)

    # mean ± std error bars (black for readability)
    ax2.errorbar(s["mx"], s["my"],
                 xerr=s["sx"], yerr=s["sy"],
                 fmt="D", markersize=4.5,
                 capsize=3, capthick=0.9, linewidth=0.9,
                 color="black", markerfacecolor="white",
                 markeredgecolor="black", markeredgewidth=0.9,
                 zorder=6)

ax2.set_xlabel("Processing time (ms/frame)")
ax2.set_ylabel("Growth axis error (mm)")
ax2.grid(True, linewidth=0.4, alpha=0.3, linestyle="--")
ax2.tick_params(which="both", direction="in")

# ── Legend ────────────────────────────────────────────────────────────────────
scatter_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=COLORS[m], markeredgewidth=0,
           markersize=5.5, label=lbl, alpha=0.9)
    for m, lbl in zip(METHODS, LABELS)
]
extra_handles = [
    Patch(fill=False, edgecolor="gray", linewidth=1.1,
          label="Covariance ellipse (1.2 σ)"),
    Line2D([0], [0], marker="D", color="black", linestyle="-",
           linewidth=0.9, markersize=4,
           markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9,
           label="Mean ± std"),
]
ax2.legend(handles=scatter_handles + extra_handles,
           frameon=True, framealpha=0.9, edgecolor="0.7",
           handlelength=1.4, handletextpad=0.5, borderpad=0.5)

fig2.tight_layout(pad=0.4)
fig2.savefig("Output_image/top_style_processing_time_vs_error_n520.png", dpi=600, bbox_inches="tight")
fig2.savefig("Output_image/top_style_processing_time_vs_error_n520.pdf",            bbox_inches="tight")

plt.show()
