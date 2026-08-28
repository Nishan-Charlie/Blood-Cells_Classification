"""Result figures for Chapter 4.

Forms chosen by the data's job:
  4.2  arm comparison        -> dot plot + error bars (small differences, need
                                position encoding, not truncated bars)
  4.3  per-class hier v flat -> dumbbell plot (paired values + the gap)
  4.4  stage distance        -> bar chart from zero (counts; the shape is the point)
  4.5  cross-lineage by seed -> paired slope plot (consistency is why p is small)

Palette: Okabe-Ito blue/amber, validated colourblind-safe
(protan dE 29.2, deutan/tritan >= 30). Every mark is direct-labelled, so
identity never rests on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts"

BLUE, AMBER = "#0072B2", "#E69F00"
INK, INK2, MUTED = "#22262b", "#555b62", "#8b9198"
GRID = "#e3e6e9"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def tidy(ax, *, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if xgrid:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)


d = pd.read_csv(REPO / "results" / "summary.csv")
d = d[~d.arm.str.startswith(("smoke_", "screen_"))]

LABEL = {"hierarchical": "Hierarchical\n(proposed)", "flat_baseline": "Flat baseline",
         "no_imbalance": "No imbalance term", "stain_norm": "Stain-normalised"}
ORDER = ["hierarchical", "flat_baseline", "stain_norm", "no_imbalance"]

# ---------------------------------------------------------------- Figure 4.2
g = d.groupby("arm")
fig, ax = plt.subplots(figsize=(7.0, 3.4))
y = np.arange(len(ORDER))[::-1]

for off, (col, colour, name) in enumerate([
        ("test_macro_f1", BLUE, "Macro F1 (all 18 classes)"),
        ("test_minority_macro_f1", AMBER, "Minority macro F1 (8 rarest)")]):
    m = np.array([g.get_group(a)[col].mean() for a in ORDER])
    s = np.array([g.get_group(a)[col].std(ddof=1) for a in ORDER])
    yy = y + (0.16 if off == 0 else -0.16)
    ax.errorbar(m, yy, xerr=s, fmt="o", ms=7, color=colour, ecolor=colour,
                elinewidth=1.4, capsize=3, label=name, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    for xi, yi in zip(m, yy):
        ax.text(xi, yi + 0.135, f"{xi:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK2)

ax.set_yticks(y, [LABEL[a] for a in ORDER], fontsize=9, color=INK)
ax.set_xlabel("F1 score  (dot = mean of 3 seeds, bars = ±1 SD)")
ax.set_xlim(0.74, 0.925)
ax.set_ylim(-0.55, len(ORDER) - 0.30)
ax.set_title("Overall and minority-class performance by experimental arm")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2,
          frameon=False, fontsize=8.5, labelcolor=INK2)
tidy(ax)
fig.savefig(OUT / "fig_arm_comparison.png")
plt.close(fig)
print("wrote fig_arm_comparison.png")

# ---------------------------------------------------------------- Figure 4.3
pc = pd.read_csv(REPO / "results" / "per_class_by_arm.csv", index_col=0)
names = list(pc.index)
h, f = pc["hierarchical"].values, pc["flat_baseline"].values
yy = np.arange(len(names))[::-1]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
for i, (hi, fi) in enumerate(zip(h, f)):
    ax.plot([fi, hi], [yy[i], yy[i]], color=GRID, linewidth=2.2, zorder=1,
            solid_capstyle="round")
ax.scatter(f, yy, s=42, color=AMBER, zorder=3, label="Flat baseline",
           edgecolor=SURFACE, linewidth=1.1)
ax.scatter(h, yy, s=42, color=BLUE, zorder=3, label="Hierarchical (proposed)",
           edgecolor=SURFACE, linewidth=1.1)

# Direct-label only the two classes that carry the aggregate difference.
for i, n in enumerate(names):
    delta = h[i] - f[i]
    if abs(delta) >= 0.04:
        ax.annotate(f"+{delta:.3f}", (h[i], yy[i]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=BLUE, fontweight="bold")

ax.set_yticks(yy, names, fontsize=8.5, color=INK)
ax.set_xlabel("Test F1 (mean of 3 seeds)")
ax.set_xlim(0.30, 1.06)
ax.set_title("Per-class F1: the hierarchical gain is confined to two rare classes")
ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
tidy(ax)
fig.savefig(OUT / "fig_per_class_dumbbell.png")
plt.close(fig)
print("wrote fig_per_class_dumbbell.png")

# ---------------------------------------------------------------- Figure 4.4
# Measured in results/maturation_adjacency.txt (3 seeds pooled, 305 in-chain errors)
dist = {"-2": 7, "-1": 120, "+1": 174, "+2": 4}
tot = sum(dist.values())
fig, ax = plt.subplots(figsize=(6.4, 3.1))
xs = np.arange(len(dist))
vals = [dist[k] / tot for k in dist]
bars = ax.bar(xs, vals, width=0.62, color=[MUTED, BLUE, BLUE, MUTED], zorder=3)
for x, v, k in zip(xs, vals, dist):
    ax.text(x, v + 0.012, f"{v:.1%}\n({dist[k]})", ha="center", va="bottom",
            fontsize=8.5, color=INK2, linespacing=1.35)
ax.set_xticks(xs, ["−2 stages", "−1 stage", "+1 stage", "+2 stages"],
              fontsize=9.5, color=INK)
ax.set_ylabel("Share of maturation-chain errors")
ax.set_ylim(0, 0.70)
ax.set_yticks(np.arange(0, 0.71, 0.2))
ax.set_title("96% of maturation errors land on an immediately adjacent stage")
ax.set_axisbelow(True)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.grid(axis="x", visible=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
fig.savefig(OUT / "fig_stage_distance.png")
plt.close(fig)
print("wrote fig_stage_distance.png")

# ---------------------------------------------------------------- Figure 4.5
piv = d.pivot_table(index="seed", columns="arm", values="test_cross_lineage_error")
fig, ax = plt.subplots(figsize=(4.9, 3.3))
for seed in piv.index:
    ax.plot([0, 1], [piv.loc[seed, "flat_baseline"], piv.loc[seed, "hierarchical"]],
            marker="o", ms=7, linewidth=1.8, color=BLUE, alpha=0.85, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.1)
    ax.annotate(f"seed {seed}", (1, piv.loc[seed, "hierarchical"]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK2)
ax.set_xticks([0, 1], ["Flat baseline", "Hierarchical"], fontsize=9.5, color=INK)
ax.set_xlim(-0.28, 1.42)
ax.set_ylabel("Cross-lineage error rate")
ax.set_title("Cross-lineage error falls in every seed", fontsize=10.5, pad=26)
ax.text(0.5, 1.035, "narrow axis: the mean drop is 0.0004, about 3 images in 6,244",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
        color=MUTED, style="italic")
ax.set_axisbelow(True)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.grid(axis="x", visible=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
fig.savefig(OUT / "fig_cross_lineage_paired.png")
plt.close(fig)
print("wrote fig_cross_lineage_paired.png")
