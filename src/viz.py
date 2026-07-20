"""Figures for the dissertation.

Palette note: the three lineage colours are categorical slots 1-3 of the
reference palette, validated for colour-vision deficiency under the strict
all-pairs rule (worst pair dE 13.0 protan, 27.5 normal vision). Magenta sits
below 3:1 against the light surface, so every chart that uses it carries direct
value labels - identity and magnitude are never colour-alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .hierarchy import CLASSES, LINEAGES

# Categorical slots 1-3, in fixed order. Never cycled, never reassigned by rank.
LINEAGE_COLORS: dict[str, str] = {
    "Lymphoid": "#2a78d6",
    "Myeloid": "#008300",
    "Erythroid": "#e87ba4",
}

# Sequential blue ramp (reference palette, light->dark) for magnitude encodings
# such as the inter-class distance heatmap. One hue, never a rainbow.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

SPLIT_COLORS: dict[str, str] = {
    "train": "#2a78d6",
    "val": "#008300",
    "test": "#e87ba4",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

_BAR_HEIGHT = 0.72  # leaves a visible surface gap between adjacent bars


def apply_style() -> None:
    """Recessive chrome: hairline grid, muted axes, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK_PRIMARY,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def _thousands(x, _pos) -> str:
    return f"{int(x):,}"


def class_distribution(counts: pd.Series, *, title: str | None = None):
    """Per-class image counts, ordered by fine-class index.

    Two panels: a linear scale that makes the 260:1 imbalance visceral, and a
    log scale that makes the rare classes legible. Both carry direct value
    labels, so the rare classes remain readable where their bars vanish.

    Args:
        counts: Image count indexed by ``y2``.
        title: Optional suptitle.
    """
    names = [c.name for c in CLASSES]
    lineages = [c.lineage for c in CLASSES]
    vals = np.array([counts.get(c.idx, 0) for c in CLASSES], dtype=float)
    colors = [LINEAGE_COLORS[l] for l in lineages]
    ypos = np.arange(len(CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)

    for ax, logscale in zip(axes, (False, True)):
        ax.barh(ypos, vals, height=_BAR_HEIGHT, color=colors, zorder=3)
        if logscale:
            ax.set_xscale("log")
            ax.set_xlim(20, vals.max() * 2.4)
            ax.set_xlabel("Images (log scale)")
        else:
            ax.set_xlim(0, vals.max() * 1.18)
            ax.set_xlabel("Images (linear scale)")
            ax.xaxis.set_major_formatter(FuncFormatter(_thousands))

        for y, v in zip(ypos, vals):
            ax.text(
                v * (1.12 if logscale else 1) + (0 if logscale else vals.max() * 0.012),
                y,
                f"{int(v):,}",
                va="center",
                ha="left",
                fontsize=8,
                color=INK_SECONDARY,
            )
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)

    # Invert once, not per-axis: the axes share a y-axis, so inverting both
    # cancels out and silently reverses the class ordering.
    axes[0].invert_yaxis()
    axes[0].set_yticks(ypos, names, fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=LINEAGE_COLORS[l]) for l in LINEAGES]
    fig.legend(
        handles, list(LINEAGES), loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.035), fontsize=9, labelcolor=INK_SECONDARY,
    )
    if title:
        fig.suptitle(title, y=0.98, fontsize=12, ha="center")
    fig.tight_layout()
    return fig


def split_composition(summary: pd.DataFrame):
    """Stacked per-class train/val/test counts, to show no split starves a class."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ypos = np.arange(len(summary))
    left = np.zeros(len(summary))

    for split in ("train", "val", "test"):
        vals = summary[split].to_numpy(dtype=float)
        ax.barh(
            ypos, vals, left=left, height=_BAR_HEIGHT,
            color=SPLIT_COLORS[split], label=split, zorder=3,
            edgecolor=SURFACE, linewidth=1.0,  # 2px surface gap between segments
        )
        left += vals

    ax.set_yticks(ypos, summary.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_xscale("log")
    ax.set_xlim(20, left.max() * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for y, total in zip(ypos, left):
        ax.text(total * 1.08, y, f"{int(total):,}", va="center", fontsize=8, color=INK_SECONDARY)

    ax.legend(loc="lower right", fontsize=9, labelcolor=INK_SECONDARY)
    ax.set_title("Stratified split composition per class (log scale)", pad=12)
    fig.tight_layout()
    return fig


def sample_grid(images, labels, *, ncols: int = 6, title: str | None = None):
    """Grid of denormalised sample images with their class names."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.85, nrows * 2.05))
    for ax, img, lab in zip(np.ravel(axes), images, labels):
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(lab, fontsize=7.5, color=INK_SECONDARY, pad=4)
    for ax in np.ravel(axes):
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Dataset-analysis figures (notebook 02)
# --------------------------------------------------------------------------- #
def _lineage_cmap():
    """Discrete colormap over the three lineages, in fixed index order."""
    from matplotlib.colors import ListedColormap

    return ListedColormap([LINEAGE_COLORS[l] for l in LINEAGES])


def embedding_scatter(coords, meta, *, colour_by: str = "lineage", method: str = "UMAP",
                      title: str | None = None):
    """2-D embedding coloured by lineage or fine class.

    Lineage colouring uses the three fixed categorical hues and a legend, so
    identity is never colour-alone. Fine-class colouring (18 classes) exceeds the
    categorical cap, so it uses a lightness ramp *within* each lineage's hue and
    is presented as an exploratory view, with the lineage panel as the one that
    carries the argument.
    """
    fig, ax = plt.subplots(figsize=(8.4, 7.4))

    if colour_by == "lineage":
        y = meta["y1"].to_numpy()
        for i, lin in enumerate(LINEAGES):
            m = y == i
            ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.55,
                       color=LINEAGE_COLORS[lin], label=lin, linewidths=0)
        ax.legend(loc="best", fontsize=10, labelcolor=INK_SECONDARY, markerscale=2)
    else:
        # Per-fine-class: hue by lineage, lightness by within-lineage rank.
        from matplotlib.colors import to_rgb
        for lin in LINEAGES:
            members = [c for c in CLASSES if c.lineage == lin]
            base = np.array(to_rgb(LINEAGE_COLORS[lin]))
            for j, c in enumerate(members):
                m = meta["y2"].to_numpy() == c.idx
                shade = 0.35 + 0.6 * (j / max(len(members) - 1, 1))
                ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5,
                           color=tuple(base * shade + (1 - shade) * 0.55), linewidths=0)

    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title or f"{method} of frozen-backbone features, coloured by {colour_by}", pad=10)
    fig.tight_layout()
    return fig


def distance_heatmap(dist: pd.DataFrame, *, title: str | None = None):
    """Inter-class centroid-distance matrix as a sequential-blue heatmap.

    Class order follows the maturation continuum, so a real hierarchy appears as
    darker (nearer) blocks along the lineage diagonal.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE[::-1])
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(dist.to_numpy(), cmap=cmap, aspect="equal")

    names = list(dist.index)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(names)), names, fontsize=7.5, color=INK_SECONDARY)

    # Lineage boundaries as white gridlines: they demarcate the expected blocks.
    lut = [c.lineage for c in CLASSES]
    bounds = [i + 0.5 for i in range(len(lut) - 1) if lut[i] != lut[i + 1]]
    for b in bounds:
        ax.axhline(b, color=SURFACE, lw=2); ax.axvline(b, color=SURFACE, lw=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance between class centroids", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title or "Inter-class distance in feature space", pad=10)
    fig.tight_layout()
    return fig


def quality_distributions(stats: pd.DataFrame, metrics=("brightness", "contrast", "saturation", "sharpness")):
    """Per-lineage distributions of image-quality metrics as overlaid densities."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2))
    for ax, metric in zip(np.ravel(axes), metrics):
        for lin in LINEAGES:
            vals = stats.loc[stats.lineage == lin, metric].to_numpy()
            if len(vals) < 2:
                continue
            lo, hi = np.percentile(vals, [1, 99])
            grid = np.linspace(lo, hi, 200)
            from scipy.stats import gaussian_kde
            ax.plot(grid, gaussian_kde(vals)(grid), color=LINEAGE_COLORS[lin], lw=2, label=lin)
        ax.set_title(metric, fontsize=10, color=INK_PRIMARY)
        ax.set_yticks([]); ax.grid(axis="x", visible=True)
    axes[0].legend(fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle("Image-quality metric distributions by lineage", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def stain_comparison(triples, *, title: str | None = None):
    """Rows of (original, Macenko, Reinhard) for a handful of images.

    Args:
        triples: list of (original, macenko, reinhard) uint8 RGB arrays.
    """
    n = len(triples)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 2.1, n * 2.1))
    axes = np.atleast_2d(axes)
    heads = ["Original", "Macenko", "Reinhard"]
    for r, (orig, mac, rein) in enumerate(triples):
        for c, img in enumerate((orig, mac, rein)):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(heads[c], fontsize=10, color=INK_SECONDARY)
    fig.suptitle(title or "Stain normalisation", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig
