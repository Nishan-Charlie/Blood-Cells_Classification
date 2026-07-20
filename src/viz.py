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
