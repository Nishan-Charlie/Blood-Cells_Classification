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


# --------------------------------------------------------------------------- #
# Modelling figures (notebooks 03-05)
# --------------------------------------------------------------------------- #
def training_curves(history: pd.DataFrame, *, title: str | None = None):
    """Loss components and validation metrics over epochs.

    Three panels. The left one separates the loss into its fine, lineage, and
    consistency terms: if the hierarchy is doing anything, the lineage and
    consistency curves must actually move, and a flat consistency curve is
    evidence the term is inert rather than helping.

    The right panel plots validation macro F1 and accuracy together. The gap
    between them *is* the imbalance story - accuracy running far ahead of macro
    F1 means the model is riding the head classes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ep = history["epoch"]

    # --- loss components ---
    parts = [("train_fine", "fine", "#2a78d6"),
             ("train_lineage", "lineage", "#008300"),
             ("train_consistency", "consistency", "#e87ba4")]
    for col, label, colour in parts:
        if col in history.columns:
            axes[0].plot(ep, history[col], label=label, color=colour, lw=2)
    axes[0].set_title("Training loss components", pad=10)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8, labelcolor=INK_SECONDARY)

    # --- selection metric vs accuracy ---
    axes[1].plot(ep, history["val_macro_f1"], label="macro F1", color="#2a78d6", lw=2)
    axes[1].plot(ep, history["val_accuracy"], label="accuracy", color=INK_MUTED, lw=1.6, ls="--")
    if "val_balanced_accuracy" in history.columns:
        axes[1].plot(ep, history["val_balanced_accuracy"], label="balanced acc",
                     color="#008300", lw=1.6)
    best = int(history["val_macro_f1"].idxmax())
    axes[1].axvline(history.loc[best, "epoch"], color=BASELINE, lw=1, zorder=0)
    axes[1].annotate(f"selected\nep {int(history.loc[best, 'epoch'])}",
                     (history.loc[best, "epoch"], history["val_macro_f1"].max()),
                     textcoords="offset points", xytext=(6, -18),
                     fontsize=8, color=INK_SECONDARY)
    axes[1].set_title("Validation metrics (selection on macro F1)", pad=10)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8, labelcolor=INK_SECONDARY)

    # --- hierarchical error split ---
    if "val_cross_lineage_error" in history.columns:
        axes[2].plot(ep, history["val_within_lineage_error"], label="within-lineage",
                     color="#008300", lw=2)
        axes[2].plot(ep, history["val_cross_lineage_error"], label="cross-lineage",
                     color="#e87ba4", lw=2)
        axes[2].set_title("Error type over training", pad=10)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Share of all samples")
        axes[2].legend(fontsize=8, labelcolor=INK_SECONDARY)
    else:
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


def confusion_heatmap(cm: pd.DataFrame, *, title: str | None = None, annotate: bool = False):
    """Row-normalised confusion matrix with lineage blocks marked.

    Class order is the maturation continuum, never alphabetical, so adjacent
    maturation stages sit adjacent to the diagonal and the clinically expected
    confusions appear as near-diagonal mass rather than scattered noise. White
    rules mark the lineage boundaries: anything outside those blocks is a severe
    cross-lineage error.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ("#ffffff",) + SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="equal")

    names = list(cm.index)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(names)), names, fontsize=7.5, color=INK_SECONDARY)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.grid(False)

    lut = [c.lineage for c in CLASSES]
    for i in range(len(lut) - 1):
        if lut[i] != lut[i + 1]:
            ax.axhline(i + 0.5, color=SURFACE, lw=2.5)
            ax.axvline(i + 0.5, color=SURFACE, lw=2.5)

    if annotate:
        vals = cm.to_numpy()
        for i in range(len(names)):
            for j in range(len(names)):
                if vals[i, j] >= 0.01:
                    ax.text(j, i, f"{vals[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                            fontsize=6, color=SURFACE if vals[i, j] > 0.55 else INK_SECONDARY)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of true class (recall)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title or "Confusion matrix, row-normalised", pad=12)
    fig.tight_layout()
    return fig


def arm_comparison(summary: pd.DataFrame, *, metric: str = "test_macro_f1",
                   title: str | None = None):
    """Per-arm mean +/- std of one metric, as a labelled bar chart.

    Error bars are the standard deviation across seeds. Direct value labels are
    always drawn, so magnitude never depends on reading a bar against an axis.
    """
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    df = summary.sort_values(mean_col)

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(df) + 2.2))
    ypos = np.arange(len(df))
    ax.barh(ypos, df[mean_col], height=_BAR_HEIGHT, color="#2a78d6", zorder=3,
            xerr=df[std_col] if std_col in df.columns else None,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3})

    ax.set_yticks(ypos, df["arm"], fontsize=9)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_xlim(0, min(1.0, float(df[mean_col].max()) * 1.28))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for y, (m, s) in enumerate(zip(df[mean_col], df.get(std_col, [0] * len(df)))):
        ax.text(m + float(df[mean_col].max()) * 0.02 + (s or 0), y,
                f"{m:.3f} ± {s:.3f}" if s else f"{m:.3f}",
                va="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_title(title or f"{metric.replace('_', ' ')} by experiment arm", pad=12)
    fig.tight_layout()
    return fig


def error_composition(summary: pd.DataFrame, *, title: str | None = None):
    """Stacked correct / within-lineage / cross-lineage shares per arm.

    The figure that carries the hierarchical claim. A lineage-aware model should
    shrink the cross-lineage segment specifically - shifting error toward the
    clinically milder within-lineage kind - and that is visible here in a way no
    scalar metric shows.
    """
    cols = [("test_correct_mean", "correct", "#2a78d6"),
            ("test_within_lineage_error_mean", "within-lineage error", "#008300"),
            ("test_cross_lineage_error_mean", "cross-lineage error", "#e87ba4")]
    present = [(c, lab, col) for c, lab, col in cols if c in summary.columns]

    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(summary) + 2.2))
    ypos = np.arange(len(summary))
    left = np.zeros(len(summary))

    for c, label, colour in present:
        vals = summary[c].to_numpy(dtype=float)
        ax.barh(ypos, vals, left=left, height=_BAR_HEIGHT, color=colour, label=label,
                zorder=3, edgecolor=SURFACE, linewidth=1.0)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v > 0.035:  # below this the label would not fit inside the segment
                ax.text(l + v / 2, y, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color=SURFACE)
        left += vals

    ax.set_yticks(ypos, summary["arm"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of test samples")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.set_title(title or "Prediction composition by arm", pad=12)
    fig.tight_layout()
    return fig


def per_class_f1_comparison(tables: dict[str, pd.DataFrame], *, title: str | None = None):
    """Per-class F1 for several arms, ordered by class index.

    Minority classes are marked, because the whole argument is about them. Where
    arms differ most on the rare classes is the dissertation's central result.
    """
    fig, ax = plt.subplots(figsize=(11, 6.4))
    names = list(next(iter(tables.values()))["class_name"])
    ypos = np.arange(len(names))
    palette = ["#2a78d6", "#008300", "#e87ba4", "#7b4fa8", "#c07a1e"]

    n = len(tables)
    height = _BAR_HEIGHT / n
    for k, (arm, tbl) in enumerate(tables.items()):
        offset = (k - (n - 1) / 2) * height
        ax.barh(ypos + offset, tbl["f1"], height=height, zorder=3,
                color=palette[k % len(palette)], label=arm)

    minority = next(iter(tables.values()))["is_minority"].to_numpy()
    labels = [f"{n}  *" if m else n for n, m in zip(names, minority)]
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower right")
    ax.set_title(title or "Per-class F1 by arm  (*  = minority class)", pad=12)
    fig.tight_layout()
    return fig


def gradcam_grid(panels, *, ncols: int = 4, title: str | None = None):
    """Grid of Grad-CAM overlays.

    Args:
        panels: list of ``(overlay_rgb, caption)``.
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.75))
    for ax, (img, caption) in zip(np.ravel(axes), panels):
        ax.imshow(img)
        ax.set_title(caption, fontsize=7.5, color=INK_SECONDARY, pad=4)
    for ax in np.ravel(axes):
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
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
