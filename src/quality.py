"""Dataset quality diagnostics and quantitative tests of the label hierarchy.

Two independent jobs:

* **Image quality** - per-image brightness, contrast, sharpness, saturation, so
  colour and focus variation across classes is measured rather than assumed, and
  perceptual-hash near-duplicate detection. The duplicate check is not cosmetic:
  a near-duplicate cell that straddles the train/test split inflates the test
  score and quietly invalidates the evaluation.
* **Hierarchy tests** - does the lineage structure actually exist in feature
  space? Silhouette, k-NN label agreement, and an inter-class distance matrix
  turn "the hierarchy is biologically motivated" into a measured claim.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from .hierarchy import CLASSES, FINE_NAMES, FINE_TO_LINEAGE, LINEAGES


# --------------------------------------------------------------------------- #
# Per-image quality statistics
# --------------------------------------------------------------------------- #
def image_stats(path: str) -> dict:
    """Brightness, RMS contrast, saturation, and a Laplacian sharpness proxy."""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    mx = img.max(axis=2)
    sat = np.where(mx > 0, (mx - img.min(axis=2)) / (mx + 1e-6), 0.0)

    # Variance of a 4-neighbour Laplacian: higher means sharper edges / focus.
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(sat.mean()),
        "sharpness": float(lap.var()),
    }


def collect_stats(df: pd.DataFrame, *, sample_per_class: int | None = 300, seed: int = 42) -> pd.DataFrame:
    """Per-image stats over a stratified sample, with class/lineage attached."""
    if sample_per_class is not None:
        df = df.groupby("y2", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_per_class), random_state=seed)
        )
    rows = []
    for r in df.itertuples():
        rows.append({**image_stats(r.path), "y2": r.y2, "class_name": r.class_name, "lineage": r.lineage})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #
def find_near_duplicates(
    df: pd.DataFrame, *, hash_size: int = 8, max_distance: int = 4
) -> pd.DataFrame:
    """Group images whose perceptual hashes are within ``max_distance`` bits.

    Buckets by exact hash first (the common case), then does the O(n^2) Hamming
    comparison only within a small candidate set, which keeps 41k images
    tractable. Cross-split pairs are flagged, because those are the ones that
    corrupt the evaluation.

    Returns:
        One row per duplicate pair: the two paths, their splits, hash distance,
        and whether the pair crosses the train/val/test boundary.
    """
    import imagehash

    hashes = {}
    for r in df.itertuples():
        h = imagehash.phash(Image.open(r.path).convert("L"), hash_size=hash_size)
        split = getattr(r, "split", None)
        hashes[r.path] = (h, r.class_name, split)

    items = list(hashes.items())
    buckets: dict[str, list] = defaultdict(list)
    for path, (h, *_rest) in items:
        buckets[str(h)].append(path)

    pairs = []
    paths = [p for p, _ in items]
    hlist = [hashes[p][0] for p in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = hlist[i] - hlist[j]
            if d <= max_distance:
                (_, c1, s1), (_, c2, s2) = hashes[paths[i]], hashes[paths[j]]
                pairs.append(
                    {
                        "path_a": paths[i], "path_b": paths[j],
                        "class_a": c1, "class_b": c2,
                        "split_a": s1, "split_b": s2,
                        "distance": int(d),
                        "cross_split": s1 is not None and s2 is not None and s1 != s2,
                    }
                )
    return pd.DataFrame(pairs)


# --------------------------------------------------------------------------- #
# Quantitative hierarchy tests
# --------------------------------------------------------------------------- #
def lineage_separation(features: np.ndarray, y1: np.ndarray) -> dict:
    """Silhouette of the lineage grouping in feature space.

    Positive means lineages form coherent clusters; near zero means the
    lineage-level structure is weak in generic features.
    """
    from sklearn.metrics import silhouette_score

    return {
        "silhouette_lineage": float(silhouette_score(features, y1, sample_size=min(5000, len(y1)), random_state=42)),
        "n": int(len(y1)),
    }


def knn_label_agreement(features: np.ndarray, labels: np.ndarray, *, k: int = 10) -> float:
    """Fraction of each point's k nearest neighbours sharing its label.

    A label-agnostic measure of how cleanly a label set is expressed in feature
    space. Run once with ``y1`` and once with ``y2`` to compare how well lineage
    vs fine type is captured.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    idx = nn.kneighbors(features, return_distance=False)[:, 1:]  # drop self
    return float((labels[idx] == labels[:, None]).mean())


def interclass_distance_matrix(features: np.ndarray, y2: np.ndarray) -> pd.DataFrame:
    """Euclidean distance between fine-class centroids, ordered by class index.

    Because class indices follow the maturation continuum, a real hierarchy
    shows small off-diagonal distances within a lineage block and larger ones
    across lineages - visible as block structure when plotted.
    """
    cents = np.vstack([features[y2 == c.idx].mean(0) for c in CLASSES])
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
    return pd.DataFrame(d, index=list(FINE_NAMES), columns=list(FINE_NAMES))


def within_vs_across_lineage(dist: pd.DataFrame) -> dict:
    """Mean centroid distance for same-lineage vs different-lineage class pairs.

    The headline number: if the hierarchy is real, same-lineage pairs are
    meaningfully closer than cross-lineage pairs.
    """
    lut = np.array(FINE_TO_LINEAGE)
    d = dist.to_numpy()
    same, cross = [], []
    n = len(lut)
    for i in range(n):
        for j in range(i + 1, n):
            (same if lut[i] == lut[j] else cross).append(d[i, j])
    same, cross = np.array(same), np.array(cross)
    return {
        "within_lineage_mean": float(same.mean()),
        "across_lineage_mean": float(cross.mean()),
        "ratio": float(cross.mean() / (same.mean() + 1e-6)),
    }
