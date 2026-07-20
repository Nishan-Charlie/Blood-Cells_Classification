"""Embedding extraction from a frozen pretrained backbone.

The point is diagnostic, not predictive: project every image into the feature
space of an ImageNet-pretrained backbone *before any fine-tuning*, so the
embedding reflects generic visual structure rather than a model already told
what the hierarchy is. If the three lineages separate here, the lineage-aware
premise has support that does not depend on the model we are about to train.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import MLL23Dataset


@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    *,
    backbone: str = "resnet50",
    batch_size: int = 128,
    device: str | None = None,
    max_per_class: int | None = None,
    seed: int = config.SPLIT_SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return pooled features and the aligned metadata rows.

    Args:
        df: Manifest rows to embed.
        backbone: Any ``timm`` model name; loaded pretrained with the classifier
            head removed so the output is the pooled feature vector.
        max_per_class: If set, subsample this many images per fine class. UMAP on
            all 41k images is slow and visually dense; a few hundred per class
            shows the structure. Sampling is stratified and seeded.

    Returns:
        ``(features, meta)`` where ``features`` is ``(N, D)`` and ``meta`` is the
        corresponding manifest rows, index-aligned to the feature rows.
    """
    import timm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if max_per_class is not None:
        df = (
            df.groupby("y2", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
            .reset_index(drop=True)
        )

    model = timm.create_model(backbone, pretrained=True, num_classes=0).eval().to(device)

    # Use the eval (deterministic) transform: no augmentation should perturb an
    # embedding meant to characterise the data.
    ds = MLL23Dataset(df, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats = []
    for x, _, _ in loader:
        feats.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0), df.reset_index(drop=True)


def project(features: np.ndarray, *, method: str = "umap", seed: int = config.SPLIT_SEED) -> np.ndarray:
    """Reduce features to 2-D for plotting.

    Standardises first, since both UMAP and t-SNE are distance-based and raw
    backbone features are not scale-comparable across dimensions.
    """
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(features)

    if method == "umap":
        import umap

        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(x)
    raise ValueError(f"unknown method {method!r}")


def save_embeddings(features: np.ndarray, meta: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (config.INTERIM_DIR / "embeddings.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, y1=meta["y1"].to_numpy(), y2=meta["y2"].to_numpy())
    return path
