"""Deterministic stratified train/validation/test partitioning.

Stratification is on the *fine* label (``y2``), which automatically stratifies
the lineage too, since every fine class belongs to exactly one lineage.

The rarest class holds 33 images, so a 70/15/15 split leaves roughly 23/5/5.
That is thin but non-empty in every split, which is the guarantee the proposal
makes. :func:`check_split` asserts it rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .hierarchy import CLASSES

SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    df: pd.DataFrame,
    *,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    seed: int = config.SPLIT_SEED,
) -> pd.DataFrame:
    """Assign each manifest row to train/val/test.

    Uses two successive stratified splits: first carve off the training set,
    then divide the remainder into validation and test in proportion.

    Returns:
        The manifest with an added ``split`` column.
    """
    df = df.reset_index(drop=True)
    idx = df.index.to_numpy()
    y = df["y2"].to_numpy()

    train_idx, holdout_idx = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=seed, shuffle=True
    )

    # Within the held-out 30%, val and test should end up equal for a 70/15/15.
    holdout_val_frac = val_frac / (1.0 - train_frac)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=holdout_val_frac,
        stratify=y[holdout_idx],
        random_state=seed,
        shuffle=True,
    )

    out = df.copy()
    out["split"] = pd.NA
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"

    if out["split"].isna().any():
        raise RuntimeError("some rows were not assigned a split")
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts in each split, ordered by fine-class index."""
    table = (
        df.pivot_table(index=["y2", "class_name"], columns="split", values="path", aggfunc="count")
        .reindex(columns=list(SPLIT_NAMES))
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    return table.reset_index().sort_values("y2").set_index("class_name").drop(columns="y2")


def check_split(df: pd.DataFrame) -> list[str]:
    """Validate the split. Returns a list of human-readable problems (empty = good)."""
    problems: list[str] = []

    for cls in CLASSES:
        sub = df[df["y2"] == cls.idx]
        for name in SPLIT_NAMES:
            n = int((sub["split"] == name).sum())
            if n == 0:
                problems.append(f"{cls.name}: {name} split is empty")

    # No image may appear in more than one split.
    dupes = df[df.duplicated("path", keep=False)]
    if len(dupes):
        problems.append(f"{len(dupes)} duplicate image paths across splits")

    # Realised proportions should track the requested ones closely.
    fracs = df["split"].value_counts(normalize=True)
    for name, want in zip(SPLIT_NAMES, (config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC)):
        got = float(fracs.get(name, 0.0))
        if abs(got - want) > 0.01:
            problems.append(f"{name} fraction {got:.3f} deviates from {want:.2f}")

    return problems


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.SPLIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.SPLIT_PATH)
