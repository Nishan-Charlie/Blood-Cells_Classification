"""Build and verify the image manifest.

The manifest is the single source of truth for what is on disk: one row per
image, carrying both hierarchy levels. Everything downstream (splits, datasets,
class weights) reads it rather than re-walking the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .hierarchy import CLASSES, LINEAGES, TOTAL_EXPECTED_IMAGES

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_manifest(raw_dir: Path | None = None) -> pd.DataFrame:
    """Walk the extracted class folders and assemble the manifest.

    Returns:
        DataFrame with columns ``path, filename, class_key, class_name, y2,
        lineage, y1``. Rows are sorted by ``(y2, filename)`` so the ordering is
        stable across machines regardless of directory iteration order.
    """
    raw_dir = raw_dir or config.RAW_DIR
    rows = []
    for cls in CLASSES:
        folder = raw_dir / cls.key
        if not folder.is_dir():
            raise FileNotFoundError(f"missing class folder: {folder}")
        for p in folder.iterdir():
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "class_key": cls.key,
                    "class_name": cls.name,
                    "y2": cls.idx,
                    "lineage": cls.lineage,
                    "y1": cls.lineage_idx,
                }
            )

    df = pd.DataFrame(rows).sort_values(["y2", "filename"]).reset_index(drop=True)
    return df


def verify_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Compare observed per-class counts against the published MLL23 counts.

    Returns:
        A report DataFrame with an ``ok`` column. The caller decides whether a
        discrepancy is fatal; we surface it rather than raising, because a
        partial download is a normal intermediate state.
    """
    observed = df["y2"].value_counts().to_dict()
    report = pd.DataFrame(
        [
            {
                "class_name": c.name,
                "lineage": c.lineage,
                "expected": c.expected_count,
                "observed": observed.get(c.idx, 0),
                "delta": observed.get(c.idx, 0) - c.expected_count,
            }
            for c in CLASSES
        ]
    )
    report["ok"] = report["delta"] == 0
    return report


def imbalance_summary(df: pd.DataFrame) -> dict:
    """Headline imbalance statistics quoted in the dissertation."""
    counts = df["y2"].value_counts()
    largest, smallest = counts.max(), counts.min()
    return {
        "n_images": len(df),
        "n_classes": df["y2"].nunique(),
        "largest_class": df.loc[df.y2 == counts.idxmax(), "class_name"].iloc[0],
        "largest_count": int(largest),
        "smallest_class": df.loc[df.y2 == counts.idxmin(), "class_name"].iloc[0],
        "smallest_count": int(smallest),
        "imbalance_ratio": round(largest / smallest, 1),
        "lineage_counts": df["lineage"].value_counts().reindex(LINEAGES).to_dict(),
    }


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.MANIFEST_PATH)


def is_complete(df: pd.DataFrame) -> bool:
    return len(df) == TOTAL_EXPECTED_IMAGES and bool(verify_manifest(df)["ok"].all())
