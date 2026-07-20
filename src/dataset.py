"""PyTorch dataset over the MLL23 manifest.

Yields ``(image, y1, y2)`` where ``y1`` is the lineage and ``y2`` the fine cell
type, so the same dataset feeds the hierarchical model, the flat baseline
(which ignores ``y1``), and both ablations. Keeping one dataset class is what
stops the four experiment configurations from drifting apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config, transforms as T
from .hierarchy import FINE_TO_LINEAGE


class MLL23Dataset(Dataset):
    """Single-cell peripheral blood images with two-level labels.

    Args:
        df: Manifest rows for one split; needs ``path``, ``y1``, ``y2``.
        train: Enable augmentation. When True the augmentation strength is
            chosen per sample from its class, so minority classes receive the
            intensified regime.
        transform: Override the built-in transform selection entirely.
    """

    def __init__(self, df: pd.DataFrame, *, train: bool = False, transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.train = train

        if transform is not None:
            self._fixed = transform
            self._majority = self._minority = None
        elif train:
            self._fixed = None
            # Built once, not per __getitem__, which would rebuild the compose
            # object 41k times per epoch.
            self._majority = T.train_transform(minority=False)
            self._minority = T.train_transform(minority=True)
        else:
            self._fixed = T.eval_transform()
            self._majority = self._minority = None

        self._paths = self.df["path"].to_numpy()
        self._y1 = torch.as_tensor(self.df["y1"].to_numpy(), dtype=torch.long)
        self._y2 = torch.as_tensor(self.df["y2"].to_numpy(), dtype=torch.long)
        self._use_minority = [T.is_minority(int(v)) for v in self.df["y2"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with Image.open(self._paths[i]) as im:
            img = im.convert("RGB")

        tf = self._fixed or (self._minority if self._use_minority[i] else self._majority)
        return tf(img), self._y1[i], self._y2[i]

    @property
    def class_counts(self) -> torch.Tensor:
        """Per-fine-class sample counts, indexed 0..17. Feeds the class-balanced loss."""
        counts = self.df["y2"].value_counts().reindex(range(len(FINE_TO_LINEAGE)), fill_value=0)
        return torch.as_tensor(counts.to_numpy(), dtype=torch.float32)


def from_splits(
    split_df: pd.DataFrame, split: str, **kwargs
) -> MLL23Dataset:
    """Build a dataset for one named split, enabling augmentation only on train."""
    sub = split_df[split_df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split {split!r}")
    return MLL23Dataset(sub, train=(split == "train"), **kwargs)


def load_split_file(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.SPLIT_PATH)
