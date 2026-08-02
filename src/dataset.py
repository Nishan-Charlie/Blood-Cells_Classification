"""PyTorch dataset over the MLL23 manifest.

Yields ``(image, y1, y2)`` where ``y1`` is the lineage and ``y2`` the fine cell
type, so the same dataset feeds the hierarchical model, the flat baseline
(which ignores ``y1``), and both ablations. Keeping one dataset class is what
stops the four experiment configurations from drifting apart.

Images come either from a decoded-pixel memmap cache (see ``cache.py``) or, when
no cache is supplied or the cache does not match the manifest, from the original
TIFFs. Both paths yield identical pixels; the cache only removes decode cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config, transforms as T
from .hierarchy import FINE_TO_LINEAGE


class MLL23Dataset(Dataset):
    """Single-cell peripheral blood images with two-level labels.

    Args:
        df: Manifest rows for one split; needs ``path``, ``y1``, ``y2``. When a
            cache is supplied, the DataFrame's *index* must still be the row
            position in the full manifest the cache was built from - which is
            what ``splits.csv`` naturally carries, since ``make_splits`` only
            appends a column and never reorders.
        train: Enable augmentation. When True the augmentation strength is
            chosen per sample from its class, so minority classes receive the
            intensified regime.
        transform: Override the built-in transform selection entirely.
        cache_path: Path to a decoded-image cache built by ``cache.build_cache``,
            indexed by manifest row. ``None`` reads the original TIFFs instead.

    Note:
        The cache is stored as a **path**, not as an open memmap, and is opened
        lazily per worker. This is not incidental: Windows DataLoader workers are
        spawned rather than forked, so the Dataset is pickled into each one - and
        pickling a ``np.memmap`` serialises all 10 GB of it per worker. Holding
        the path and mapping it on first use inside the worker keeps each worker's
        copy to a file handle.
    """

    def __init__(self, df: pd.DataFrame, *, train: bool = False, transform=None,
                 cache_path: Path | None = None, aug_policy: str = "basic") -> None:
        # Capture the manifest row positions *before* reset_index discards them;
        # these are what index into the cache.
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache_idx = df.index.to_numpy() if cache_path is not None else None
        self._mm: np.ndarray | None = None  # opened lazily, per process

        self.df = df.reset_index(drop=True)
        self.train = train

        if transform is not None:
            self._fixed = transform
            self._majority = self._minority = None
        elif train:
            self._fixed = None
            # Built once, not per __getitem__, which would rebuild the compose
            # object 41k times per epoch.
            self._majority = T.train_transform(minority=False, policy=aug_policy)
            self._minority = T.train_transform(minority=True, policy=aug_policy)
        else:
            self._fixed = T.eval_transform()
            self._majority = self._minority = None

        self._paths = self.df["path"].to_numpy()
        self._y1 = torch.as_tensor(self.df["y1"].to_numpy(), dtype=torch.long)
        self._y2 = torch.as_tensor(self.df["y2"].to_numpy(), dtype=torch.long)
        self._use_minority = [T.is_minority(int(v)) for v in self.df["y2"]]

    def __len__(self) -> int:
        return len(self.df)

    def _memmap(self) -> np.ndarray:
        """Open the cache in this process, once. See the class note on pickling."""
        if self._mm is None:
            self._mm = np.load(self._cache_path, mmap_mode="r")
        return self._mm

    def _read(self, i: int):
        """Fetch image i as something the v2 transforms accept.

        Cache path returns a uint8 CHW tensor; the ``np.array`` copy is required
        because a memmap slice is read-only and torch would otherwise hand the
        augmentation pipeline a non-writable buffer. Fallback path returns a PIL
        image. The transforms begin with ``v2.ToImage()``, which normalises
        either form, so the two paths are interchangeable and produce identical
        pixels.
        """
        if self._cache_path is not None:
            arr = np.array(self._memmap()[self._cache_idx[i]])  # (H, W, C) uint8 copy
            return torch.from_numpy(arr).permute(2, 0, 1)
        with Image.open(self._paths[i]) as im:
            return im.convert("RGB")

    def __getstate__(self):
        """Drop the open memmap before pickling to a worker process."""
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self._read(i)
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
    """Build a dataset for one named split, enabling augmentation only on train.

    ``sub`` keeps ``split_df``'s index, so cache row positions survive the
    selection. Do not insert a ``reset_index`` here.
    """
    sub = split_df[split_df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split {split!r}")
    return MLL23Dataset(sub, train=(split == "train"), **kwargs)


def load_split_file(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.SPLIT_PATH)
