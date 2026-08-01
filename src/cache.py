"""Decoded-image memmap cache.

Why this exists
---------------
Measured dataloader throughput on the target machine (RTX 4070 Laptop) is
**178 img/s at num_workers=4**, and *lower* at num_workers=8 (119 img/s) from
worker contention. A full ResNet-50 training step runs at 82 img/s with a peak
of only 3.3 GB of the 8.6 GB available. The pipeline is **data-bound**: the GPU
idles while workers decode TIFFs, so neither a larger batch nor a smaller
backbone buys anything.

Decoding a 288x288 TIFF costs roughly 4 ms. Reading the same pixels from a
uint8 memmap is a ~250 KB memcpy. Removing the decode is the single change that
moves the bottleneck onto the GPU where it belongs.

What is cached
--------------
**Decoded pixels at native 288x288 resolution** - not preprocessed tensors, and
not 224x224.

That distinction is load-bearing. ``transforms.py`` rotates at the native 288
before downsampling to 224 so the image is resampled once rather than twice.
Caching at 224 would silently break that property and change the augmentation
semantics of every experiment. Caching the decoded 288 px image removes only
the decode cost and leaves the rest of the pipeline bit-for-bit identical.

Staleness
---------
Each cache carries a JSON sidecar recording the row count and a hash of the
manifest's ``path`` column. :func:`load_cache` verifies the sidecar against the
manifest it is handed and returns ``None`` on mismatch, so a stale or partial
cache degrades to correct-but-slow direct TIFF reads rather than silently
serving the wrong image for a label.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

#: Cached images keep the corpus's native resolution. See the module docstring
#: for why this is 288 and not config.IMAGE_SIZE (224).
CACHE_SIZE = config.SOURCE_IMAGE_SIZE

RAW_CACHE_PATH = config.INTERIM_DIR / "images_288.npy"
REINHARD_CACHE_PATH = config.INTERIM_DIR / "images_288_reinhard.npy"

#: Rows per task handed to a worker process. Large enough that per-task overhead
#: is negligible, small enough that progress is visible and a failure loses little.
_CHUNK = 256


def _sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def manifest_fingerprint(df: pd.DataFrame) -> str:
    """Hash the manifest's path column, in order.

    Identifies *which images, in which row order* a cache was built from. Row
    order matters because the cache is indexed positionally: if the manifest were
    re-sorted, row i would no longer be the image the caller expects.
    """
    joined = "\n".join(df["path"].astype(str))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _decode(path: str) -> np.ndarray:
    """Decode one image to a (288, 288, 3) uint8 array.

    MLL23 is uniformly 288x288, but a defensive resize keeps one odd file from
    aborting a 40-minute build. Anything already the right size skips the resize
    entirely, so the common path costs nothing.
    """
    with Image.open(path) as im:
        img = im.convert("RGB")
        if img.size != (CACHE_SIZE, CACHE_SIZE):
            img = img.resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _decode_chunk(args) -> tuple[int, np.ndarray]:
    """Decode a contiguous block of rows. Runs in a worker process.

    Returns ``(start_index, stacked_array)`` so the parent can write the block
    into the memmap at the right offset without the workers sharing the file.
    """
    start, paths, reinhard_ref = args
    out = np.empty((len(paths), CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        rgb = _decode(p)
        if reinhard_ref is not None:
            # Imported lazily: the stain module pulls in scikit-image, which is
            # slow to import and pointless for the raw cache.
            from .stain import ReinhardReference, normalise_reinhard

            rgb = normalise_reinhard(rgb, ReinhardReference(**reinhard_ref))
        out[i] = rgb
    return start, out


def build_cache(
    df: pd.DataFrame,
    cache_path: Path,
    *,
    reinhard_ref: dict | None = None,
    workers: int = 6,
    overwrite: bool = False,
) -> Path:
    """Decode every manifest row into a uint8 memmap on disk.

    Args:
        df: Manifest rows, in the order the cache will be indexed by.
        cache_path: Destination ``.npy``. A ``.json`` sidecar is written beside it.
        reinhard_ref: If given, apply Reinhard stain normalisation while decoding
            and store the normalised pixels. This is how the stain-normalised arm
            gets its own cache; normalising on the fly would cost ~6-10 ms/image
            and push that arm back to data-bound, making it slower than the other
            arms for reasons unrelated to the science.
        workers: Decode processes. Six leaves headroom on an 8-core machine.
        overwrite: Rebuild even if a valid cache is already present.

    Returns:
        The cache path.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = manifest_fingerprint(df)

    if not overwrite and validate_cache(cache_path, df) is not None:
        print(f"cache already valid: {cache_path}")
        return cache_path

    n = len(df)
    print(f"building {cache_path.name}: {n:,} images, "
          f"{n * CACHE_SIZE * CACHE_SIZE * 3 / 1e9:.1f} GB")

    # Open in w+ mode so the file is allocated up front. Writing through a memmap
    # keeps peak RSS at one chunk rather than the whole 10 GB array.
    mm = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint8, shape=(n, CACHE_SIZE, CACHE_SIZE, 3)
    )

    paths = df["path"].astype(str).tolist()
    tasks = [
        (s, paths[s: s + _CHUNK], reinhard_ref)
        for s in range(0, n, _CHUNK)
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start, block in pool.map(_decode_chunk, tasks):
            mm[start: start + len(block)] = block
            done += len(block)
            if done % (_CHUNK * 20) == 0 or done == n:
                print(f"  {done:,}/{n:,}  ({100 * done / n:.0f}%)", flush=True)

    mm.flush()
    del mm  # release the mapping before the sidecar advertises the cache as ready

    _sidecar_path(cache_path).write_text(
        json.dumps({
            "n": n,
            "size": CACHE_SIZE,
            "fingerprint": fingerprint,
            "reinhard": reinhard_ref is not None,
        })
    )
    print(f"done: {cache_path}")
    return cache_path


def validate_cache(cache_path: Path, df: pd.DataFrame) -> Path | None:
    """Return the cache path if it can be trusted for this manifest, else None.

    Returns None -- rather than raising -- when the cache is absent, incomplete,
    or built from a different manifest. The caller falls back to direct TIFF
    reads, so a stale cache costs speed, never correctness.

    This returns a *path*, not an open memmap, because the path is what gets
    handed to ``MLL23Dataset``: an open memmap cannot cross a Windows process
    boundary without being serialised in full. See the note in ``dataset.py``.
    """
    sidecar = _sidecar_path(cache_path)
    if not cache_path.exists() or not sidecar.exists():
        return None

    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return None

    if meta.get("n") != len(df) or meta.get("size") != CACHE_SIZE:
        return None
    if meta.get("fingerprint") != manifest_fingerprint(df):
        return None

    return cache_path


def load_cache(cache_path: Path, df: pd.DataFrame) -> np.ndarray | None:
    """Memory-map a validated cache for single-process use (notebooks, figures).

    Do not hand the result to a DataLoader with ``num_workers > 0``; use
    :func:`validate_cache` and pass the path instead.
    """
    path = validate_cache(cache_path, df)
    return None if path is None else np.load(path, mmap_mode="r")


def fit_reinhard_reference(train_df: pd.DataFrame, *, n_sample: int = 400, seed: int = 42) -> dict:
    """Fit the Reinhard reference on a training-split sample.

    Fitting on train only is not a detail: a reference fitted over all splits
    would leak test-set colour statistics into training. Returns a plain dict so
    it can be pickled to decode workers.
    """
    from .stain import fit_reinhard

    sample = train_df.sample(min(n_sample, len(train_df)), random_state=seed)
    ref = fit_reinhard([_decode(p) for p in sample["path"].astype(str)])
    return {"mean": ref.mean, "std": ref.std}
