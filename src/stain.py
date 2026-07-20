"""Stain normalisation for Pappenheim-stained smears.

Two classical methods, both operating on the idea that a stained image is a
product of a stain-colour basis and per-pixel stain concentrations:

* **Macenko** - estimates the stain-colour basis from the image itself via the
  optical-density plane's principal directions, then re-projects onto a fixed
  reference basis. Pappenheim is a Romanowsky stain, so the two-stain optical
  density model that Macenko designed for H&E transfers directly.
* **Reinhard** - matches the mean and standard deviation of each channel in Lab
  colour space to a reference. Cruder but ~20x cheaper, so it is the on-the-fly
  fallback when Macenko would bottleneck the loader.

Both are *fit on the training split only*. Fitting the reference on all data
would leak test-set colour statistics into training. The fitted reference is a
small array that is saved and reloaded, so a run never re-fits.

References:
    Macenko et al. (2009), "A method for normalizing histology slides for
    quantitative analysis." Reinhard et al. (2001), "Color transfer between
    images."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_EPS = 1e-6

# ImageNet-independent; these are the Macenko reference concentrations most
# implementations use, kept so results are comparable with the literature.
_DEFAULT_MAX_C = np.array([1.9705, 1.0308], dtype=np.float64)


def _to_od(rgb: np.ndarray, io: float = 255.0) -> np.ndarray:
    """Convert an 8-bit RGB image to optical density. Shape (..., 3)."""
    rgb = rgb.astype(np.float64)
    return -np.log((rgb + 1.0) / io)


def _from_od(od: np.ndarray, io: float = 255.0) -> np.ndarray:
    rgb = io * np.exp(-od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


@dataclass
class MacenkoReference:
    """Fitted Macenko reference: a stain-colour basis and concentration scale."""

    stain_matrix: list  # 3x2
    max_concentration: list  # 2

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "MacenkoReference":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def he(self) -> np.ndarray:
        return np.asarray(self.stain_matrix, dtype=np.float64)

    @property
    def maxc(self) -> np.ndarray:
        return np.asarray(self.max_concentration, dtype=np.float64)


def _stain_matrix(od: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """Estimate a 3x2 stain-colour basis from optical-density pixels.

    Pixels with near-zero density carry no stain information and are dropped;
    the two stain directions are the extreme angles in the plane spanned by the
    top two principal components of the remainder.
    """
    od_flat = od.reshape(-1, 3)
    od_hat = od_flat[np.all(od_flat > beta, axis=1)]
    if od_hat.shape[0] < 10:
        raise ValueError("too few tissue pixels to estimate stain matrix")

    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = od_hat @ eigvecs[:, 1:3]  # top-2 principal directions

    angles = np.arctan2(plane[:, 1], plane[:, 0])
    lo, hi = np.percentile(angles, alpha), np.percentile(angles, 100 - alpha)
    v_lo = eigvecs[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = eigvecs[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])

    # Order the two stains consistently so re-projection is deterministic.
    he = np.array([v_lo, v_hi]).T if v_lo[0] > v_hi[0] else np.array([v_hi, v_lo]).T
    return he / (np.linalg.norm(he, axis=0, keepdims=True) + _EPS)


def _concentrations(od: np.ndarray, he: np.ndarray) -> np.ndarray:
    """Solve od = he @ c for per-pixel stain concentrations c. Returns (2, N)."""
    od_flat = od.reshape(-1, 3).T
    return np.linalg.lstsq(he, od_flat, rcond=None)[0]


def fit_macenko(sample_rgb: list[np.ndarray]) -> MacenkoReference:
    """Fit a Macenko reference from a list of RGB training images.

    A few hundred images is ample; the reference is a robust aggregate of stain
    colour, not a per-image quantity.
    """
    hes, maxcs = [], []
    for rgb in sample_rgb:
        od = _to_od(rgb)
        try:
            he = _stain_matrix(od)
        except (ValueError, np.linalg.LinAlgError):
            continue
        c = _concentrations(od, he)
        hes.append(he)
        maxcs.append(np.percentile(c, 99, axis=1))

    if not hes:
        raise ValueError("could not fit Macenko reference from any sample image")

    he_ref = np.mean(hes, axis=0)
    he_ref /= np.linalg.norm(he_ref, axis=0, keepdims=True) + _EPS
    return MacenkoReference(stain_matrix=he_ref.tolist(), max_concentration=np.mean(maxcs, axis=0).tolist())


def normalise_macenko(rgb: np.ndarray, ref: MacenkoReference) -> np.ndarray:
    """Re-render an image with its own stain concentrations on the reference basis."""
    od = _to_od(rgb)
    he = _stain_matrix(od)
    c = _concentrations(od, he)

    maxc = np.percentile(c, 99, axis=1, keepdims=True) + _EPS
    c_norm = c / maxc * ref.maxc[:, None]  # rescale to the reference dynamic range

    od_norm = (ref.he @ c_norm).T.reshape(rgb.shape)
    return _from_od(od_norm)


@dataclass
class ReinhardReference:
    """Per-channel Lab mean and std of the training split."""

    mean: list
    std: list

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "ReinhardReference":
        return cls(**json.loads(Path(path).read_text()))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab

    return rgb2lab(rgb / 255.0)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    from skimage.color import lab2rgb

    return np.clip(lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)


def fit_reinhard(sample_rgb: list[np.ndarray]) -> ReinhardReference:
    labs = np.concatenate([_rgb_to_lab(r).reshape(-1, 3) for r in sample_rgb], axis=0)
    return ReinhardReference(mean=labs.mean(0).tolist(), std=labs.std(0).tolist())


def normalise_reinhard(rgb: np.ndarray, ref: ReinhardReference) -> np.ndarray:
    lab = _rgb_to_lab(rgb)
    mean, std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + _EPS
    lab_norm = (lab - mean) / std * np.asarray(ref.std) + np.asarray(ref.mean)
    return _lab_to_rgb(lab_norm)
