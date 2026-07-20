"""Hierarchical MixUp and CutMix.

MixUp and CutMix blend two examples and their labels. Because MLL23 has two
label levels, a mix must stay *consistent across levels*: the same mixing
coefficient lambda applies to both the lineage target y1 and the fine target y2,
so the coarse and fine losses see the same blend. Applying independent mixes per
level would let the loss disagree with itself about how much of each example is
present.

These operate on a whole batch after collation, so they live here rather than in
the per-image transform pipeline. The output labels are returned as
``(y_a, y_b, lam)`` triples per level; the loss computes
``lam * L(pred, y_a) + (1 - lam) * L(pred, y_b)`` rather than mixing hard labels,
which keeps the class-balanced weighting well defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MixedTargets:
    """A mixed batch and the two label sets to interpolate the loss between."""

    images: torch.Tensor
    y1_a: torch.Tensor
    y1_b: torch.Tensor
    y2_a: torch.Tensor
    y2_b: torch.Tensor
    lam: float

    def loss(self, criterion, logits1, logits2):
        """Interpolated two-level loss for a ``(logits1, logits2)`` prediction.

        ``criterion`` must accept ``(logits, target)`` and return a scalar; it is
        called four times (both levels, both label sets) and combined by lam.
        """
        l1 = self.lam * criterion(logits1, self.y1_a) + (1 - self.lam) * criterion(logits1, self.y1_b)
        l2 = self.lam * criterion(logits2, self.y2_a) + (1 - self.lam) * criterion(logits2, self.y2_b)
        return l1, l2


def _sample_lambda(alpha: float, rng: np.random.Generator) -> float:
    return float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0


def mixup(images, y1, y2, *, alpha: float = 0.2, rng: np.random.Generator | None = None) -> MixedTargets:
    """Convex-combine each image with a shuffled partner; share lam across levels."""
    rng = rng or np.random.default_rng()
    lam = _sample_lambda(alpha, rng)
    perm = torch.randperm(images.size(0), device=images.device)

    mixed = lam * images + (1 - lam) * images[perm]
    return MixedTargets(mixed, y1, y1[perm], y2, y2[perm], lam)


def cutmix(images, y1, y2, *, alpha: float = 1.0, rng: np.random.Generator | None = None) -> MixedTargets:
    """Paste a random patch from a partner image; lam is the *area* kept.

    lam is recomputed from the realised patch area (integer pixels rarely match
    the sampled ratio exactly), so the loss weighting matches what the network
    actually sees.
    """
    rng = rng or np.random.default_rng()
    lam = _sample_lambda(alpha, rng)
    perm = torch.randperm(images.size(0), device=images.device)

    _, _, h, w = images.shape
    cut = np.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = rng.integers(w), rng.integers(h)
    x1, x2 = np.clip([cx - cw // 2, cx + cw // 2], 0, w)
    y1b, y2b = np.clip([cy - ch // 2, cy + ch // 2], 0, h)

    out = images.clone()
    out[:, :, y1b:y2b, x1:x2] = images[perm, :, y1b:y2b, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2b - y1b) / (w * h))
    return MixedTargets(out, y1, y1[perm], y2, y2[perm], lam_adj)


def maybe_mix(images, y1, y2, *, kind: str = "none", p: float = 0.5,
              alpha: float = 0.2, rng: np.random.Generator | None = None):
    """Apply MixUp/CutMix with probability ``p``; otherwise pass through.

    Returns a :class:`MixedTargets` either way (an unmixed batch has ``lam=1`` and
    ``*_b == *_a``), so the training loop treats every batch uniformly.
    """
    rng = rng or np.random.default_rng()
    if kind == "none" or rng.random() > p:
        return MixedTargets(images, y1, y1, y2, y2, 1.0)
    if kind == "mixup":
        return mixup(images, y1, y2, alpha=alpha, rng=rng)
    if kind == "cutmix":
        return cutmix(images, y1, y2, alpha=alpha, rng=rng)
    raise ValueError(f"unknown mix kind {kind!r}")
