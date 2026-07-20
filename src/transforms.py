"""Image transforms.

Two regimes, per the proposal:

* **Evaluation** - deterministic. Bilinear downsample 288 -> 224, scale to
  [0, 1], normalise with ImageNet statistics.
* **Training** - the evaluation pipeline preceded by online geometric and
  photometric augmentation, applied *more aggressively to minority classes*.

Rotation happens at the native 288 px resolution before the downsample, so the
resampling to 224 is performed once rather than compounded. Rotation of a square
frame leaves empty corners; these are filled with white, which matches the
bright background of a Pappenheim-stained smear, so the fill does not introduce
a colour the network would not otherwise see.
"""

from __future__ import annotations

from torchvision.transforms import v2
import torch

from . import config
from .hierarchy import BY_IDX

#: Fill colour for rotation corners: the bright background of a stained smear.
_ROTATION_FILL = 255


def _finalise() -> list:
    """Resize to the model input size, convert to float, normalise."""
    return [
        v2.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
        v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]


def eval_transform() -> v2.Compose:
    """Deterministic pipeline for validation and test."""
    return v2.Compose([v2.ToImage(), *_finalise()])


def train_transform(minority: bool = False) -> v2.Compose:
    """Augmenting pipeline for training.

    Args:
        minority: Apply the intensified regime. Rotation range and colour
            jitter are scaled by ``config.MINORITY_AUG_MULTIPLIER``. Blood cells
            have no canonical orientation, so widening rotation toward a full
            circle is label-preserving rather than distorting.
    """
    scale = config.MINORITY_AUG_MULTIPLIER if minority else 1.0
    degrees = min(config.MAX_ROTATION_DEGREES * scale, 180.0)
    jitter = min(config.JITTER_STRENGTH * scale, 0.5)

    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(
                degrees=degrees,
                interpolation=v2.InterpolationMode.BILINEAR,
                fill=_ROTATION_FILL,
            ),
            v2.ColorJitter(brightness=jitter, contrast=jitter),
            *_finalise(),
        ]
    )


def is_minority(y2: int) -> bool:
    """Whether a fine class qualifies for the intensified augmentation regime."""
    return BY_IDX[y2].expected_count <= config.MINORITY_THRESHOLD


def minority_class_names() -> list[str]:
    return [c.name for c in BY_IDX.values() if is_minority(c.idx)]


def denormalise(t: torch.Tensor) -> torch.Tensor:
    """Invert normalisation for display. Accepts (C,H,W) or (B,C,H,W)."""
    mean = torch.tensor(config.IMAGENET_MEAN, device=t.device).view(-1, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)
