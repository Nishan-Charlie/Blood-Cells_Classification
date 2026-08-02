"""Backbone + two-head classifier.

Architecture
------------
::

    input 224x224x3
          |
       timm backbone (pretrained, num_classes=0)  ->  pooled features, D-dim
          |
          +--- head_lineage:  Linear(D -> 3)     -- logits1
          +--- head_fine:     Linear(D -> 18)    -- logits2

One class serves every experiment arm. ``mode="hier"`` returns both heads;
``mode="flat"`` never constructs the lineage head and returns only the fine
logits. The flat baseline and the hierarchy ablation are therefore literally the
same code path as the proposed model with one field changed, which is what stops
the four configurations from drifting apart.

Why the lineage head is a regulariser, not a gate
-------------------------------------------------
An obvious alternative is a hard gate: predict the lineage, then route the sample
to one of three per-lineage sub-classifiers. That design assumes lineages form
separable clusters. Notebook 02 measured exactly this on frozen ImageNet features
and found **lineage silhouette ~= 0** - the lineages do *not* separate globally -
while k-NN lineage agreement was ~0.78 against a ~0.45 chance baseline, meaning
the structure is real but **local**.

A hard gate would therefore be built on a premise the data rejects, and every
gate misfire would be an unrecoverable cross-lineage error. The soft coupling
used here - a shared trunk plus a consistency penalty in the loss - exploits the
local structure that does exist without asserting the global structure that does
not.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

from . import config
from .hierarchy import NUM_FINE_CLASSES, NUM_LINEAGES

#: timm names for the four backbones the proposal compares. Keys are the short
#: names used in configs, filenames, and figure labels.
BACKBONES: dict[str, str] = {
    "mobilenet": "mobilenetv3_large_100",
    "resnet": "resnet50",
    "convnext": "convnext_tiny",
    "vit": "vit_small_patch16_224",
}


@dataclass
class ModelOutput:
    """Predictions from one forward pass.

    ``logits1`` is ``None`` in flat mode - there is no lineage head to produce it.
    Callers must branch on that rather than assuming both levels are present.
    """

    logits2: torch.Tensor              # (B, 18) fine cell type
    logits1: torch.Tensor | None = None  # (B, 3) lineage, hierarchical mode only


class HierarchicalClassifier(nn.Module):
    """Shared pretrained backbone with one or two classification heads.

    Args:
        backbone: Key in :data:`BACKBONES`, or any timm model name.
        mode: ``"hier"`` for the two-head model, ``"flat"`` for the 18-class
            baseline.
        pretrained: Load ImageNet weights. Only ever False for fast tests -
            the whole design is a transfer-learning one.
        dropout: Applied to the pooled features before both heads. Rare classes
            (33 images) overfit readily; this is cheap insurance.
    """

    def __init__(
        self,
        backbone: str = "resnet",
        *,
        mode: str = "hier",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mode not in ("hier", "flat"):
            raise ValueError(f"unknown mode {mode!r}; expected 'hier' or 'flat'")

        self.mode = mode
        self.backbone_name = BACKBONES.get(backbone, backbone)

        # num_classes=0 strips timm's classifier and returns pooled features, so
        # the same call works for CNNs and ViTs without special-casing pooling.
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self._infer_feature_dim()

        self.dropout = nn.Dropout(dropout)
        self.head_fine = nn.Linear(feat_dim, NUM_FINE_CLASSES)
        # Constructed only in hierarchical mode: an unused head would still
        # appear in the parameter count and in weight decay, quietly making the
        # "identical backbone" comparison less identical than it claims.
        self.head_lineage = nn.Linear(feat_dim, NUM_LINEAGES) if mode == "hier" else None

    def _infer_feature_dim(self) -> int:
        """Measure the backbone's pooled output width with one dummy forward.

        ``backbone.num_features`` is *not* reliable for this. For
        ``mobilenetv3_large_100`` it reports 960 - the width before the
        ``conv_head`` - while the actual pooled output with ``num_classes=0`` is
        1280, because MobileNetV3 expands through conv_head before its
        classifier. Trusting the attribute builds heads of the wrong width and
        fails at the first forward pass with a shape mismatch.

        Probing sidesteps every such architecture-specific quirk, at the cost of
        one forward pass at construction time.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            dim = int(self.backbone(dummy).shape[-1])
        self.backbone.train(was_training)
        return dim

    def forward(self, x: torch.Tensor) -> ModelOutput:
        feats = self.dropout(self.backbone(x))
        logits2 = self.head_fine(feats)
        logits1 = self.head_lineage(feats) if self.head_lineage is not None else None
        return ModelOutput(logits2=logits2, logits1=logits1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: return ``(y1_hat, y2_hat)`` as a combined lineage + type call.

        In flat mode there is no lineage head, so the lineage prediction is
        *derived* from the fine prediction through the fixed class->lineage map.
        This is not a courtesy: it is what makes the hierarchical error
        decomposition in ``metrics.py`` computable for the baseline too, and
        hence what makes the two arms comparable on the within- vs cross-lineage
        analysis that the dissertation turns on.
        """
        from .hierarchy import FINE_TO_LINEAGE

        out = self(x)
        y2_hat = out.logits2.argmax(dim=1)
        if out.logits1 is not None:
            y1_hat = out.logits1.argmax(dim=1)
        else:
            lut = torch.as_tensor(FINE_TO_LINEAGE, device=y2_hat.device)
            y1_hat = lut[y2_hat]
        return y1_hat, y2_hat


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, for the backbone comparison table."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
