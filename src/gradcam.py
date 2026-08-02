"""Grad-CAM saliency over the backbone's final feature stage.

CLAUDE.md treats these maps as a deliverable, not decoration: they are the
evidence that the model attends to nucleus, chromatin, and cytoplasm rather than
to background artefacts, staining blotches, or slide edges. A model with strong
macro F1 and saliency sitting on the background is a model that has learned an
acquisition confound, and only this figure can reveal that.

The ViT case
------------
Grad-CAM assumes activations are shaped ``(B, C, H, W)``. A ViT's final block
emits ``(B, N+1, D)`` - a sequence of patch tokens preceded by a class token.
Applied naively, Grad-CAM silently produces a meaningless strip rather than
failing, which is the worst outcome. :func:`_reshape_vit` drops the class token
and folds the remaining 196 patch tokens back into the 14x14 grid they came
from, so the ViT maps are comparable with the CNN ones.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import transforms as T


def _reshape_vit(t: torch.Tensor) -> torch.Tensor:
    """(B, N+1, D) token sequence -> (B, D, H, W) spatial map.

    At 224 px with 16 px patches there are 14x14 = 196 patch tokens plus one
    class token. The class token carries no spatial position, so it is dropped
    rather than folded into the grid.
    """
    if t.dim() != 3:
        return t
    tokens = t[:, 1:, :]                      # drop the class token
    b, n, d = tokens.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"{n} patch tokens is not a square grid; cannot reshape")
    return tokens.reshape(b, side, side, d).permute(0, 3, 1, 2)


def _target_layer(model) -> tuple[torch.nn.Module, bool]:
    """Pick the layer to hook, and say whether its output needs reshaping.

    Returns ``(layer, needs_reshape)``. Chosen by architecture family because
    timm's module names differ; falls back to the last module that produces a
    4-D output.
    """
    name = model.backbone_name
    bb = model.backbone

    if name.startswith("vit"):
        # NOT blocks[-1]. timm's ViT pools with global_pool='token', so the
        # classifier reads only the class token: at the *final* block the 196
        # patch tokens no longer influence the output and their gradient is
        # exactly zero. Since Grad-CAM needs the patch tokens (the class token
        # has no spatial position), hooking blocks[-1] yields an all-zero map
        # that min-max normalisation renders as a flat, plausible-looking image.
        #
        # blocks[-1].norm1 is the input to the final block's attention, where
        # patch tokens still route information into the class token, so the
        # gradient is non-zero. Measured on this model: gradient magnitude
        # 6.6e-2 here versus exactly 0.0 at blocks[-1].
        return bb.blocks[-1].norm1, True
    if name.startswith("resnet"):
        return bb.layer4[-1], False
    if name.startswith("convnext"):
        return bb.stages[-1], False
    if name.startswith("mobilenet"):
        return bb.blocks[-1], False

    # Generic fallback: deepest leaf module with parameters.
    candidates = [m for m in bb.modules() if isinstance(m, torch.nn.Conv2d)]
    if not candidates:
        raise ValueError(f"cannot infer a Grad-CAM target layer for {name!r}")
    return candidates[-1], False


class GradCAM:
    """Grad-CAM for :class:`~src.models.HierarchicalClassifier`.

    Usage::

        cam = GradCAM(model)
        heat = cam(x, class_idx=7)      # (B, 224, 224) in [0, 1]
        cam.close()

    Args:
        model: A trained classifier. Put it in eval mode; gradients are still
            required, so this cannot run under ``torch.no_grad``.
        head: ``"fine"`` targets the 18-class head, ``"lineage"`` the 3-class
            head. Comparing the two shows whether the coarse and fine decisions
            rest on the same evidence.
    """

    def __init__(self, model, *, head: str = "fine") -> None:
        self.model = model.eval()
        self.head = head
        self.layer, self.needs_reshape = _target_layer(model)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # Only a forward hook is registered. The gradient is captured by a hook
        # placed on the activation *tensor* from inside it - see _save_activation.
        self._handles = [self.layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _module, _inp, out) -> None:
        """Record the activation and attach a tensor hook for its gradient.

        A tensor hook is used rather than ``register_full_backward_hook`` on the
        module. The module hook is unreliable on container modules: on a timm ViT
        ``Block`` it never fires at all, so the gradient stayed None, the map came
        out uniformly zero, and min-max normalisation turned that into a flat
        array instead of an error. Grad-CAM silently returning a blank map is
        worse than crashing, since the figure still looks plausible.

        Hooking the tensor captures the gradient flowing through exactly the
        activation being used, which behaves identically for CNNs and ViTs.
        """
        self._activations = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def close(self) -> None:
        """Remove the hooks. Leaving them attached leaks memory across calls."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, x: torch.Tensor, class_idx=None) -> np.ndarray:
        """Saliency for a batch.

        Args:
            x: Normalised input batch, ``(B, 3, 224, 224)``.
            class_idx: Target class per sample. ``None`` uses each sample's own
                predicted class, which is what you want for "why did it say
                that"; pass the true label instead to ask "where would the
                evidence for the correct answer have been".

        Returns:
            ``(B, 224, 224)`` float array in [0, 1], upsampled to input size.
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out.logits1 if self.head == "lineage" else out.logits2
        if logits is None:
            raise ValueError("model has no lineage head; use head='fine'")

        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=logits.device)

        # Backprop the summed target logits: with one scalar per sample and no
        # cross-sample interaction in the heads, the batch's gradients stay
        # independent, so this is equivalent to looping and far cheaper.
        selected = logits.gather(1, class_idx[:, None]).sum()
        selected.backward()

        acts, grads = self._activations, self._gradients
        # Fail loudly rather than returning a blank map that reads as "the model
        # attends nowhere" when the real cause is a hook that never fired.
        if acts is None or grads is None:
            raise RuntimeError(
                f"Grad-CAM captured no {'activations' if acts is None else 'gradients'} "
                f"from {type(self.layer).__name__}; the hooked layer may not be on the "
                f"autograd path for this model."
            )
        if self.needs_reshape:
            acts, grads = _reshape_vit(acts), _reshape_vit(grads)

        # Channel importance = spatially averaged gradient; the map is the
        # ReLU'd weighted sum over channels.
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Per-sample min-max: absolute activation magnitude is not comparable
        # across images, only the relative spatial pattern is.
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        return ((cam - lo) / (hi - lo + 1e-8)).cpu().numpy()


def overlay(image: torch.Tensor, heat: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """Blend a saliency map over a denormalised image for display.

    Args:
        image: One normalised CHW tensor, as fed to the model.
        heat: One (H, W) map in [0, 1].
        alpha: Heatmap opacity. Kept below 0.5 so cell morphology stays visible
            underneath - a figure where the overlay hides the cell cannot
            support a claim about which structures the model used.

    Returns:
        (H, W, 3) float RGB in [0, 1].
    """
    import matplotlib

    rgb = T.denormalise(image).permute(1, 2, 0).cpu().numpy()
    # matplotlib.colormaps, not cm.get_cmap: the latter was removed in 3.9.
    colour = matplotlib.colormaps["inferno"](heat)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * colour, 0, 1)
