"""Imbalance-robust hierarchical training objective.

Three components, assembled by :class:`HierarchicalLoss`:

* **Class-balanced weighting** (Cui et al. 2019) - reweight by the *effective
  number* of samples rather than raw frequency.
* **Focal loss** (Lin et al. 2017) - down-weight easy examples so the gradient
  is not dominated by the 8,606 myeloblasts.
* **Hierarchical consistency** - tie the fine head's implied lineage posterior
  to the lineage head's own prediction.

The imbalance is never resampled away. It is the dissertation's subject matter,
so it is handled in the objective and left intact in the data.

Ablation switches
-----------------
``use_hierarchy=False``  ->  lambda_lin = lambda_cons = 0, leaving fine-level loss only.
``use_imbalance=False``  ->  plain unweighted cross-entropy: no class-balanced
                             weights, no focal term.

Both ablations are therefore field values on one object rather than separate
loss classes, which is what keeps the arms honest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchy import FINE_TO_LINEAGE, NUM_LINEAGES


def class_balanced_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """Effective-number class weights (Cui et al. 2019).

    The idea: samples of the same class overlap in feature space, so the *n*-th
    sample of a class adds less new information than the first. The effective
    number of a class with n samples is ``(1 - beta^n) / (1 - beta)``, and the
    weight is its reciprocal. As ``beta -> 0`` this becomes uniform weighting; as
    ``beta -> 1`` it approaches inverse-frequency weighting.

    Args:
        counts: Per-class sample counts, indexed by class. Zeros are clamped to 1
            so an absent class yields a finite weight rather than a division by
            zero - a class missing from a split is a data problem to be surfaced
            by ``splits.check_split``, not silently turned into NaN here.
        beta: Re-weighting strength. 0.9999 is the value Cui et al. use for
            long-tailed sets of this size.

    Returns:
        Weights normalised to mean 1, so the loss magnitude stays comparable
        against the unweighted ablation. Without this normalisation the weighted
        arm would train at a different effective learning rate and the ablation
        would confound reweighting with step size.
    """
    counts = counts.clamp(min=1.0)
    effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    return weights / weights.mean()


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017), optionally class-weighted.

    Scales each sample's cross-entropy by ``(1 - p_t) ** gamma``, so confidently
    correct samples contribute almost nothing. Composes with class-balanced
    weights: the weight addresses *how many* samples a class has, the focal term
    addresses *how hard* each individual sample is.

    Args:
        gamma: Focusing strength. 0 recovers weighted cross-entropy; 2.0 is the
            value from the paper and the default here.
        weight: Per-class weights, typically from :func:`class_balanced_weights`.
        label_smoothing: Applied to the underlying cross-entropy. Small values
            help with the noisy rare classes without blunting the focal term.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # register_buffer, not a plain attribute: the weights must follow the
        # module across .to(device) and be captured in the checkpoint.
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.gamma == 0:
            return ce.mean()
        # p_t recovered from the *unweighted* log-probability of the true class:
        # the modulating factor should reflect how hard the sample is, not how
        # rare its class is - the class weight already carries that, and folding
        # it in twice would double-count.
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        focal = (1.0 - log_pt.exp()).pow(self.gamma)
        return (focal * ce).mean()


def fine_to_lineage_matrix(device=None) -> torch.Tensor:
    """(18, 3) one-hot matrix mapping fine classes to lineages.

    Right-multiplying a fine probability vector by this marginalises it into a
    lineage probability vector, since each fine class belongs to exactly one
    lineage. Built from ``hierarchy.FINE_TO_LINEAGE`` so the mapping has exactly
    one definition in the codebase.
    """
    m = torch.zeros(len(FINE_TO_LINEAGE), NUM_LINEAGES, device=device)
    for fine_idx, lineage_idx in enumerate(FINE_TO_LINEAGE):
        m[fine_idx, lineage_idx] = 1.0
    return m


class HierarchicalLoss(nn.Module):
    """The combined two-level objective.

    ::

        L = lambda_lin  * L(logits1, y1)
          + lambda_fine * L(logits2, y2)
          + lambda_cons * KL(marginalised fine posterior || lineage posterior)

    The consistency term is what makes this *hierarchical* rather than merely
    multi-task. It marginalises the fine posterior into a lineage posterior via
    :func:`fine_to_lineage_matrix` and penalises disagreement with the lineage
    head. Without it the two heads are free to contradict each other, and
    CLAUDE.md's claim that "the coarse decision regularises the fine one" has no
    mechanism behind it.

    ``lambda_cons`` is deliberately small (0.1). Notebook 02 found the lineage
    structure to be local rather than global, so consistency is applied as a
    nudge; forcing it hard would impose a global structure the data does not
    support.

    Args:
        class_counts: Per-fine-class training counts, for the class-balanced
            weights. Required when ``use_imbalance`` is True.
        use_hierarchy: False zeroes the lineage and consistency terms.
        use_imbalance: False replaces the whole imbalance treatment with plain
            unweighted cross-entropy.
        gamma: Focal focusing parameter.
        beta: Class-balanced re-weighting strength.
        lambda_lin, lambda_fine, lambda_cons: Term weights. The fine term stays
            at 1.0 - it is the task being evaluated.
    """

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        *,
        use_hierarchy: bool = True,
        use_imbalance: bool = True,
        gamma: float = 2.0,
        beta: float = 0.9999,
        lambda_lin: float = 0.3,
        lambda_fine: float = 1.0,
        lambda_cons: float = 0.1,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_hierarchy = use_hierarchy
        self.use_imbalance = use_imbalance
        self.lambda_lin = lambda_lin if use_hierarchy else 0.0
        self.lambda_fine = lambda_fine
        self.lambda_cons = lambda_cons if use_hierarchy else 0.0

        if use_imbalance:
            if class_counts is None:
                raise ValueError("class_counts is required when use_imbalance=True")
            w_fine = class_balanced_weights(class_counts, beta=beta)
            # Lineage counts are the sum of their members' counts. The lineage
            # level is imbalanced too (myeloid is ~63% of the corpus), so it gets
            # the same treatment rather than being left unweighted.
            w_lin = torch.zeros(NUM_LINEAGES)
            for fine_idx, lineage_idx in enumerate(FINE_TO_LINEAGE):
                w_lin[lineage_idx] += class_counts[fine_idx]
            w_lin = class_balanced_weights(w_lin, beta=beta)

            self.crit_fine = FocalLoss(gamma, w_fine, label_smoothing)
            self.crit_lin = FocalLoss(gamma, w_lin, label_smoothing)
        else:
            # The ablation: no weights, no focal term, no smoothing. Deliberately
            # the plainest possible objective, so any difference is attributable
            # to the imbalance treatment and nothing else.
            self.crit_fine = FocalLoss(gamma=0.0, weight=None, label_smoothing=0.0)
            self.crit_lin = FocalLoss(gamma=0.0, weight=None, label_smoothing=0.0)

        self.register_buffer("f2l", fine_to_lineage_matrix())

    def consistency(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """KL between the fine head's implied lineage belief and the lineage head's.

        The fine posterior is marginalised into lineage space and used as the
        *target*; gradients flow into both heads, pulling them toward agreement
        rather than forcing one to chase the other.
        """
        p_fine_as_lineage = F.softmax(logits2, dim=1) @ self.f2l
        log_q = F.log_softmax(logits1, dim=1)
        # clamp: a marginalised probability can reach exactly 0 for a lineage
        # whose classes all have vanishing posterior, and log(0) would poison
        # the batch with NaN.
        return F.kl_div(log_q, p_fine_as_lineage.clamp_min(1e-8), reduction="batchmean")

    def forward(self, out, y1: torch.Tensor, y2: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the total loss.

        Args:
            out: A :class:`~src.models.ModelOutput`.
            y1: Lineage targets, ``(B,)``.
            y2: Fine targets, ``(B,)``.

        Returns:
            ``(total, parts)`` where ``parts`` holds the detached scalar
            components for logging - the loss curves in notebook 03 need to show
            the three terms separately to demonstrate the hierarchy is doing
            something.
        """
        loss_fine = self.crit_fine(out.logits2, y2)
        total = self.lambda_fine * loss_fine
        parts = {"fine": float(loss_fine.detach())}

        # Both conditions matter: flat mode has no lineage head at all, and the
        # hierarchy ablation zeroes the weights even when the head exists.
        if self.use_hierarchy and out.logits1 is not None:
            loss_lin = self.crit_lin(out.logits1, y1)
            loss_cons = self.consistency(out.logits1, out.logits2)
            total = total + self.lambda_lin * loss_lin + self.lambda_cons * loss_cons
            parts["lineage"] = float(loss_lin.detach())
            parts["consistency"] = float(loss_cons.detach())

        parts["total"] = float(total.detach())
        return total, parts
