"""Evaluation metrics.

Plain accuracy is explicitly rejected as a headline number here. At 260:1
imbalance a classifier that ignores reactive lymphocytes (33 images) entirely
loses ~0.08% accuracy, so accuracy cannot see the behaviour the dissertation is
about. **Macro F1 and balanced accuracy are the headline metrics**, and model
selection is on macro F1.

Beyond the metrics CLAUDE.md lists, this module decomposes every error into
**correct / within-lineage / cross-lineage**. That decomposition is the direct
test of the dissertation's thesis: a lineage-aware model should move errors from
cross-lineage to within-lineage even when overall accuracy barely shifts.
Clinically the two are not equivalent - confusing a myelocyte with a
metamyelocyte (adjacent maturation stages) is mild, while calling a lymphocyte a
myeloblast is severe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .hierarchy import BY_IDX, CLASSES, FINE_NAMES, FINE_TO_LINEAGE, LINEAGES, NUM_FINE_CLASSES

#: Fine-class indices receiving the minority treatment, i.e. the eight classes at
#: or below config.MINORITY_THRESHOLD (988 smudge cells down to 33 reactive
#: lymphocytes). These are the classes the dissertation is really about.
MINORITY_IDX: tuple[int, ...] = tuple(
    c.idx for c in CLASSES if c.expected_count <= config.MINORITY_THRESHOLD
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str = "") -> dict:
    """Accuracy, macro precision/recall/F1, and balanced accuracy.

    ``zero_division=0`` throughout: a rare class can receive no predictions at
    all in an early epoch, and the correct score for that is 0, not a warning and
    a NaN that propagates into the macro average.
    """
    m = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    return {f"{prefix}{k}": v for k, v in m.items()}


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-fine-class precision, recall, F1 and support, in maturation order.

    Row order follows the class index, which encodes the myeloid maturation
    continuum - so the table reads in the same order as the confusion matrix and
    adjacent rows are biologically adjacent stages.
    """
    labels = list(range(NUM_FINE_CLASSES))
    return pd.DataFrame({
        "class_name": list(FINE_NAMES),
        "lineage": [BY_IDX[i].lineage for i in labels],
        "precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "support": [int((y_true == i).sum()) for i in labels],
        "is_minority": [i in MINORITY_IDX for i in labels],
    })


def minority_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro metrics restricted to the eight minority classes.

    Computed over the *samples* whose true class is a minority class. This is a
    recall-oriented view: it answers "of the rare cells present, how many did we
    find and correctly name", which is the clinically meaningful question. It is
    not symmetric with a precision view, and is labelled ``minority_*`` rather
    than presented as an overall score.
    """
    mask = np.isin(y_true, MINORITY_IDX)
    if not mask.any():
        return {"minority_macro_f1": float("nan"), "minority_recall": float("nan"),
                "minority_n": 0}

    yt, yp = y_true[mask], y_pred[mask]
    return {
        "minority_macro_f1": float(
            f1_score(yt, yp, labels=list(MINORITY_IDX), average="macro", zero_division=0)
        ),
        "minority_recall": float(
            recall_score(yt, yp, labels=list(MINORITY_IDX), average="macro", zero_division=0)
        ),
        "minority_n": int(mask.sum()),
    }


def hierarchical_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Decompose predictions into correct / within-lineage / cross-lineage.

    The headline of the hierarchical analysis. Every prediction falls into
    exactly one bucket:

    * **correct** - right fine class.
    * **within-lineage error** - wrong fine class, right lineage. Clinically
      mild; typically adjacent maturation stages.
    * **cross-lineage error** - wrong lineage entirely. Clinically severe.

    ``within_error_share`` is the diagnostic to watch: among errors only, the
    fraction that stayed inside the correct lineage. A lineage-aware model should
    raise this even if total error is unchanged, and that is a claim flat
    accuracy cannot express.
    """
    lut = np.asarray(FINE_TO_LINEAGE)
    lin_true, lin_pred = lut[y_true], lut[y_pred]

    correct = y_true == y_pred
    wrong = ~correct
    within = wrong & (lin_true == lin_pred)
    cross = wrong & (lin_true != lin_pred)

    n = len(y_true)
    n_wrong = int(wrong.sum())
    return {
        "correct": float(correct.mean()),
        "within_lineage_error": float(within.sum() / n),
        "cross_lineage_error": float(cross.sum() / n),
        # Share *of errors*, not of all samples: undefined when there are none.
        "within_error_share": float(within.sum() / n_wrong) if n_wrong else float("nan"),
        "lineage_accuracy": float((lin_true == lin_pred).mean()),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y1_true: np.ndarray | None = None,
                         y1_pred: np.ndarray | None = None) -> dict:
    """Every scalar metric for one set of predictions, as a flat dict.

    Args:
        y_true, y_pred: Fine-class labels and predictions.
        y1_true, y1_pred: Lineage labels and predictions from the lineage *head*.
            Optional: the hierarchical error decomposition derives lineage from
            the fine prediction regardless, so the arms remain comparable. When
            supplied, the head's own lineage accuracy is reported separately as
            ``head_lineage_accuracy`` - it can differ from the derived one, and
            that gap is itself informative about head disagreement.
    """
    out = {}
    out.update(classification_metrics(y_true, y_pred))
    out.update(minority_metrics(y_true, y_pred))
    out.update(hierarchical_errors(y_true, y_pred))

    if y1_true is not None and y1_pred is not None:
        out["head_lineage_accuracy"] = float((y1_true == y1_pred).mean())
        out["head_lineage_macro_f1"] = float(
            f1_score(y1_true, y1_pred, average="macro", zero_division=0)
        )
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, *, normalise: bool = True) -> pd.DataFrame:
    """Confusion matrix in fine-class index order.

    Index order is the maturation continuum, never alphabetical or
    frequency-sorted, so adjacent-stage confusions - the clinically expected
    error mode - appear adjacent to the diagonal and lineages form contiguous
    blocks.

    Args:
        normalise: Row-normalise to per-class recall. Essential here: raw counts
            are unreadable when one class has 8,606 samples and another 33.
    """
    labels = list(range(NUM_FINE_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        # A class absent from this split has a zero row; leave it zero rather
        # than producing NaN, which would blank the row in the heatmap and read
        # as "no data" instead of "no samples".
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return pd.DataFrame(cm, index=list(FINE_NAMES), columns=list(FINE_NAMES))


def lineage_confusion(y_true: np.ndarray, y_pred: np.ndarray, *, normalise: bool = True) -> pd.DataFrame:
    """3x3 lineage-level confusion, derived from the fine predictions."""
    lut = np.asarray(FINE_TO_LINEAGE)
    cm = confusion_matrix(lut[y_true], lut[y_pred], labels=list(range(len(LINEAGES)))).astype(float)
    if normalise:
        rs = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    return pd.DataFrame(cm, index=list(LINEAGES), columns=list(LINEAGES))
