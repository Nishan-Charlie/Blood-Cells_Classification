"""Significance testing across seeds.

CLAUDE.md specifies a "paired t-test across validation folds". There are no
folds: ``splits.py`` produces one fixed deterministic 70/15/15 split, and that
split must stay identical across arms for the ablation comparison to mean
anything. Paired samples therefore come from **retraining each arm at several
seeds on the fixed split**, pairing arms by seed.

What this does and does not measure
-----------------------------------
Pairing by seed controls for initialisation and data-order variation, which is
the dominant source of run-to-run spread here. It does **not** measure
sensitivity to the data partition - a k-fold design would, at several times the
cost, and at the price of the rare classes (33 images) getting ~5 per fold.

With three seeds the t-test has 2 degrees of freedom and very little power.
Effect sizes are reported alongside p-values, and the per-seed values are always
available, so a reader is never asked to lean on the p-value alone. A
non-significant result at n=3 is weak evidence of no difference, not evidence of
no difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference over its own std.

    Uses the standard deviation *of the differences*, which is the appropriate
    denominator for a paired design.
    """
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def paired_test(df: pd.DataFrame, arm_a: str, arm_b: str,
                metric: str = "test_macro_f1") -> dict:
    """Paired t-test between two arms across shared seeds.

    Args:
        df: The summary CSV, one row per run.
        arm_a, arm_b: Arm names to compare. The difference reported is a - b.
        metric: Column to test. Defaults to test macro F1, the headline metric.

    Returns:
        Per-arm means and stds, the mean difference, t, p, Cohen's d, and n.
        A Wilcoxon signed-rank p-value is included as a distribution-free
        cross-check, since normality is untestable at n=3.
    """
    a = df[df.arm == arm_a].set_index("seed")[metric]
    b = df[df.arm == arm_b].set_index("seed")[metric]

    # Inner join on seed: only seeds where *both* arms ran can be paired.
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 2:
        return {"arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                "n_pairs": len(shared), "error": "need at least 2 shared seeds"}

    va, vb = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
    t, p = sps.ttest_rel(va, vb)

    out = {
        "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
        "n_pairs": len(shared),
        "mean_a": float(va.mean()), "std_a": float(va.std(ddof=1)),
        "mean_b": float(vb.mean()), "std_b": float(vb.std(ddof=1)),
        "mean_diff": float((va - vb).mean()),
        "t": float(t), "p_value": float(p),
        "cohens_d": cohens_d_paired(va, vb),
    }

    # Wilcoxon needs n>=some minimum to return anything meaningful and warns
    # loudly at tiny n; guard rather than emit a misleading number.
    if len(shared) >= 5:
        out["wilcoxon_p"] = float(sps.wilcoxon(va, vb).pvalue)
    return out


def arm_summary(df: pd.DataFrame, metrics: tuple[str, ...] = (
    "test_macro_f1", "test_balanced_accuracy", "test_accuracy",
    "test_minority_macro_f1", "test_cross_lineage_error",
)) -> pd.DataFrame:
    """Mean +/- std per arm across seeds, for the results table."""
    rows = []
    for arm, g in df.groupby("arm"):
        row = {"arm": arm, "n_seeds": len(g)}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("test_macro_f1_mean", ascending=False)


def comparison_table(df: pd.DataFrame, baseline: str = "flat_baseline",
                     metric: str = "test_macro_f1") -> pd.DataFrame:
    """Every arm tested against a common baseline.

    The default baseline is the flat classifier, which is what the dissertation's
    research question compares against.
    """
    arms = [a for a in df["arm"].unique() if a != baseline]
    return pd.DataFrame([paired_test(df, a, baseline, metric) for a in arms])


def format_result(r: dict) -> str:
    """One-line human-readable rendering of a paired test, for the write-up."""
    if "error" in r:
        return f"{r['arm_a']} vs {r['arm_b']}: {r['error']}"
    sig = "significant" if r["p_value"] < 0.05 else "not significant"
    return (f"{r['arm_a']} vs {r['arm_b']} on {r['metric']}: "
            f"{r['mean_a']:.4f} vs {r['mean_b']:.4f} "
            f"(diff {r['mean_diff']:+.4f}), "
            f"t({r['n_pairs'] - 1})={r['t']:.2f}, p={r['p_value']:.4f} [{sig}], "
            f"d={r['cohens_d']:.2f}, n={r['n_pairs']}")
