"""The experiment matrix, declared as data.

Two phases:

**Phase 1 - backbone screen.** Four backbones, few epochs, hierarchical config,
one seed. Selects the backbone Phase 2 uses. This is a *selection* run, not the
backbone comparison result: 6 epochs need not rank backbones the way a full run
would, and the write-up should say so.

**Phase 2 - the five required experiments.** Four distinct configurations at
three seeds each.

Why four configurations and not five
------------------------------------
CLAUDE.md lists five experiments, but two of them are the same model:

* Experiment 1 is the flat single-head 18-class baseline.
* Experiment 4 is "the hierarchical model with the hierarchy removed" - which
  *is* a flat single-head 18-class classifier.

They are trained once and reported twice. The alternative - training the
identical configuration under two names - would burn a fifth of the compute
budget to produce two draws from the same distribution and invite the reader to
mistake them for independent evidence. :data:`ARMS` records the aliasing
explicitly so the write-up cannot lose it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from . import engine, splits
from .engine import ExperimentConfig

#: Seeds for the paired significance tests. CLAUDE.md asks for a paired t-test
#: "across validation folds", but splits.py produces one fixed split and there
#: are no folds; paired samples come from retraining on the fixed split instead.
#: Three is thin for a t-test - stats.py reports effect sizes alongside p-values
#: so the reader is not asked to lean on the p-value alone.
SEEDS: tuple[int, ...] = (0, 1, 2)

SCREEN_BACKBONES: tuple[str, ...] = ("mobilenet", "resnet", "convnext", "vit")


def screen_configs(epochs: int = 6) -> list[ExperimentConfig]:
    """Phase 1: the same hierarchical config on each backbone, one seed."""
    return [
        ExperimentConfig(
            arm=f"screen_{b}", backbone=b, mode="hier",
            use_hierarchy=True, use_imbalance=True,
            epochs=epochs, seed=0,
            # Fixed budget so the cosine schedule completes; every backbone then
            # gets the same anneal and the comparison is on the backbone alone.
            use_early_stopping=False,
            use_ema=True, tta=True,
        )
        for b in SCREEN_BACKBONES
    ]


#: The four distinct Phase 2 configurations, keyed by arm name. ``reported_as``
#: records which numbered experiment(s) from CLAUDE.md each one answers.
ARMS: dict[str, dict] = {
    "flat_baseline": {
        "mode": "flat", "use_hierarchy": False, "use_imbalance": True, "stain_norm": False,
        "reported_as": ["1. Flat baseline", "4. Ablation - hierarchy removed"],
        "note": "Experiments 1 and 4 are the same model; trained once, reported twice.",
    },
    "hierarchical": {
        "mode": "hier", "use_hierarchy": True, "use_imbalance": True, "stain_norm": False,
        "reported_as": ["2. Full hierarchical model (proposed)"],
        "note": "The proposed model.",
    },
    "no_imbalance": {
        "mode": "hier", "use_hierarchy": True, "use_imbalance": False, "stain_norm": False,
        "reported_as": ["3. Ablation - imbalance term removed"],
        "note": "Plain unweighted cross-entropy; no class-balanced weights, no focal term.",
    },
    "stain_norm": {
        "mode": "hier", "use_hierarchy": True, "use_imbalance": True, "stain_norm": True,
        "reported_as": ["5. Stain-normalised input"],
        "note": "Reinhard-normalised cache. Macenko is ~20x slower and re-fits per image.",
    },
}


def main_configs(backbone: str = "resnet", epochs: int = 30,
                 seeds: tuple[int, ...] = SEEDS) -> list[ExperimentConfig]:
    """Phase 2: every arm at every seed, on one backbone.

    The defaults here encode four corrections to the first matrix, which was
    invalid for reasons unrelated to the methods under test:

    * **Early stopping off, fixed budget.** The cosine schedule is defined over
      ``epochs``; stopping early truncated the anneal, leaving runs at up to 86%
      of peak LR. Run length then correlated with score at r = +0.87, so the arm
      ranking largely measured the stopping rule rather than the method.
    * **Weight EMA on.** Raw validation macro-F1 moved by sd ~0.021 between
      consecutive epochs while the effects under study were 0.008-0.028.
    * **TTA on.** Four flip views; blood cells have no canonical orientation.
    * **Stronger augmentation and mixing.** Training loss reached 0.005 against
      a validation plateau of 0.84 - the model was memorising, and the two
      regularisers already implemented were never switched on.
    """
    return [
        ExperimentConfig(
            arm=arm, backbone=backbone, epochs=epochs, seed=seed,
            mode=spec["mode"],
            use_hierarchy=spec["use_hierarchy"],
            use_imbalance=spec["use_imbalance"],
            stain_norm=spec["stain_norm"],
            aug_policy="randaugment",
            mix_kind="cutmix",
            mix_prob=0.5,
            mix_alpha=1.0,      # CutMix convention; MixUp would use ~0.2
            use_ema=True,
            tta=True,
            use_early_stopping=False,
        )
        for arm, spec in ARMS.items()
        for seed in seeds
    ]


def arm_table() -> pd.DataFrame:
    """The arm definitions as a table, for the dissertation's methods chapter."""
    return pd.DataFrame([
        {
            "arm": arm,
            "mode": s["mode"],
            "hierarchy": "yes" if s["use_hierarchy"] else "no",
            "imbalance term": "yes" if s["use_imbalance"] else "no",
            "stain norm": "Reinhard" if s["stain_norm"] else "none",
            "reported as": "; ".join(s["reported_as"]),
        }
        for arm, s in ARMS.items()
    ])


def run_matrix(configs: list[ExperimentConfig], *, skip_completed: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """Train every configuration, skipping runs already in the summary CSV.

    Skipping is what makes an unattended multi-hour matrix resumable: an
    interruption costs the run in flight, not the ones already finished.
    """
    split_df = splits.load()
    done = engine.completed_runs() if skip_completed else set()

    rows = []
    for i, cfg in enumerate(configs, 1):
        if cfg.run_id in done:
            print(f"[{i}/{len(configs)}] skip {cfg.run_id} (already in summary.csv)")
            continue
        print(f"[{i}/{len(configs)}] {cfg.run_id}", flush=True)
        rows.append(engine.fit(cfg, split_df, verbose=verbose))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLL23 experiment matrix.")
    parser.add_argument("--phase", choices=("screen", "main", "smoke"), required=True)
    parser.add_argument("--backbone", default="resnet",
                        help="Phase 2 backbone; normally the Phase 1 winner.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.phase == "screen":
        cfgs = screen_configs(epochs=args.epochs or 6)
    elif args.phase == "main":
        cfgs = main_configs(
            backbone=args.backbone,
            epochs=args.epochs or 30,
            seeds=tuple(args.seeds) if args.seeds else SEEDS,
        )
    else:
        # Smoke: one tiny run of each arm to prove every code path executes.
        cfgs = [
            replace(c, epochs=2, limit_train=600, arm=f"smoke_{c.arm}")
            for c in main_configs(epochs=2, seeds=(0,))
        ]

    df = run_matrix(cfgs)
    if not df.empty:
        cols = ["run_id", "val_macro_f1", "test_macro_f1",
                "test_balanced_accuracy", "test_cross_lineage_error", "total_minutes"]
        print("\n" + df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
