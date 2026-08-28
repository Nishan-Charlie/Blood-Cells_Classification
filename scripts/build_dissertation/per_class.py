"""Per-class test metrics for the four arms, averaged over the three seeds.

Feeds Appendix B and the per-class discussion in Chapter 4. Reads the saved
best.pt checkpoints and re-runs test-set inference with the same 4-view TTA
used at training time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import engine, splits, metrics as M          # noqa: E402
from src.engine import ExperimentConfig               # noqa: E402
from src.models import HierarchicalClassifier         # noqa: E402
from src.hierarchy import FINE_NAMES                  # noqa: E402

ARMS = {
    "flat_baseline": dict(mode="flat", use_hierarchy=False, use_imbalance=True, stain_norm=False),
    "hierarchical":  dict(mode="hier", use_hierarchy=True,  use_imbalance=True, stain_norm=False),
    "no_imbalance":  dict(mode="hier", use_hierarchy=True,  use_imbalance=False, stain_norm=False),
    "stain_norm":    dict(mode="hier", use_hierarchy=True,  use_imbalance=True, stain_norm=True),
}
SEEDS = (0, 1, 2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

split_df = splits.load()
out = {}

for arm, flags in ARMS.items():
    per_seed = []
    for seed in SEEDS:
        cfg = ExperimentConfig(arm=arm, backbone="vit", seed=seed,
                               aug_policy="randaugment", mix_kind="cutmix",
                               mix_alpha=1.0, use_ema=True, tta=True, **flags)
        ckpt = REPO / "checkpoints" / cfg.run_id / "best.pt"
        if not ckpt.exists():
            print(f"  missing {ckpt}", flush=True)
            continue
        loaders = engine.build_loaders(cfg, split_df)
        model = HierarchicalClassifier(backbone=cfg.backbone, mode=cfg.mode, pretrained=False).to(DEV)
        model.load_state_dict(torch.load(ckpt, weights_only=True)["model"])
        y2t, y2p, _, _ = engine.predict(model, loaders["test"], DEV, tta=cfg.tta)
        tab = M.per_class_f1(y2t, y2p).set_index("class_name")
        per_seed.append(tab[["precision", "recall", "f1"]])
        del model, loaders
        torch.cuda.empty_cache()
        print(f"  {cfg.run_id} done", flush=True)
    if per_seed:
        stacked = pd.concat(per_seed)
        out[arm] = stacked.groupby(level=0).mean().reindex(list(FINE_NAMES))
        out[arm + "_sd"] = stacked.groupby(level=0).std().reindex(list(FINE_NAMES))

res = pd.DataFrame(index=list(FINE_NAMES))
for arm in ARMS:
    if arm in out:
        res[arm + "_f1"] = out[arm]["f1"].round(4)
        res[arm + "_recall"] = out[arm]["recall"].round(4)
res.to_csv(REPO / "results" / "per_class_by_arm.csv")
print()
print(res.to_string())
