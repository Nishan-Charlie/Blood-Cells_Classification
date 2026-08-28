"""Is myeloid error structured along the maturation continuum?

Tests whether errors among the neutrophil-maturation classes are predominantly
+/- one developmental stage. If so, the labels impose discrete boundaries on a
continuum and an ordinal-aware objective is justified; if errors are scattered,
they are not.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(r"c:\Users\nisha\Desktop\Freelance_Research")
sys.path.insert(0, str(REPO))

from src import engine, splits                       # noqa: E402
from src.engine import ExperimentConfig              # noqa: E402
from src.models import HierarchicalClassifier        # noqa: E402
from src.hierarchy import FINE_NAMES, FINE_TO_LINEAGE  # noqa: E402

# The neutrophil maturation chain, in true developmental order.
# Class 9 (atypical promyelocytes) is a pathological variant beside stage 8,
# not a stage of its own, so it is excluded from the ordinal chain.
CHAIN = [7, 8, 10, 11, 12, 13]
STAGE = {c: i for i, c in enumerate(CHAIN)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    split_df = splits.load()
    all_true, all_pred = [], []

    for seed in (0, 1, 2):
        cfg = ExperimentConfig(arm="hierarchical", backbone="vit", seed=seed,
                               mode="hier", use_hierarchy=True, use_imbalance=True,
                               aug_policy="randaugment", mix_kind="cutmix",
                               mix_alpha=1.0, use_ema=True, tta=True)
        ck = REPO / "checkpoints" / cfg.run_id / "best.pt"
        loaders = engine.build_loaders(cfg, split_df)
        model = HierarchicalClassifier(backbone="vit", mode="hier", pretrained=False).to(DEV)
        model.load_state_dict(torch.load(ck, weights_only=True)["model"])
        yt, yp, _, _ = engine.predict(model, loaders["test"], DEV, tta=True)
        all_true.append(yt); all_pred.append(yp)
        del model, loaders; torch.cuda.empty_cache()
        print(f"  seed {seed} done", flush=True)

    yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
    lut = np.asarray(FINE_TO_LINEAGE)

    print("\n" + "=" * 62)
    print("1. GLOBAL ERROR STRUCTURE (3 seeds pooled)")
    print("=" * 62)
    err = yt != yp
    print(f"total test predictions : {len(yt)}")
    print(f"errors                 : {err.sum()}  ({err.mean():.4f})")
    within = err & (lut[yt] == lut[yp])
    print(f"  within-lineage       : {within.sum()}  ({within.sum()/err.sum():.3f} of errors)")
    print(f"  cross-lineage        : {(err & ~within).sum()}  ({(err & ~within).sum()/err.sum():.3f} of errors)")

    print("\n" + "=" * 62)
    print("2. ERRORS INSIDE THE MATURATION CHAIN")
    print("=" * 62)
    in_chain = np.isin(yt, CHAIN) & np.isin(yp, CHAIN)
    ch_err = in_chain & err
    print(f"predictions with both true and pred in chain : {in_chain.sum()}")
    print(f"  of which errors                            : {ch_err.sum()}")

    d = np.array([STAGE[p] - STAGE[t] for t, p in zip(yt[ch_err], yp[ch_err])])
    print("\nsigned stage distance (pred - true):")
    for k in sorted(Counter(d).keys()):
        n = Counter(d)[k]
        print(f"  {k:+d} stage{'s' if abs(k)!=1 else ' '} : {n:5d}  ({n/len(d):.3f})  {'#'*int(60*n/len(d))}")
    adj = (np.abs(d) == 1).sum()
    print(f"\nadjacent (|d| = 1) : {adj}/{len(d)} = {adj/len(d):.3f} of in-chain errors")
    print(f"|d| <= 1           : {(np.abs(d)<=1).sum()/len(d):.3f}")
    print(f"|d| >= 3           : {(np.abs(d)>=3).sum()/len(d):.3f}")

    print("\n" + "=" * 62)
    print("3. WHERE DOES EACH CHAIN CLASS SEND ITS ERRORS?")
    print("=" * 62)
    for c in CHAIN:
        m = (yt == c) & err
        if not m.sum():
            continue
        tgt = Counter(yp[m]).most_common(3)
        tot = m.sum()
        parts = ", ".join(f"{FINE_NAMES[t]} {n/tot:.0%}" for t, n in tgt)
        print(f"{FINE_NAMES[c]:24s} (n_err={tot:4d}) -> {parts}")

    print("\n" + "=" * 62)
    print("4. SHARE OF ALL ERRORS THAT ARE ADJACENT-STAGE CONFUSIONS")
    print("=" * 62)
    adj_mask = np.zeros(len(yt), bool)
    idx = np.where(ch_err)[0]
    adj_mask[idx[np.abs(d) == 1]] = True
    print(f"adjacent-stage errors / all errors : {adj_mask.sum()}/{err.sum()} = {adj_mask.sum()/err.sum():.3f}")
    print("(these are the mildest errors clinically, and are currently penalised")
    print(" identically to a lymphoid/myeloid confusion)")


if __name__ == "__main__":
    main()
