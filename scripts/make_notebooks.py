"""Generate notebooks 03-05 as thin drivers over src/.

Written as a generator script rather than by hand so the notebooks can be
regenerated deterministically and kept free of stale execution counts and
embedded output. Run from the repository root:

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def nb_from(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    """Build a notebook from (kind, source) pairs. kind is 'md' or 'code'."""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "mri-diffuser", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


# --------------------------------------------------------------------------- #
# 03 - model, loss, and training mechanics
# --------------------------------------------------------------------------- #
NB03 = [
    ("md", """# 03 — Model, objective, and training mechanics

What this notebook is for: showing that the **model and the objective do what
they claim**, before any results are reported. Nothing here is a result. It is
the evidence that the machinery underneath Notebook 04 is sound.

Four things are demonstrated:

1. The architecture, and the parameter cost of each backbone.
2. The class-balanced weights — that they actually favour the rare classes.
3. The hierarchical consistency term — that it is *doing something*, rather than
   sitting inert at zero.
4. A deliberate overfitting test on a tiny subset. A model that cannot drive the
   loss to ~0 on 200 images has a bug, and no amount of training on 29,000 will
   fix it.

All code lives in `src/`. This notebook imports it, so what is demonstrated here
is the same code the experiments run — not a re-implementation that can drift."""),

    ("code", """%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import torch

from src import cache, config, engine, experiments, losses, metrics, splits, viz
from src.models import HierarchicalClassifier, BACKBONES, count_parameters
from src.hierarchy import CLASSES, FINE_NAMES

viz.apply_style()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")"""),

    ("md", """## 1. Architecture

One shared pretrained backbone feeding two linear heads. `mode="flat"` never
constructs the lineage head at all — an unused head would still appear in the
parameter count and in weight decay, which would quietly make the "identical
backbone" comparison less identical than it claims."""),

    ("code", """rows = []
for key, timm_name in BACKBONES.items():
    m = HierarchicalClassifier(key, mode="hier", pretrained=False)
    p = count_parameters(m)
    rows.append({
        "backbone": key,
        "timm name": timm_name,
        "feature dim": m.head_fine.in_features,
        "parameters (M)": round(p["total"] / 1e6, 1),
    })

pd.DataFrame(rows)"""),

    ("md", """Note the feature dimension is *measured*, not read from
`backbone.num_features`. For MobileNetV3 that attribute reports 960 — the width
before `conv_head` — while the real pooled output is 1280. Trusting it builds
heads of the wrong width and fails at the first forward pass."""),

    ("code", """# Both modes, one class. Shapes are what the loss and metrics assume downstream.
x = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

for mode in ("hier", "flat"):
    m = HierarchicalClassifier("resnet", mode=mode, pretrained=False).eval()
    out = m(x)
    y1_hat, y2_hat = m.predict(x)
    lin = tuple(out.logits1.shape) if out.logits1 is not None else None
    print(f"mode={mode:5s}  logits2={tuple(out.logits2.shape)}  logits1={lin}  "
          f"predict -> y1{tuple(y1_hat.shape)} y2{tuple(y2_hat.shape)}")"""),

    ("md", """In flat mode there is no lineage head, so `predict` *derives* the
lineage from the fine prediction through the fixed class→lineage map. That is
not a convenience: it is what makes the within- vs cross-lineage error analysis
computable for the baseline too, and therefore what makes the arms comparable on
the analysis the dissertation turns on."""),

    ("md", """## 2. Class-balanced weights

Cui et al.'s effective-number reweighting. The claim to check is simply that the
weights track rarity, and that they are normalised to mean 1 — without that
normalisation the weighted arm would train at a different effective learning
rate, and the imbalance ablation would confound reweighting with step size."""),

    ("code", """split_df = splits.load()
train_counts = (split_df[split_df.split == "train"]["y2"]
                .value_counts().reindex(range(18), fill_value=0))
counts_t = torch.tensor(train_counts.to_numpy(), dtype=torch.float32)

w = losses.class_balanced_weights(counts_t)

tbl = pd.DataFrame({
    "class": list(FINE_NAMES),
    "train count": train_counts.to_numpy(),
    "CB weight": w.numpy().round(3),
    "minority": [c.idx in metrics.MINORITY_IDX for c in CLASSES],
}).sort_values("train count")

print(f"weights mean = {w.mean():.4f}   (normalised to 1.0)")
print(f"max/min ratio = {w.max()/w.min():.1f}   vs raw count ratio "
      f"{train_counts.max()/train_counts.min():.1f}")
tbl"""),

    ("md", """The weight ratio is far smaller than the raw count ratio. That is
the point of the effective-number formulation: samples of one class overlap in
feature space, so the *n*-th sample adds less information than the first, and
pure inverse-frequency weighting over-corrects."""),

    ("md", """## 3. Is the hierarchy term doing anything?

The consistency term marginalises the fine posterior into lineage space and
penalises disagreement with the lineage head. Two sanity checks:

- On a **hierarchically consistent** prediction it should be ~0.
- On a **contradictory** one (fine head says lymphoid, lineage head says
  myeloid) it should be clearly positive.

If it cannot distinguish those, the term is inert and the model is only
multi-task, not hierarchical."""),

    ("code", """crit = losses.HierarchicalLoss(counts_t, use_hierarchy=True, use_imbalance=True)

# Class 0 is lymphoid (lineage 0); class 7 (myeloblast) is myeloid (lineage 1).
def one_hot_logits(idx, n, scale=10.0):
    z = torch.full((1, n), -scale)
    z[0, idx] = scale
    return z

agree    = crit.consistency(one_hot_logits(0, 3), one_hot_logits(0, 18))
disagree = crit.consistency(one_hot_logits(1, 3), one_hot_logits(0, 18))
uniform  = crit.consistency(torch.zeros(1, 3),    torch.zeros(1, 18))

print(f"heads agree      (lymphoid / typical lymphocyte) : {float(agree):.4f}")
print(f"heads contradict (myeloid  / typical lymphocyte) : {float(disagree):.4f}")
print(f"both uninformative (uniform)                     : {float(uniform):.4f}")"""),

    ("md", """### Ablation switches

Each arm is the same objective with different field values, so the switches must
visibly change which loss terms exist."""),

    ("code", """m = HierarchicalClassifier("resnet", mode="hier", pretrained=False)
out = m(torch.randn(8, 3, 224, 224))
y1 = torch.randint(0, 3, (8,))
y2 = torch.randint(0, 18, (8,))

for name, use_h, use_i in [
    ("full hierarchical", True,  True),
    ("- imbalance",       True,  False),
    ("- hierarchy",       False, True),
]:
    c = losses.HierarchicalLoss(counts_t, use_hierarchy=use_h, use_imbalance=use_i)
    loss, parts = c(out, y1, y2)
    print(f"{name:20s} total={float(loss):7.4f}  terms present: {sorted(parts.keys() - {'total'})}")"""),

    ("md", """## 4. Overfitting test

The strongest cheap check available. A correctly wired model must be able to
memorise a tiny subset: if the loss will not fall to near zero on 200 images,
something is broken in the data path, the labels, or the gradient flow, and
training on 29,000 images will only hide it.

This trains for a few dozen steps and takes well under a minute."""),

    ("code", """from torch.utils.data import DataLoader
from src.dataset import MLL23Dataset

cache_path = cache.validate_cache(cache.RAW_CACHE_PATH, split_df)
tiny = split_df[split_df.split == "train"].sample(200, random_state=0)

# train=False: no augmentation. Memorisation is the thing being tested, and
# augmentation would make the target move.
ds = MLL23Dataset(tiny, train=False, cache_path=cache_path)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

model = HierarchicalClassifier("resnet", mode="hier", pretrained=True).to(DEVICE)
crit  = losses.HierarchicalLoss(counts_t.to(DEVICE)).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
trace = []
for epoch in range(30):
    tot, n = 0.0, 0
    for xb, y1b, y2b in dl:
        xb, y1b, y2b = xb.to(DEVICE), y1b.to(DEVICE), y2b.to(DEVICE)
        loss, _ = crit(model(xb), y1b, y2b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss); n += 1
    trace.append(tot / n)
    if epoch % 5 == 0 or epoch == 29:
        print(f"  epoch {epoch:2d}  loss {trace[-1]:.5f}")

print(f"\\nfirst {trace[0]:.4f} -> last {trace[-1]:.5f}"
      f"   ({'PASS - model can memorise' if trace[-1] < trace[0] * 0.1 else 'FAIL - investigate'})")"""),

    ("code", """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trace, color=viz.LINEAGE_COLORS["Lymphoid"], lw=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.set_title("Overfitting test: 200 images, no augmentation", pad=10)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(config.ARTIFACT_DIR / "overfit_check.png")
plt.show()"""),

    ("md", """## 5. The experiment matrix

The arms as they will be run. Note that experiments 1 and 4 from the proposal are
the **same configuration** — "the hierarchical model with the hierarchy removed"
*is* the flat single-head baseline. It is trained once and reported twice; the
write-up must say so rather than implying five independent runs."""),

    ("code", """experiments.arm_table()"""),

    ("code", """cfgs = experiments.main_configs()
print(f"Phase 2: {len(cfgs)} runs = {len(experiments.ARMS)} distinct configs "
      f"x {len(experiments.SEEDS)} seeds")
for c in cfgs[:4]:
    print(f"  {c.run_id}")
print("  ...")"""),

    ("md", """---

## Summary

- The two-head architecture builds and runs on all four backbones, with the
  feature width measured rather than assumed.
- Class-balanced weights favour rare classes and are normalised to mean 1, so the
  imbalance ablation is not confounded with effective learning rate.
- The consistency term separates agreeing from contradicting heads, so the
  hierarchy is a mechanism rather than a label.
- The model memorises a 200-image subset, so the data path and gradient flow are
  sound.

Notebook 04 runs the experiments."""),
]


# --------------------------------------------------------------------------- #
# 04 - experiments
# --------------------------------------------------------------------------- #
NB04 = [
    ("md", """# 04 — Experiments

Runs the backbone screen and the five required experiments, then reports the
results with significance tests.

**Every run is the same `ExperimentConfig` with different field values.** There
is one training loop in `src/engine.py`; the arms differ by configuration, never
by code path. Divergent per-variant scripts drift, and a drifted baseline
invalidates every comparison made here.

Training is expensive. Both phases below are **resumable**: `run_matrix` skips
any run already present in `results/summary.csv`, so re-executing a cell after an
interruption costs only the unfinished runs."""),

    ("code", """%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import torch

from src import config, engine, experiments, metrics, splits, stats, viz

viz.apply_style()
pd.set_option("display.width", 200)
print("device:", "cuda" if torch.cuda.is_available() else "cpu")"""),

    ("md", """## Phase 1 — backbone screen

Four backbones, 6 epochs, identical hierarchical configuration, one seed.

This selects the backbone for Phase 2. It is **not** the backbone comparison
result: six epochs need not rank backbones the way a full run would, and the
write-up should present it as a selection step with that limitation stated."""),

    ("code", """# Skips anything already in results/summary.csv.
screen = experiments.run_matrix(experiments.screen_configs(epochs=6))"""),

    ("code", """summary = pd.read_csv(engine.RESULTS_DIR / "summary.csv")
scr = summary[summary.arm.str.startswith("screen_")].copy()

cols = ["backbone", "val_macro_f1", "test_macro_f1", "test_balanced_accuracy",
        "test_cross_lineage_error", "total_minutes"]
scr[cols].sort_values("val_macro_f1", ascending=False).reset_index(drop=True)"""),

    ("code", """# Selection is on validation macro F1 - never test, which stays untouched
# until the final evaluation, and never accuracy.
BEST_BACKBONE = scr.sort_values("val_macro_f1", ascending=False).iloc[0]["backbone"]
print(f"selected backbone: {BEST_BACKBONE}")"""),

    ("md", """## Phase 2 — the five required experiments

Four distinct configurations at three seeds each = 12 runs, all on the selected
backbone with identical splits.

Seeds provide the paired samples for the significance tests. CLAUDE.md asks for a
paired t-test "across validation folds", but `splits.py` produces one fixed split
and there are no folds — and that split must stay identical across arms for the
ablations to mean anything. Retraining at several seeds pairs the arms while
holding the partition fixed."""),

    ("code", """main = experiments.run_matrix(
    experiments.main_configs(backbone=BEST_BACKBONE, epochs=20)
)"""),

    ("code", """summary = pd.read_csv(engine.RESULTS_DIR / "summary.csv")
runs = summary[~summary.arm.str.startswith(("screen_", "smoke_"))].copy()
print(f"{len(runs)} runs across {runs.arm.nunique()} arms and {runs.seed.nunique()} seeds")

arm_stats = stats.arm_summary(runs)
arm_stats.round(4)"""),

    ("md", """### Headline comparison

Macro F1 and balanced accuracy are the headline metrics, not accuracy. At 260:1
imbalance a model that ignores reactive lymphocytes entirely loses ~0.08%
accuracy, so accuracy cannot see the behaviour this dissertation is about."""),

    ("code", """f = viz.arm_comparison(arm_stats, metric="test_macro_f1",
                       title="Test macro F1 by arm (mean ± sd over seeds)")
f.savefig(config.ARTIFACT_DIR / "arm_macro_f1.png")

f = viz.arm_comparison(arm_stats, metric="test_minority_macro_f1",
                       title="Test macro F1, minority classes only")
f.savefig(config.ARTIFACT_DIR / "arm_minority_f1.png")"""),

    ("md", """### Hierarchical error composition

The figure that carries the hierarchical claim. Errors are split into
within-lineage (clinically mild — typically adjacent maturation stages) and
cross-lineage (clinically severe).

A lineage-aware model should shrink the cross-lineage segment specifically,
moving error toward the milder kind. That shift can happen with little or no
change in overall accuracy, which is exactly why accuracy is the wrong headline."""),

    ("code", """f = viz.error_composition(arm_stats,
                          title="Prediction composition by arm (test set)")
f.savefig(config.ARTIFACT_DIR / "error_composition.png")"""),

    ("md", """### Significance

Paired t-tests across seeds against the flat baseline, with Cohen's d.

Read these with the sample size in mind: three seeds gives 2 degrees of freedom
and very little power. A non-significant result here is **weak evidence of no
difference, not evidence of no difference**. Effect sizes and per-seed values are
reported alongside so the conclusion never rests on the p-value alone."""),

    ("code", """comp = stats.comparison_table(runs, baseline="flat_baseline", metric="test_macro_f1")
for _, r in comp.iterrows():
    print(stats.format_result(r.to_dict()))
print()
comp.round(4)"""),

    ("code", """# The same tests on the minority-class metric and on cross-lineage error -
# the two quantities the dissertation actually argues about.
for metric in ("test_minority_macro_f1", "test_cross_lineage_error",
               "test_balanced_accuracy"):
    print(f"--- {metric} ---")
    for _, r in stats.comparison_table(runs, "flat_baseline", metric).iterrows():
        print("  " + stats.format_result(r.to_dict()))
    print()"""),

    ("md", """### Per-seed values

Shown in full rather than summarised, so the spread behind each mean is visible
and the reader can judge the t-tests for themselves."""),

    ("code", """runs.pivot_table(index="arm", columns="seed", values="test_macro_f1").round(4)"""),

    ("md", """### Training curves

Loss components, the selection metric, and how the error split evolves. If the
hierarchy is contributing, the lineage and consistency curves must actually move
— a flat consistency curve would mean the term is inert."""),

    ("code", """run_id = runs[(runs.arm == "hierarchical") & (runs.seed == 0)].iloc[0]["run_id"]
hist = pd.read_csv(engine.RESULTS_DIR / f"history_{run_id}.csv")

f = viz.training_curves(hist, title=f"Training: {run_id}")
f.savefig(config.ARTIFACT_DIR / "training_curves.png")"""),

    ("md", """---

## Summary

Results are written to `results/summary.csv` (one row per run) and
`results/history_*.csv` (per-epoch). Notebook 05 does the detailed evaluation:
confusion matrices, per-class behaviour, and Grad-CAM."""),
]


# --------------------------------------------------------------------------- #
# 05 - evaluation
# --------------------------------------------------------------------------- #
NB05 = [
    ("md", """# 05 — Evaluation, error analysis, and Grad-CAM

Detailed evaluation of the trained arms on the **test split**, which has been
untouched until now — every selection decision in Notebooks 03 and 04 was made
on validation.

Three parts:

1. Confusion matrices, read hierarchically.
2. Per-class behaviour, with the minority classes called out.
3. Grad-CAM — evidence that the model attends to nucleus, chromatin, and
   cytoplasm rather than to background or staining artefacts."""),

    ("code", """%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src import cache, config, engine, gradcam, metrics, splits, transforms as T, viz
from src.dataset import MLL23Dataset
from src.models import HierarchicalClassifier
from src.hierarchy import BY_IDX, CLASSES, FINE_NAMES, LINEAGES

viz.apply_style()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

split_df = splits.load()
summary = pd.read_csv(engine.RESULTS_DIR / "summary.csv")
runs = summary[~summary.arm.str.startswith(("screen_", "smoke_"))].copy()
runs[["run_id", "arm", "seed", "test_macro_f1"]].head()"""),

    ("md", """## Loading a trained arm

Checkpoints hold the weights selected on **validation macro F1**, not the final
epoch's weights."""),

    ("code", """def load_arm(arm: str, seed: int = 0):
    \"\"\"Restore the selected checkpoint for one arm, plus its config.\"\"\"
    row = runs[(runs.arm == arm) & (runs.seed == seed)].iloc[0]
    ckpt = torch.load(engine.CHECKPOINT_DIR / row["run_id"] / "best.pt",
                      weights_only=True)
    model = HierarchicalClassifier(row["backbone"], mode=row["mode"],
                                   pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    return model.eval(), row


def test_loader(stain_norm: bool = False, batch_size: int = 128):
    \"\"\"Test-split loader reading the cache the arm was trained against.\"\"\"
    want = cache.REINHARD_CACHE_PATH if stain_norm else cache.RAW_CACHE_PATH
    cp = cache.validate_cache(want, split_df)
    ds = MLL23Dataset(split_df[split_df.split == "test"], train=False, cache_path=cp)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=engine.NUM_WORKERS, pin_memory=True)


ARMS_TO_EVAL = ["flat_baseline", "hierarchical", "no_imbalance", "stain_norm"]
preds = {}
for arm in ARMS_TO_EVAL:
    model, row = load_arm(arm)
    dl = test_loader(stain_norm=bool(row["stain_norm"]))
    # tta must match what the run recorded, or the numbers here will not
    # reconcile with results/summary.csv.
    preds[arm] = engine.predict(model, dl, DEVICE, tta=bool(row.get("tta", False)))
    del model; torch.cuda.empty_cache()
    print(f"  {arm:16s} done  (tta={bool(row.get('tta', False))})")"""),

    ("code", """# Reconciliation check: recomputed metrics must match the recorded run.
# A mismatch means the notebook and the results table describe different models.
for arm in ARMS_TO_EVAL:
    y2t, y2p, _, _ = preds[arm]
    here = metrics.classification_metrics(y2t, y2p)["macro_f1"]
    there = float(runs[(runs.arm == arm) & (runs.seed == 0)].iloc[0]["test_macro_f1"])
    flag = "OK" if abs(here - there) < 1e-6 else "MISMATCH"
    print(f"  {arm:16s} notebook {here:.4f}  summary.csv {there:.4f}  [{flag}]")"""),

    ("md", """## 1. Confusion matrices

Class order is the **maturation continuum**, never alphabetical or
frequency-sorted. Adjacent maturation stages therefore sit adjacent to the
diagonal, so the clinically expected confusions appear as near-diagonal mass.
White rules mark the lineage boundaries: anything outside those blocks is a
severe cross-lineage error."""),

    ("code", """for arm in ("flat_baseline", "hierarchical"):
    y2t, y2p, _, _ = preds[arm]
    cm = metrics.confusion(y2t, y2p, normalise=True)
    f = viz.confusion_heatmap(cm, title=f"Confusion matrix — {arm} (row-normalised)")
    f.savefig(config.ARTIFACT_DIR / f"confusion_{arm}.png")
plt.show()"""),

    ("code", """# Lineage-level view: the 3x3 summary of where cross-lineage error lands.
for arm in ("flat_baseline", "hierarchical"):
    y2t, y2p, _, _ = preds[arm]
    print(f"--- {arm} ---")
    print(metrics.lineage_confusion(y2t, y2p).round(3).to_string())
    print()"""),

    ("md", """## 2. Hierarchical error decomposition

The direct test of the thesis. `within_error_share` is the number to watch: among
errors only, the fraction that stayed inside the correct lineage."""),

    ("code", """rows = []
for arm, (y2t, y2p, y1t, y1p) in preds.items():
    r = metrics.hierarchical_errors(y2t, y2p)
    r["arm"] = arm
    rows.append(r)

pd.DataFrame(rows).set_index("arm").round(4)"""),

    ("md", """## 3. Per-class behaviour

Macro averages hide exactly what matters here. This is the per-class view, with
minority classes marked."""),

    ("code", """tables = {}
for arm, (y2t, y2p, _, _) in preds.items():
    tables[arm] = metrics.per_class_f1(y2t, y2p)

f = viz.per_class_f1_comparison(
    {k: tables[k] for k in ("flat_baseline", "hierarchical")},
    title="Per-class F1: flat baseline vs hierarchical  (* = minority class)")
f.savefig(config.ARTIFACT_DIR / "per_class_f1.png")
plt.show()"""),

    ("code", """# Side-by-side per-class F1, sorted by support so the tail is visible.
comp = tables["flat_baseline"][["class_name", "support", "is_minority"]].copy()
for arm in ARMS_TO_EVAL:
    comp[arm] = tables[arm]["f1"].round(3)
comp["hier - flat"] = (comp["hierarchical"] - comp["flat_baseline"]).round(3)
comp.sort_values("support")"""),

    ("md", """**Reactive lymphocytes (33 images total, ~5 in test) deserve explicit
caution.** With a test support that small, per-class F1 moves in large
increments — a single prediction changing flips it substantially. Report the
number, but do not build an argument on its movement between arms."""),

    ("md", """## 4. Grad-CAM

CLAUDE.md treats these as a deliverable, not decoration: they are the evidence
that the model keys on cell morphology rather than on an acquisition confound. A
model with strong macro F1 and saliency sitting on the background has learned the
wrong thing, and only this figure reveals it.

Note on the ViT: its saliency is taken at `blocks[-1].norm1`, not at the final
block. timm's ViT pools with `global_pool='token'`, so at the last block the
patch tokens no longer influence the output and their gradient is exactly zero —
which would yield a flat map that still *looks* like a plausible figure."""),

    ("code", """model, row = load_arm("hierarchical")
cache_path = cache.validate_cache(cache.RAW_CACHE_PATH, split_df)
test_df = split_df[split_df.split == "test"]

# One correctly-classified example per lineage, plus the rare classes.
show_idx = []
for lin_i, lin in enumerate(LINEAGES):
    sub = test_df[test_df.y1 == lin_i]
    show_idx.extend(sub.sample(2, random_state=0).index.tolist())
for cls_idx in sorted(metrics.MINORITY_IDX)[:6]:
    sub = test_df[test_df.y2 == cls_idx]
    if len(sub):
        # .index[0] keeps the manifest row label, which is what indexes the cache.
        show_idx.append(sub.sample(1, random_state=0).index[0])

panels_ds = MLL23Dataset(test_df.loc[show_idx], train=False, cache_path=cache_path)
batch = torch.stack([panels_ds[i][0] for i in range(len(panels_ds))]).to(DEVICE)
true_y2 = [int(panels_ds[i][2]) for i in range(len(panels_ds))]"""),

    ("code", """with gradcam.GradCAM(model, head="fine") as cam:
    heat = cam(batch)                       # saliency for the predicted class

_, pred_y2 = model.predict(batch)
pred_y2 = pred_y2.cpu().numpy()

panels = []
for i in range(len(batch)):
    ok = "OK" if pred_y2[i] == true_y2[i] else "MISS"
    caption = (f"{BY_IDX[true_y2[i]].name}\\n-> {BY_IDX[int(pred_y2[i])].name} [{ok}]")
    panels.append((gradcam.overlay(batch[i], heat[i]), caption))

f = viz.gradcam_grid(panels, ncols=4,
                     title="Grad-CAM on the fine head (true -> predicted)")
f.savefig(config.ARTIFACT_DIR / "gradcam_fine.png")
plt.show()"""),

    ("md", """### Fine head vs lineage head

Comparing where the two heads look shows whether the coarse and fine decisions
rest on the same evidence. Divergent saliency would suggest the heads have
learned separate features, which would undercut the "coarse decision regularises
the fine one" argument."""),

    ("code", """with gradcam.GradCAM(model, head="lineage") as cam:
    heat_lin = cam(batch[:8])

panels = []
for i in range(8):
    panels.append((gradcam.overlay(batch[i], heat[i]),
                   f"fine: {BY_IDX[true_y2[i]].name}"))
    panels.append((gradcam.overlay(batch[i], heat_lin[i]),
                   f"lineage: {BY_IDX[true_y2[i]].lineage}"))

f = viz.gradcam_grid(panels, ncols=4, title="Fine head vs lineage head saliency")
f.savefig(config.ARTIFACT_DIR / "gradcam_heads.png")
plt.show()"""),

    ("md", """## 5. Worst confusions

The specific class pairs the model conflates most, checked against the
biologically expected error mode. Adjacent maturation stages confusing with each
other is the *expected* pattern; cross-lineage pairs in this list are the ones to
explain."""),

    ("code", """y2t, y2p, _, _ = preds["hierarchical"]
cm_counts = metrics.confusion(y2t, y2p, normalise=False).to_numpy()
np.fill_diagonal(cm_counts, 0)

pairs = []
for i in range(18):
    for j in range(18):
        if cm_counts[i, j] > 0:
            pairs.append({
                "true": FINE_NAMES[i],
                "predicted": FINE_NAMES[j],
                "count": int(cm_counts[i, j]),
                "share of true class": round(cm_counts[i, j] / max((y2t == i).sum(), 1), 3),
                "same lineage": BY_IDX[i].lineage == BY_IDX[j].lineage,
                "index gap": abs(i - j),
            })

pd.DataFrame(pairs).sort_values("count", ascending=False).head(15).reset_index(drop=True)"""),

    ("md", """---

## Summary

All figures are written to `artifacts/`. The claims this notebook supports:

- Where each arm's errors fall, hierarchically — the within- vs cross-lineage
  split, which is the dissertation's central measurement.
- How the arms differ on the rare classes specifically, with the caveat that the
  rarest class has a test support of ~5 and should not carry an argument.
- That the model attends to cell morphology rather than background, via Grad-CAM
  on both heads."""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in [("03_model_and_training", NB03),
                        ("04_experiments", NB04),
                        ("05_evaluation", NB05)]:
        path = NB_DIR / f"{name}.ipynb"
        nbf.write(nb_from(cells), path)
        print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
