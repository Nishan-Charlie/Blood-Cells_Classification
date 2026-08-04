# Modelling pipeline design

Date: 2026-08-01
Status: approved

## Context

Data preparation and exploratory analysis are complete. `notebooks/01_data_preparation.ipynb`
acquired and verified the 41,621-image MLL23 corpus, built the manifest, and persisted a
deterministic stratified 70/15/15 split. `notebooks/02_dataset_analysis.ipynb` tested the
central premise on frozen ImageNet features and found the lineage structure to be **local,
not global** (silhouette ~= 0, k-NN lineage agreement ~= 0.78 against a ~0.45 chance
baseline).

The modelling half does not exist. This spec covers everything from the model definition to
the significance tests.

## Goals

Deliver the five experiments CLAUDE.md requires, under identical splits and backbones, with
results reportable in the dissertation:

1. Flat single-head 18-class classifier (baseline)
2. Full hierarchical model (proposed)
3. Ablation: remove the imbalance term
4. Ablation: remove the hierarchy
5. Stain-normalised input

Plus a backbone comparison (MobileNet / ResNet / ConvNeXt / ViT), the hierarchical confusion
analysis, Grad-CAM saliency maps, and paired significance testing.

## Non-goals

- Re-running data preparation. The manifest, splits, and embeddings on `D:\MLL23` are treated
  as fixed inputs.
- Updating the `.tex` sources. Two wording corrections are identified below and left for a
  separate writing pass.
- Full 5x4 experiment matrix. The backbone comparison is a short screening run; the five arms
  are trained properly on the screen winner only.

## Module design

All reusable logic lives in `src/`, following the existing repository convention. The
notebooks are thin drivers that import and call it.

### `src/cache.py`

A uint8 memmap cache of decoded images at native 288x288 resolution.

Motivation is measured, not assumed. Benchmarked dataloader throughput on the target machine
(RTX 4070 Laptop, 8.6 GB) is **178 img/s at `num_workers=4`**, and *lower* at
`num_workers=8` (119 img/s) from worker contention. That is ~2.7 min/epoch of TIFF decode
with the GPU idle. The pipeline is data-bound, not compute-bound, so backbone choice barely
moves wall-clock and no amount of batch-size tuning helps.

The cache stores **decoded pixels at native resolution**, not preprocessed tensors. This is
deliberate: `transforms.py` rotates at 288 before downsampling to 224 so resampling happens
once rather than compounding. Caching at 224 would break that property. Caching at 288
preserves the augmentation semantics exactly while removing only the decode cost.

Two caches are built:

| Cache | Contents | Size | Consumer |
|---|---|---:|---|
| `images_288.npy` | raw decoded RGB | ~10.4 GB | arms 1-4 |
| `images_288_reinhard.npy` | Reinhard-normalised RGB | ~10.4 GB | arm 5 |

Arm 5 gets its own cache because Reinhard normalisation costs ~6-10 ms/image on the fly,
which would push that arm back to data-bound and make it slower than the others for reasons
unrelated to the science. Pre-normalising keeps all five arms at the same throughput, so a
runtime difference between arms is never a confound.

Row order is the manifest order, and an accompanying `.json` sidecar records the manifest
row count and a hash of the path column. `MLL23Dataset` verifies the sidecar against the
manifest it was handed and falls back to direct TIFF reads if they disagree, so a stale cache
degrades to correct-but-slow rather than silently serving the wrong images.

D: has 112 GB free; 21 GB of cache is not a space concern.

### `src/models.py`

```
input 224x224x3
      |
   timm backbone (pretrained, num_classes=0)  ->  pooled features, D-dim
      |
      +--- head_lineage:  Linear(D -> 3)     -- logits1
      +--- head_fine:     Linear(D -> 18)    -- logits2
```

`HierarchicalClassifier` carries a `mode` field. In `"hier"` mode both heads are returned;
in `"flat"` mode the lineage head is not constructed and only `logits2` is returned. One
class therefore serves the flat baseline, the hierarchical model, and both ablations, which
is what stops the arms from drifting apart.

The lineage head is a **regulariser, not a gate**. A hard gate -- routing each sample to a
per-lineage sub-classifier -- would assume lineages form separable clusters, which notebook
02 measured and rejected (silhouette ~= 0). The soft coupling below is consistent with the
"local, not global" finding: it lets lineage information shape the shared representation
without asserting global separability.

### `src/losses.py`

Three pieces:

- `class_balanced_weights(counts, beta)` -- Cui et al. effective-number reweighting,
  `(1 - beta) / (1 - beta^n)`, normalised to mean 1 so the loss scale stays comparable
  across arms.
- `FocalLoss(gamma, weight)` -- Lin et al., accepting the class-balanced weights.
- `HierarchicalLoss` -- the combined objective:

```
L = lambda_lin * L(logits1, y1) + lambda_fine * L(logits2, y2) + lambda_cons * L_consistency
```

`L_consistency` is what makes this hierarchical rather than merely multi-task. It marginalises
the fine posterior into a lineage posterior using `FINE_TO_LINEAGE` (already declared in
`hierarchy.py`) and penalises disagreement with the lineage head's own prediction. Without
it the two heads are free to contradict each other and CLAUDE.md's claim that "the coarse
decision regularises the fine one" has no mechanism behind it.

`lambda_cons` is held small (0.1) for the reason given above: the empirical support is for
local structure, so consistency is a nudge, not a constraint.

Two boolean switches implement the ablations:

- `use_hierarchy=False` -> `lambda_lin = lambda_cons = 0`, leaving fine-level loss only.
- `use_imbalance=False` -> plain unweighted cross-entropy; no class-balanced weights, no focal term.

### `src/metrics.py`

Accuracy, macro precision, macro recall, macro F1, balanced accuracy, per-class F1, and the
confusion matrix -- as CLAUDE.md requires, with macro F1 and balanced accuracy as headline.

Beyond the spec, every error is decomposed into **correct / within-lineage / cross-lineage**.
This is the measurement that actually tests the dissertation's thesis: a lineage-aware model
should shift errors from cross-lineage to within-lineage even when overall accuracy moves
little. Flat accuracy cannot show that, and within-lineage errors are clinically mild while
cross-lineage errors are severe.

A minority-class subset view reports the same metrics restricted to the eight classes at or
below `MINORITY_THRESHOLD` (988 down to 33 images), since minority behaviour is the point of
the dissertation.

### `src/engine.py`

A single config-driven trainer. `ExperimentConfig` is a frozen dataclass holding backbone,
arm name, seed, epochs, learning rate, batch size, the two ablation booleans, augmentation
policy, mixing kind, and which cache to read. `fit(config)` is the only entry point.

- AMP (fp16) with `GradScaler`.
- AdamW, cosine schedule with linear warmup.
- **Model selection on validation macro-F1, never accuracy.** At 260:1 imbalance an
  accuracy-selected checkpoint is one that has learned to ignore reactive lymphocytes
  entirely.
- Deterministic seeding of Python, NumPy, and Torch per run.
- Per-epoch checkpointing and an append-only results CSV, so a long unattended run is
  resumable after an interruption.

### `src/gradcam.py`

Grad-CAM over the final backbone stage. CNNs hook the last convolutional block directly. ViT
hooks the final block's norm layer with a reshape transform that maps the token sequence back
to a 14x14 spatial grid -- without that step Grad-CAM produces meaningless output on
transformers.

### `src/stats.py`

Paired t-test across seeds with Cohen's d effect size, plus a per-arm mean +/- std table.

### `src/experiments.py`

Declares the arms and the screen as data, and runs the matrix. Also the CLI entry point for
unattended execution.

## Experiment arms

Every arm is the same `ExperimentConfig` with different field values.

| Arm | `mode` | `use_hierarchy` | `use_imbalance` | cache |
|---|---|---|---|---|
| 1. Flat baseline | `flat` | no | yes | raw |
| 2. Full hierarchical | `hier` | yes | yes | raw |
| 3. Ablation - imbalance | `hier` | yes | **no** | raw |
| 4. Ablation - hierarchy | `flat` | **no** | yes | raw |
| 5. Stain-normalised | `hier` | yes | yes | **reinhard** |

**Arms 1 and 4 are the same configuration.** The hierarchy ablation applied to the proposed
model *is* the flat baseline. They are trained once and referenced twice, and the write-up
must say so rather than implying five independent training runs. This leaves four distinct
configurations.

Stain normalisation uses **Reinhard, not Macenko**. `stain.py`'s own docstring records that
Macenko is ~20x more expensive and it re-estimates a stain matrix per image; at the measured
throughput that would starve the GPU. Macenko remains available for the qualitative figure in
notebook 02.

## Training protocol

- Input 224x224 for every backbone including ViT, per the resolution decision already
  recorded in CLAUDE.md and `config.py`.
- Backbones initialised from ImageNet weights, fully fine-tuned.
- The class imbalance is **not resampled away** -- it is handled in the loss. This is the
  dissertation's subject matter, not an obstacle to be removed.
- Phase 1 screens four backbones for ~6 epochs each; the winner on validation macro-F1
  carries Phase 2.
- Phase 2 trains the four distinct configurations for ~20 epochs at 3 seeds each = 12 runs.

## Significance testing

CLAUDE.md specifies a "paired t-test across validation folds", but `splits.py` produces a
single fixed 70/15/15 split and there are no folds. Paired samples come instead from
**retraining each arm at 3 seeds on the fixed split**, pairing arms by seed.

This keeps the split identical across arms, which the ablation design depends on, and is
cheaper than k-fold. The trade-off is that it measures training variance rather than data
variance, and 3 seeds is a small n for a t-test -- effect sizes are reported alongside
p-values so the reader is not asked to lean on the p-value alone.

## Deviations from CLAUDE.md, for the writing pass

1. "Paired t-test across validation folds" should become "paired t-test across N training
   seeds on a fixed split". There are no folds.
2. `Data_Analysis.tex` still specifies 384x384 for Vision Transformers. The resolution was
   resolved to 224 for every backbone; this was already flagged in CLAUDE.md and remains
   uncorrected in the `.tex`.

## Implementation findings

Four non-obvious things surfaced during implementation. Each was a silent
failure - code that ran and produced plausible output while being wrong - so
each is recorded here and commented at the site in the code.

### 1. Mixed precision must be bf16, not fp16

An fp16 screening run reached validation macro-F1 0.80 at epoch 2 and then went
**NaN at epoch 3**, staying NaN for the rest of the run. NaN weights never
recover, so the remaining epochs were wasted; only the checkpoint-on-best-F1
policy preserved a usable model at all.

Switched to **bfloat16**, which keeps float32's exponent range and so cannot
overflow or underflow the way fp16 does. The RTX 4070 is Ada (compute 8.9) with
native bf16 support. `GradScaler` is disabled under bf16 - it exists to stop fp16
gradient underflow, which bf16 does not suffer.

Two supporting changes: the loss is computed in float32 regardless of autocast
dtype (the class-balanced weights span a 175x range and the focal term
exponentiates a log-probability), and a non-finite loss now skips the optimiser
step rather than poisoning the weights.

Related: the consistency term's `clamp_min(1e-8)` is a **no-op under fp16** -
float16's smallest subnormal is ~6e-8, so 1e-8 rounds to exactly zero. Guard
constants must be checked against the dtype they will actually run in.

### 2. `backbone.num_features` is not the head input width

For `mobilenetv3_large_100`, timm reports `num_features = 960` - the width before
`conv_head` - while the actual pooled output with `num_classes=0` is **1280**,
because MobileNetV3 expands through conv_head before its classifier. Building
heads from the attribute fails at the first forward pass with a shape mismatch.

`models.py` measures the width with one dummy forward pass at construction
instead, which is correct for every architecture.

### 3. Grad-CAM on ViT must not hook the final block

timm's ViT pools with `global_pool='token'`: the classifier reads **only the
class token**. At the final block the 196 patch tokens no longer influence the
output, so their gradient is *exactly zero* - and Grad-CAM needs the patch
tokens, since the class token has no spatial position.

Hooking `blocks[-1]` therefore produces an all-zero map, which min-max
normalisation renders as a flat image that still looks like a legitimate figure.
Measured gradient magnitude: **0.0 at `blocks[-1]`, 6.6e-2 at
`blocks[-1].norm1`**. The latter is the input to the final block's attention,
where patch tokens still route into the class token, and is what `gradcam.py`
hooks.

Separately, `register_full_backward_hook` never fires on a timm ViT `Block`. The
gradient is captured with a tensor hook on the activation instead, which behaves
identically across CNNs and ViTs. Grad-CAM now raises if it captured no
gradient rather than returning a blank map.

### 4. Memmaps must not cross a Windows process boundary

Windows DataLoader workers are spawned, not forked, so the Dataset is pickled
into each one - and pickling an open `np.memmap` serialises all 10 GB per
worker. `MLL23Dataset` holds the cache **path** and maps it lazily inside each
worker, with `__getstate__` dropping any open mapping before pickling.

### Measured throughput

| Configuration | Throughput |
|---|---:|
| TIFF decode, `num_workers=4` | 178 img/s |
| TIFF decode, `num_workers=8` | 119 img/s (worker contention) |
| Full ResNet-50 training step, no cache | 82 img/s, 5.95 min/epoch |
| Cached, warm page cache | ~850 img/s |

Cache build took 13.9 min for both caches (3.7 min raw, 10.2 min Reinhard),
against a 40 min estimate.

## Post-hoc: why the first matrix (v1) was invalid

The first complete matrix produced macro F1 ~0.85 but the arm comparison did not measure the
methods. Four findings from the run histories, archived in `archive/v1_baseline/`.

### The confound: run length determined score

`corr(epochs_run, test_macro_f1) = +0.87`.

| Arm | Epochs survived | Test macro F1 |
|---|---|---:|
| hierarchical | 37, 20, 29 | 0.853 |
| flat_baseline | 14, 13, 35 | 0.845 |
| stain_norm | 21, 14, 12 | 0.824 |
| no_imbalance | 13, 12, 12 | 0.824 |

The arm ranking is nearly the ranking of epoch counts. Two mechanisms produced this:

1. **The cosine schedule was truncated.** It is defined over `cfg.epochs`, which was set to
   50, but early stopping fired at 12-37. A run halting at epoch 12 stopped at **86% of peak
   LR** - mid-training, never annealed. The 6-epoch screen runs, whose cosine completed,
   scored *higher* on test (ViT 0.864) than any 50-epoch main run (0.853). Training longer
   made results worse, purely through this interaction.
2. **Early stopping fired on noise.** Validation macro-F1 moved with sd **0.021** between
   consecutive epochs (minority F1: sd 0.038), while the effects under study were 0.008-0.028.
   Patience-6 on that signal is close to a random stopping rule.

The second point is the deeper one: **the measurement noise exceeded the effect size**, so no
number of seeds would have made the v1 comparison conclusive.

### The hierarchy was inert by the end of training

Final validation lineage accuracy **98.9%**, with both auxiliary losses collapsed:

```
train_lineage     = 0.00074
train_consistency = 0.00068
```

Lineage is the *easy* axis. The hard problem is discrimination *within* a lineage - myeloid
maturation stages, and the rare lymphoid types. A lineage-aware auxiliary task therefore
regularises an axis the model has already solved, which explains why the hierarchical arm did
not improve minority-class F1. This is a design insight, not just a tuning issue: if the
hierarchy is to help the tail, the auxiliary signal must target a distinction that is still
unresolved late in training.

### Two regularisers were implemented but never active

Every v1 run used `mix_kind="none"` and `aug_policy="basic"` while training loss fell to
**0.005** against a validation plateau of 0.84 - plain memorisation.

Worse, `aug_policy` was a **dead config field**: `build_loaders` never passed it to
`MLL23Dataset`, so every run trained on the basic policy regardless of what its config
recorded. The `aug_policy` column in v1's `summary.csv` is fiction. Fixed; the parameter is
now threaded through.

## v2 protocol changes

| Change | Rationale |
|---|---|
| Early stopping **off**, fixed 30-epoch budget | Cosine always completes; identical anneal for every arm |
| **Weight EMA** (decay 0.999), evaluated instead of live weights | Halves selection-metric noise |
| **TTA** over four flip views | Cells have no canonical orientation; flips are label-preserving |
| **RandAugment + CutMix** enabled | Counters the observed memorisation |
| `aug_policy` threaded to the dataset | It previously did nothing |

Validated on one config before committing to the full matrix (hierarchical / ViT / seed 0,
identical data and seed):

| Metric | v1 | v2 | Δ |
|---|---:|---:|---:|
| test macro F1 | 0.8695 | 0.8809 | +0.0114 |
| test balanced accuracy | 0.8797 | 0.8929 | +0.0132 |
| test minority macro F1 | 0.8201 | 0.8370 | +0.0169 |
| cross-lineage error | 0.0133 | 0.0119 | −0.0014 |
| within-lineage error | 0.0413 | 0.0307 | −0.0106 |
| **val macro-F1 epoch-to-epoch sd** | **0.0211** | **0.0097** | **−54%** |

Cost: ~101 s/epoch versus v1's ~60 s, from RandAugment on the CPU dataloader. Roughly 10 h
for 12 runs rather than the 6-8 h originally estimated. Actual: **24.2 GPU-hours**, inflated by
occasional multi-minute epoch stalls (one epoch took 65 minutes against a ~100 s median),
attributable to page-cache pressure - the 10.4 GB cache sits against ~10 GB of free RAM.

### Final v2 results (12 runs, 3 seeds per arm, 30 epochs each)

| Arm | Macro F1 | Balanced acc | Minority F1 | Minority recall | Cross-lineage |
|---|---:|---:|---:|---:|---:|
| hierarchical | 0.8792 ± 0.0028 | 0.8936 | 0.8400 | 0.7987 | 0.0115 |
| flat_baseline | 0.8742 ± 0.0060 | 0.8832 | 0.8233 | 0.7761 | 0.0119 |
| no_imbalance | 0.8733 ± 0.0093 | 0.8620 | 0.7799 | 0.7226 | 0.0104 |
| stain_norm | 0.8714 ± 0.0031 | 0.8832 | 0.8206 | 0.7753 | 0.0127 |

Every run completed exactly 30 epochs, so the run-length confound is eliminated by
construction rather than merely reduced.

**Two v1 conclusions reversed sign.** v1 reported the hierarchy reducing minority F1 by 0.028
and balanced accuracy by 0.010; both flip positive under the corrected protocol (+0.017,
+0.010). The v1 write-up's statement that the rare-cell-type claim was "not supported" was an
artifact of the flat baseline being the arm least penalised by schedule truncation.

The hierarchy now leads on all four primary metrics with consistent moderate-to-large effect
sizes (d = +0.57 to +1.08), but **none reach p < 0.05 at n = 3** - power is roughly 15-20%, so
about 10-13 seeds would be required. The imbalance term remains the dominant component
(removing it costs 0.053 minority recall, d = -2.28).

### Methodological finding: macro F1 is insensitive to imbalance interventions

Removing the imbalance term moves macro precision **+0.017** and macro recall **-0.032**, so
macro F1 nets **-0.006 (p = 0.90)** while balanced accuracy - which is macro recall, and has no
cancelling term - moves the full **-0.032**. Minority recall moves -0.053.

Macro F1 is therefore a poor headline metric *and a poor model-selection criterion* for this
study, despite CLAUDE.md designating it as both. Plain accuracy is worse still: it *rises*
(+0.0035) when imbalance handling is removed, which is the failure mode the dissertation
exists to study.

Re-selecting checkpoints on validation balanced accuracy was evaluated without re-training, by
re-reading the per-epoch histories: it would change the selected epoch in **5 of 10 runs** but
gain only **+0.008** validation minority F1 on average. Real, small, and not worth a re-run -
recorded here so the decision is not revisited blindly.

## Not implemented: further improvements, in priority order

These follow from the v1 diagnosis and are the natural next steps if v2 still leaves the
minority-class claim unsupported.

**Make the hierarchy target an axis that is still hard.** The lineage head saturates at 98.9%,
so it stops contributing gradient long before training ends. Two options:

* **Conditional inference** - factor the prediction as `p(y2|x) = p(y1|x) · p(y2|y1,x)` so the
  coarse decision genuinely constrains the fine one, rather than two heads coexisting on a
  shared trunk and being nudged toward agreement.
* **Retarget the auxiliary task** at maturation stage within the myeloid branch, which is
  where the residual confusion actually lives, instead of at lineage.

**Attack the tail directly.** Class-balanced weights plus focal loss is a relatively weak
long-tail treatment. Stronger, well-established alternatives:

* **Decoupled training (cRT)** - learn the representation with instance-balanced sampling,
  then retrain *only* the classifier with class-balanced sampling. Consistently among the
  strongest long-tail methods and a much larger lever than reweighting alone.
* **Logit adjustment** or **LDAM margin loss** - enforce larger margins for rare classes.

**Raise statistical power.** Three seeds gives 2 degrees of freedom. Five seeds, combined with
the halved measurement noise from EMA, would make the paired tests meaningfully informative
rather than merely indicative.

## Risks

- **3 seeds is a thin basis for a t-test.** Mitigated by reporting effect sizes and per-seed
  numbers, not just p-values.
- **Reactive lymphocytes (n=33) leave ~23 training images.** Per-class F1 for this class will
  be noisy across seeds. Report it, do not hide it in the macro average alone.
- **A stale cache would silently serve wrong images.** Mitigated by the sidecar check
  described above, which falls back to direct reads on mismatch.
- **Phase 1 screening at 6 epochs may not rank backbones the same way a full run would.**
  Stated as a limitation; the screen selects a backbone, it does not constitute the backbone
  comparison result.
