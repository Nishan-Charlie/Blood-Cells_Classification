# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A master's dissertation project by Arthiga Karthigesu: **"Lineage-Aware Hierarchical Deep Learning for Imbalanced Multi-Class Classification of Peripheral Blood Cells."**

The repository holds both the written dissertation (LaTeX sources and compiled PDFs) and the implementation. The "Research design" section below is transcribed from the approved proposal and is **authoritative** — [Data_Analysis.tex](Data_Analysis.tex) and [Proposal_Slides.tex](Proposal_Slides.tex) restate it in less detail, and where they disagree with this file, this file wins.

`Proposal.pdf` is password-protected and cannot be read by tooling — treat the transcribed section below as the accessible copy of record.

Not a git repository. There is no version control here — be careful with destructive edits.

## Code layout

```
src/
  config.py      paths, resolution, split fractions, augmentation strength
  hierarchy.py   the 18 classes, lineages, and index ordering  <- start here
  download.py    Zenodo fetch, MD5 verify, extract
  manifest.py    build/verify the image manifest
  splits.py      deterministic stratified 70/15/15
  transforms.py  288->224 pipeline, class-conditional augmentation
  dataset.py     MLL23Dataset yielding (image, y1, y2)
  viz.py         figures for the write-up
notebooks/
  01_data_preparation.ipynb   drives the above end to end; idempotent
artifacts/       generated figures
```

`hierarchy.py` is the keystone — it declares the class indices, their lineage mapping, and the published per-class counts used to verify the download. Its module-level `_validate()` runs at import and fails loudly if the invariants break.

**Data lives at `D:\MLL23`, not in the repo.** C: sits at ~97% capacity and the corpus is ~10 GB. Override with the `MLL23_ROOT` environment variable.

## Commands

```bash
# One-off: fetch the 41,621-image corpus (9.1 GB of archives, ~10 GB extracted).
# Resumable and idempotent - already-extracted classes are skipped.
python -c "from src.download import download_all; download_all()"

# Full data preparation: manifest, verification, splits, figures.
jupyter notebook notebooks/01_data_preparation.ipynb

# LaTeX (see below) - run each twice.
pdflatex -interaction=nonstopmode -halt-on-error Literature_Review.tex
```

There is no test suite yet. Correctness is currently enforced by assertions at import time (`hierarchy._validate`), count verification against published figures (`manifest.verify_manifest`), and split validation (`splits.check_split`) — keep adding to these rather than trusting a pipeline that merely runs.

## LaTeX build

Two documents are compiled from source. Both use `\begin{thebibliography}` inline, so **no BibTeX/biber run is needed** — but each still needs **two `pdflatex` passes** to resolve cross-references and (for the slides) the navigation bar.

```bash
pdflatex -interaction=nonstopmode -halt-on-error Literature_Review.tex   # run twice
pdflatex -interaction=nonstopmode -halt-on-error Proposal_Slides.tex     # run twice
```

TeX Live 2026 is installed at `/c/texlive/2026/bin/windows/`.

`Proposal_Slides.tex` `\includegraphics` two figure PDFs that must stay in the same folder: `cell_category.pdf` (the lineage tree) and `Methodology-1.pdf` (the pipeline diagram).

After a successful build, clean the intermediates rather than leaving them beside the sources:

```bash
rm -f Literature_Review.aux Literature_Review.log Literature_Review.out
rm -f Proposal_Slides.aux Proposal_Slides.log Proposal_Slides.nav \
      Proposal_Slides.out Proposal_Slides.snm Proposal_Slides.toc
```

`Data_Analysis.tex` is a standalone fragment that is not yet wired into a parent document.

## The research design the code must implement

**Research question.** Does a lineage-aware hierarchical classifier, combined with an imbalance-robust training objective, improve fine-grained classification of single peripheral blood cell images — especially for rare cell types — compared with a flat transfer-learning classifier on the same data?

### Dataset

MLL23 peripheral blood single-cell dataset (*Scientific Data*, 2025), Zenodo DOI [10.5281/zenodo.14277609](https://doi.org/10.5281/zenodo.14277609). 41,621 Pappenheim-stained single nucleated-cell images, **288×288 px TIFF** (25 µm × 25 µm), annotated by expert cytomorphologists into 18 classes.

The imbalance is the whole point of the dissertation — **do not silently resample it away.** Largest class (myeloblasts, 8,606) is ~260× the smallest (reactive lymphocytes, 33).

### Label hierarchy and exact class counts

`y1 ∈ {0,1,2}` = lineage, `y2 ∈ {0..17}` = fine cell type. Each image maps to a joint target `[y1, y2]`.

| Lineage (L1) | Cell type (L2) | Images |
|---|---|---:|
| **Lymphoid** | Typical lymphocytes | 5,532 |
| | Hairy cells | 3,265 |
| | Large granular lymphocytes | 1,849 |
| | Atypical lymphocytes (plasma cells) | 1,658 |
| | Smudge cells | 988 |
| | Neoplastic lymphocytes | 180 |
| | Reactive lymphocytes | 33 |
| **Myeloid** | Myeloblasts | 8,606 |
| | Segmented neutrophils | 7,170 |
| | Monocytes | 2,510 |
| | Eosinophil granulocytes | 2,448 |
| | Atypical promyelocytes | 2,033 |
| | Myelocytes | 747 |
| | Promyelocytes | 745 |
| | Band neutrophils | 687 |
| | Basophil granulocytes | 616 |
| | Metamyelocytes | 483 |
| **Erythroid** | Normoblasts | 2,071 |
| | **Total** | **41,621** |

Within the myeloid branch the cell types are ordered along the **maturation continuum** (myeloblast → promyelocyte → myelocyte → metamyelocyte → band → segmented neutrophil). This ordering is biologically meaningful — preserve it in class-index assignment and in confusion-matrix axis ordering, since adjacent-stage confusions are the clinically expected error mode.

### Preprocessing pipeline

1. **Resize/normalise** — 288×288 → **224×224 bilinear**, scale to [0,1], normalise with ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

   > **Measured 2026-07-20:** MLL23's own channel means are `[0.742, 0.655, 0.781]` — brighter than ImageNet in every channel, widest in blue (0.781 vs 0.406), from the violet Pappenheim stain against a bright background. So after ImageNet normalisation the data centres near `[1.12, 0.89, 1.67]`, **not** zero. This is expected, not a bug: the pretrained backbones require inputs standardised the way they were trained. Do not "fix" the non-zero mean. Switching to dataset-specific statistics is a defensible alternative and worth reporting as such, but it trades away part of the transfer-learning benefit.
2. **Hierarchical label mapping** — emit `[y1, y2]` per image.
3. **Stratified split** — *deterministic* 70/15/15 train/val/test, guaranteeing rare populations (reactive lymphocytes, n=33) appear proportionally in every set. Seed it and persist the split.
4. **Lineage-aware augmentation** — online random flips, rotation up to 90°, mild brightness/contrast jitter (0.1). **Training set only**, with an *intensified regime on minority classes*. Augmentation strength is class-conditional, not global.

> **Resolved 2026-07-20:** input resolution is **224×224 for every backbone, including the ViT.** This keeps the backbone comparison controlled (identical splits *and* identical input resolution), which the ablation design depends on. [Data_Analysis.tex](Data_Analysis.tex) still says 384×384 for Vision Transformers and must be corrected before write-up.

### Architecture

One shared pretrained CNN/ViT backbone → two heads (L1 lineage, L2 fine type), trained **jointly** so the coarse decision regularises the fine one and rare types borrow strength from lineage neighbours. A single class-balanced hierarchical loss backpropagates into the backbone. Inference emits a combined lineage + cell-type prediction. Backbones to compare: MobileNet, ResNet, ConvNeXt, ViT.

Candidate imbalance objectives named in the literature review: class-balanced loss via effective number of samples (Cui et al.), focal loss (Lin et al.), and margin-/triplet-based objectives for imbalanced medical classification.

### Evaluation

Report accuracy, macro precision, macro recall, macro F1, balanced accuracy, and the confusion matrix. Plain accuracy alone is explicitly rejected as misleading here — macro F1 and balanced accuracy are the headline metrics because the focus is minority-class behaviour.

Analyse the confusion matrix **hierarchically**: within-lineage errors are clinically mild, cross-lineage errors are severe. Grad-CAM saliency maps are a deliverable, not an afterthought — generated from backbone feature maps to confirm the model attends to nucleus, chromatin, and cytoplasm rather than background artefacts.

### Required experiments

Under **identical data splits and backbones**:

1. Flat single-head 18-class classifier (baseline).
2. Full hierarchical model (proposed).
3. Ablation — remove the imbalance term.
4. Ablation — remove the hierarchy.

Significance via paired t-test across validation folds. All four configurations must stay runnable from the same codebase, which argues for a config-driven trainer rather than divergent per-variant scripts.

## Writing conventions in the .tex sources

- British spelling in the slides ("haematology", "generalise"); the literature review mixes in American forms. Match the file you are editing.
- Citations use inline `\bibitem` keys of the form `author + year` (`matek2021`, `rahimzadeh2021`, `mll23`).
- The literature review targets ~4,500 words and restricts sources to 2021–2026, preferring Q1 journals. Do not add older or lower-tier citations without flagging it.
- The research gap and research question are set in `gapbox` tcolorbox environments — these are the load-bearing claims of the dissertation; changing their wording changes what the code has to prove.
