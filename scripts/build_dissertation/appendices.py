from pathlib import Path

import pandas as pd
from docx.shared import Pt, Inches
from build_common import *

REPO = Path(__file__).resolve().parents[2]

MINORITY = {
    "Smudge cells", "Neoplastic lymphocytes", "Reactive lymphocytes",
    "Promyelocytes", "Myelocytes", "Metamyelocytes", "Band neutrophils",
    "Basophil granulocytes",
}

TEST_SUPPORT = {
    "Typical lymphocytes": 830, "Hairy cells": 490,
    "Large granular lymphocytes": 278,
    "Atypical lymphocytes (plasma cells)": 248, "Smudge cells": 148,
    "Neoplastic lymphocytes": 27, "Reactive lymphocytes": 5,
    "Myeloblasts": 1291, "Promyelocytes": 112,
    "Atypical promyelocytes": 305, "Myelocytes": 112, "Metamyelocytes": 72,
    "Band neutrophils": 103, "Segmented neutrophils": 1076,
    "Eosinophil granulocytes": 367, "Basophil granulocytes": 92,
    "Monocytes": 377, "Normoblasts": 311,
}


def appendices(doc):
    # ------------------------------------------------ Appendix A
    h1(doc, "Appendix A.  Experimental Configuration Reference")
    para(doc,
         "Every run is fully specified by a frozen configuration object, and the "
         "configuration is written to the results file alongside the metrics so "
         "that any recorded run can be reconstructed exactly. Table A1 lists the "
         "fields and the values used in Phase 2. Only the four fields marked vary "
         "between arms; every other field is held constant, which is what makes "
         "the comparison between arms attributable to the factor under test.")
    table(doc, ["Field", "Phase 2 value", "Varies by arm?"], [
        ["arm", "flat_baseline / hierarchical / no_imbalance / stain_norm", "Yes"],
        ["backbone", "vit (ViT-Small/16)", "No"],
        ["mode", "flat or hier", "Yes"],
        ["use_hierarchy", "True or False", "Yes"],
        ["use_imbalance", "True or False", "Yes"],
        ["stain_norm", "True or False", "Yes"],
        ["seed", "0, 1, 2", "No (all arms use all three)"],
        ["epochs", "30", "No"],
        ["batch_size", "64", "No"],
        ["lr", "0.0003", "No"],
        ["weight_decay", "0.05", "No"],
        ["warmup_epochs", "1", "No"],
        ["label_smoothing", "0.05", "No"],
        ["aug_policy", "randaugment", "No"],
        ["mix_kind", "cutmix", "No"],
        ["mix_prob", "0.5", "No"],
        ["mix_alpha", "1.0", "No"],
        ["use_ema", "True", "No"],
        ["ema_decay", "0.999", "No"],
        ["tta", "True (4 flip views)", "No"],
        ["use_early_stopping", "False", "No"],
    ], cap="Table A1  Configuration fields and Phase 2 values",
        widths=[1.5, 2.8, 1.4], font_size=9)

    para(doc,
         "Table A2 records the wall-clock cost of the complete matrix. Times vary "
         "between runs of the same arm because the workstation was in "
         "intermittent use for other tasks; they are reported for completeness "
         "rather than as a controlled measurement of computational cost.")
    table(doc, ["Arm", "Runs", "Mean minutes per run"], [
        ["flat_baseline", "3", "55.7"],
        ["hierarchical", "3", "75.2"],
        ["no_imbalance", "3", "239.4"],
        ["stain_norm", "3", "114.3"],
        ["Total", "12", "1,453.8 minutes (24.2 hours)"],
    ], cap="Table A2  Computational cost of the Phase 2 matrix",
        widths=[1.6, 1.0, 2.6], font_size=10)
    pagebreak(doc)

    # ------------------------------------------------ Appendix B
    h1(doc, "Appendix B.  Per-Class Precision and Recall")
    para(doc,
         "Table 4.5 in the main text reports per-class F1. This appendix "
         "decomposes that figure into its precision and recall components for the "
         "proposed model and the flat baseline, since the two behave differently "
         "and the distinction matters clinically: recall determines how many cells "
         "of a given type are found, while precision determines how many of the "
         "cells so labelled genuinely belong to it. For screening applications "
         "recall is generally the more important of the two.")
    para(doc,
         "All figures are averaged over the three seeds and computed on the test "
         "partition with four-view test-time augmentation. Rows follow "
         "class-index order, so within the myeloid block adjacent rows are "
         "adjacent maturation stages. Minority classes are marked with a dagger.")

    arms = [("hierarchical", "Hierarchical"), ("flat_baseline", "Flat baseline")]
    frames = {}
    for key, _ in arms:
        f = REPO / "results" / f"per_class_{key}.csv"
        if f.exists():
            frames[key] = pd.read_csv(f, index_col=0)

    if frames:
        order = list(TEST_SUPPORT.keys())
        rows = []
        for name in order:
            dagger = " †" if name in MINORITY else ""
            row = [f"{name}{dagger}", str(TEST_SUPPORT[name])]
            for key, _ in arms:
                df = frames.get(key)
                if df is not None and name in df.index:
                    row.append(f"{df.loc[name, 'precision']:.3f}")
                    row.append(f"{df.loc[name, 'recall']:.3f}")
                else:
                    row += ["—", "—"]
            rows.append(row)
        table(doc,
              ["Cell type", "Test n",
               "Hier. precision", "Hier. recall",
               "Flat precision", "Flat recall"],
              rows,
              cap="Table B1  Per-class precision and recall on the test "
                  "partition, averaged over three seeds. Minority classes "
                  "marked †",
              widths=[1.9, 0.55, 0.85, 0.8, 0.85, 0.8], font_size=8)
        para(doc,
             "The decomposition clarifies the nature of the residual failures. For "
             "the rare lymphoid classes, recall is substantially below precision, "
             "meaning the model is conservative: when it does assign the label it "
             "is usually right, but it frequently fails to assign it at all. That "
             "is the signature of an under-represented class whose decision "
             "boundary has been drawn too tightly, and it is the behaviour the "
             "class-balanced objective is intended to counteract.")
        para(doc,
             "For the intermediate myeloid stages the two quantities are closer "
             "together, indicating symmetric confusion with neighbouring stages "
             "rather than systematic under-prediction. This supports the argument "
             "in Section 4.3.5 that the two groups of difficult classes fail for "
             "different reasons, and that a single remedy is unlikely to address "
             "both.")
    else:
        para(doc,
             "Per-class results are generated by re-running test-set inference "
             "over the saved checkpoints and averaging within each arm across the "
             "three seeds, writing results/per_class_<arm>.csv.")

    para(doc,
         "Aggregate minority-class statistics quoted throughout Chapter 4 are "
         "computed over the 671 test images whose true class is one of the eight "
         "daggered classes above. Note that the minority macro F1 reported in "
         "Table 4.2 is computed on that restricted subset rather than as the mean "
         "of the full-data per-class F1 values in Table 4.5, so the two are "
         "related but not arithmetically identical.")
