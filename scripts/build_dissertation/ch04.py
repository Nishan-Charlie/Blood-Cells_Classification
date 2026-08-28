from docx.shared import Pt, Inches
from build_common import *


def chapter4(doc):
    h1(doc, "Chapter 4  Experimental Results")
    para(doc,
         "This chapter reports the experimental setup, restates the properties of "
              "the dataset relevant to interpreting the results, presents the "
              "quantitative findings of the four experimental arms, analyses them at the "
              "level of individual classes, and compares the proposed system against the "
              "flat baseline and against reported results from the literature. Findings "
              "that did not support the hypothesis are reported alongside those that did.")

    # ------------------------------------------------ 4.1
    h2(doc, "4.1  Experimental Setup")
    para(doc,
         "All experiments were conducted on a single workstation with an NVIDIA RTX "
              "4070 Laptop GPU providing 8 GB of video memory, running Windows 11. The "
              "implementation uses PyTorch with the timm library supplying "
              "ImageNet-pretrained backbones, scikit-learn for metrics and partitioning, "
              "and NumPy and pandas for data handling. The complete Phase 2 matrix of "
              "twelve runs consumed approximately 24.2 GPU-hours.")
    para(doc,
         "Two implementation decisions materially affected the results and are "
              "recorded here instead of in the methodology, because both were found by "
              "experiment, not chosen in advance. First, mixed-precision training uses "
              "bfloat16 in place of float16. An initial float16 run reached a validation "
              "macro F1 of 0.80 and then diverged to a non-finite loss at the third "
              "epoch, wasting the remainder of the run since such weights never recover. "
              "The bfloat16 format retains the exponent range of float32 and cannot "
              "overflow in the same way. Gradient scaling is correspondingly disabled, "
              "and the loss is computed in float32 regardless of autocast context, "
              "because the class-balanced weights span a wide range and the focal term "
              "exponentiates a log-probability.")
    para(doc,
         "Second, profiling established that the pipeline was limited by data "
              "loading, not by computation. Decoding source images sustained 178 images "
              "per second at four dataloader workers and fell to 119 at eight through "
              "worker contention, against a full training step of 82 images per second. "
              "Caching decoded images at native resolution raised throughput to "
              "approximately 850 images per second, without which the twelve-run matrix "
              "would not have been feasible. The observation is recorded because it "
              "inverts the usual assumption that backbone choice dominates training cost; "
              "here it barely moved wall-clock time at all.")
    para(doc,
         "The backbone screen of Phase 1 evaluated MobileNetV3-Large, ResNet-50, "
              "ConvNeXt-Tiny and ViT-Small/16 under the hierarchical configuration for a "
              "short fixed budget at a single seed. ViT-Small/16 achieved the highest "
              "validation macro F1 at 0.849 and was carried into Phase 2. As stated in "
              "Chapter 3, this is a selection result and not a backbone comparison: the "
              "budget was too short to rank architectures reliably, and the margin over "
              "ConvNeXt-Tiny was narrow.")
    table(doc, ["Parameter", "Value"], [
        ["Backbone", "ViT-Small/16, ImageNet-pretrained, fully fine-tuned"],
        ["Input resolution", "224 x 224 for every backbone"],
        ["Optimiser", "AdamW"],
        ["Learning rate", "3 x 10^-4"],
        ["Weight decay", "0.05"],
        ["Batch size", "64"],
        ["Epochs", "30, fixed; no early stopping"],
        ["Warmup", "1 epoch, linear"],
        ["Schedule", "Cosine annealing"],
        ["Label smoothing", "0.05"],
        ["Focal gamma", "2.0"],
        ["Class-balanced beta", "0.9999"],
        ["Loss weights", "lambda_lin 0.3, lambda_fine 1.0, lambda_cons 0.1"],
        ["Augmentation", "RandAugment, class-conditional"],
        ["Batch mixing", "CutMix, p = 0.5, alpha = 1.0"],
        ["Weight EMA decay", "0.999"],
        ["Test-time augmentation", "4 flip views"],
        ["Mixed precision", "bfloat16"],
        ["Model selection", "Validation macro F1"],
        ["Seeds", "0, 1, 2"],
    ], cap="Table 4.1  Training hyperparameters, held identical across all arms",
        widths=[2.0, 3.6], font_size=10)

    # ------------------------------------------------ 4.2
    h2(doc, "4.2  Dataset Description")
    para(doc,
         "The MLL23 corpus [1] provides 41,621 Pappenheim-stained single "
              "nucleated-cell images at 288 by 288 pixels, annotated by expert "
              "cytomorphologists into 18 classes across three haematopoietic lineages. "
              "The full class listing with partition sizes is given in Table 3.1. Four "
              "properties govern the interpretation of every result that follows.")
    para(doc,
         "The distribution is severely long-tailed, with an imbalance ratio of 261 "
              "between myeloblasts at 8,606 images and reactive lymphocytes at 33. After "
              "the 70/15/15 partition the rarest class contributes 23 training images, 5 "
              "validation images and 5 test images. Per-class metrics for that class are "
              "consequently unstable across seeds, and this is reported openly instead of "
              "being buried in the macro average.")
    para(doc,
         "The lineages are unequal in size, myeloid accounting for approximately 63% "
              "of the corpus, lymphoid for 32% and erythroid for 5%. The lineage task is "
              "both imbalanced and, as the results show, easy, which turns out to be "
              "central to interpreting the hierarchical result.")
    para(doc,
         "The rare classes are not distributed evenly across lineages. Of the eight "
              "minority classes, five are myeloid maturation stages and two are rare "
              "lymphoid types. This asymmetry matters because the two groups stand in "
              "different relations to their lineage neighbours, and Section 4.3.5 shows "
              "that the proposed architecture treats them very differently.")
    para(doc,
         "The test partition contains 6,244 images, of which 671 belong to the eight "
              "minority classes. All minority-class aggregates reported below are "
              "computed over those 671 samples, which is a small enough basis that "
              "differences of less than approximately 0.01 in minority macro F1 should "
              "not be over-interpreted.")

    # ------------------------------------------------ 4.3
    h2(doc, "4.3  Results")
    para(doc,
         "Table 4.2 reports test-set performance for the four distinct "
              "configurations, each averaged over three seeds with the standard deviation "
              "across seeds. The proposed hierarchical model achieves the highest macro "
              "F1, minority macro F1, minority recall and balanced accuracy of the four.")
    table(doc,
          ["Arm", "Accuracy", "Macro F1", "Balanced acc.",
           "Minority F1", "Minority recall"],
          [
              ["Hierarchical (proposed)", "0.9567", "0.8792 ± 0.0028",
               "0.8936", "0.8400 ± 0.0073", "0.7987"],
              ["Flat baseline", "0.9565", "0.8742 ± 0.0060",
               "0.8832", "0.8233 ± 0.0140", "0.7761"],
              ["No imbalance term", "0.9602", "0.8733 ± 0.0093",
               "0.8620", "0.7799 ± 0.0271", "0.7226"],
              ["Stain-normalised", "0.9546", "0.8714 ± 0.0031",
               "0.8832", "0.8206 ± 0.0189", "0.7753"],
          ],
          cap="Table 4.2  Test-set performance by experimental arm, mean ± standard "
               "deviation across three seeds",
          widths=[1.55, 0.85, 1.15, 0.95, 1.05, 1.05], font_size=9)
    figure(doc, "fig_arm_comparison.png",
           "Figure 4.1  Overall and minority-class F1 by arm. Each dot is the mean of "
                "three seeds and the bars span one standard deviation. The four arms sit "
                "almost on top of one another on overall macro F1 but spread widely on the "
                "rare classes.", width_in=6.0)
    para(doc,
         "Figure 4.1 shows why the two columns must be read together. On overall "
              "macro F1 the four arms are separated by 0.008 in total, a spread so narrow "
              "that the error bars overlap everywhere. On the minority classes the same "
              "four arms spread across 0.060, and the ordering changes. Whatever these "
              "experiments are measuring, it is not visible in the aggregate number.")

    h3(doc, "4.3.1  Accuracy is the wrong metric, demonstrated directly")
    para(doc,
         "The single most instructive row of Table 4.2 is the imbalance ablation. "
              "Removing the class-balanced and focal terms produced the highest overall "
              "accuracy of any arm, at 0.9602, while simultaneously producing the lowest "
              "minority macro F1 at 0.7799 and the lowest minority recall at 0.7226. The "
              "arm that looks best on the conventional headline metric is the worst on "
              "the classes the study exists to examine.")
    para(doc,
         "This is not a coincidence but the predicted arithmetic consequence of the "
              "imbalance. An objective that is not class-balanced allocates capacity in "
              "proportion to class frequency, which maximises the probability of a "
              "correct prediction on a randomly drawn test image. Since approximately 89% "
              "of the test partition belongs to the ten majority classes, improving those "
              "at the expense of the tail is the accuracy-optimal strategy, and an "
              "optimiser given a symmetric objective will find it.")
    para(doc,
         "The finding provides direct empirical support for the methodological "
              "argument of Chapter 2, that reporting accuracy on clinically "
              "representative data conceals the behaviour of interest, and it does so "
              "from within one controlled experiment instead of an appeal to principle. "
              "Two models differing only in their loss function are separated by 0.0035 "
              "in accuracy, in favour of the weaker one, and by 0.060 in minority macro "
              "F1, in favour of the stronger. A reader given only the accuracy column "
              "would draw precisely the wrong conclusion about which system to deploy. "
              "Since accuracy remains the dominant reported metric in the literature "
              "reviewed in Chapter 2, this is not a hypothetical concern.")

    h3(doc, "4.3.2  The imbalance objective is the load-bearing component")
    para(doc,
         "Comparing the proposed model against the imbalance ablation isolates the "
              "contribution of the class-balanced and focal terms, since the two arms are "
              "identical in every other respect including the hierarchy. The differences "
              "are large and, unusually for a three-seed design, statistically supported.")
    table(doc,
          ["Comparison", "Metric", "Difference", "t", "p", "Cohen's d_z"],
          [
              ["Hier. vs no-imbalance", "Minority recall", "+0.0760", "6.36", "0.024", "3.67"],
              ["Hier. vs no-imbalance", "Balanced accuracy", "+0.0316", "5.84", "0.028", "3.37"],
              ["Hier. vs no-imbalance", "Minority macro F1", "+0.0601", "4.21", "0.052", "2.43"],
              ["Hier. vs no-imbalance", "Macro F1", "+0.0059", "0.97", "0.436", "0.56"],
              ["Hier. vs flat", "Cross-lineage error", "-0.0004", "-7.00", "0.020", "-4.04"],
              ["Hier. vs flat", "Balanced accuracy", "+0.0103", "1.87", "0.202", "1.08"],
              ["Hier. vs flat", "Minority recall", "+0.0226", "1.80", "0.214", "1.04"],
              ["Hier. vs flat", "Minority macro F1", "+0.0167", "1.45", "0.285", "0.84"],
              ["Hier. vs flat", "Macro F1", "+0.0050", "1.00", "0.424", "0.57"],
              ["Stain vs hier.", "Macro F1", "-0.0078", "-2.29", "0.149", "-1.32"],
              ["Stain vs hier.", "Minority macro F1", "-0.0194", "-1.71", "0.229", "-0.99"],
              ["Stain vs hier.", "Cross-lineage error", "+0.0012", "3.62", "0.069", "2.09"],
          ],
          cap="Table 4.3  Paired comparisons across three seeds. Differences are stated "
               "for the first-named arm relative to the second",
          widths=[1.45, 1.35, 0.9, 0.55, 0.6, 0.85], font_size=9)
    para(doc,
         "Removing the imbalance treatment costs 0.076 in minority recall (p = "
              "0.024) and 0.032 in balanced accuracy (p = 0.028), with effect sizes above "
              "three standard deviations of the paired difference. Minority macro F1 "
              "falls by 0.060 at p = 0.052, marginally outside conventional significance "
              "but with a very large effect size of 2.43. Macro F1 over all eighteen "
              "classes, by contrast, moves by only 0.006 and is not significant, because "
              "the ten non-minority classes dominate that average and are largely "
              "unaffected.")
    para(doc,
         "The pattern is coherent. The class-balanced and focal terms operate almost "
              "exclusively on the tail, leaving the head of the distribution unchanged, "
              "which is exactly what they are designed to do. Their contribution to the "
              "rare classes is an order of magnitude larger than the contribution of the "
              "hierarchy discussed below. On the evidence of this experiment, the "
              "imbalance-robust objective is the load-bearing component of the proposed "
              "design, and a practitioner forced to adopt only one of the two mechanisms "
              "should adopt this one.")

    h3(doc, "4.3.3  The hierarchy operates through the predicted mechanism")
    para(doc,
         "The proposed model outperforms the flat baseline on every metric in Table "
              "4.2, and by more on the rare classes, at +0.017 minority macro F1, than on "
              "the corpus as a whole, at +0.005 macro F1. That ordering is the pattern "
              "the design predicted, and it should be stressed that the prediction was "
              "directional and specific, not a general hope of improvement. None of these "
              "aggregate differences, however, reaches statistical significance at three "
              "seeds. Minority recall improves by 0.023 with p = 0.214, and balanced "
              "accuracy by 0.010 with p = 0.202. The effect sizes, at d_z above 1.0, are "
              "large; the sample of runs is too small to resolve them.")
    para(doc,
         "One comparison does reach significance, and it is the one the theoretical "
              "argument identified in advance as the direct test. The hierarchical model "
              "reduces cross-lineage error, the clinically severe category, from 0.0119 "
              "to 0.0115, a difference of only 0.0004 in absolute terms but one that held "
              "in the same direction across all three seeds, giving t = -7.00, p = 0.020 "
              "and d_z = -4.04. Correspondingly, the share of errors that remained within "
              "the correct lineage rose from 0.726 to 0.733.")
    table(doc,
          ["Arm", "Correct", "Within-lineage error", "Cross-lineage error",
           "Within-error share", "Lineage accuracy"],
          [
              ["Hierarchical", "0.9567", "0.0318", "0.0115", "0.7333", "0.9885"],
              ["Flat baseline", "0.9565", "0.0316", "0.0119", "0.7261", "0.9881"],
              ["No imbalance term", "0.9602", "0.0294", "0.0104", "0.7396", "0.9896"],
              ["Stain-normalised", "0.9546", "0.0327", "0.0127", "0.7200", "0.9873"],
          ],
          cap="Table 4.4  Hierarchical decomposition of prediction outcomes on the test "
               "partition",
          widths=[1.4, 0.8, 1.2, 1.1, 1.05, 1.0], font_size=9)
    figure(doc, "fig_cross_lineage_paired.png",
           "Figure 4.2  Cross-lineage error for each seed, flat baseline against the "
                "proposed model. All three seeds move the same way, which is what produces "
                "a low p-value from a very small difference.", width_in=4.4)
    para(doc,
         "The interpretation requires care in both directions. The consistent sign "
              "across seeds indicates a real effect and supports the mechanism the "
              "architecture was built to exploit: coupling the fine head to a lineage "
              "posterior does shift errors toward the clinically milder kind. But the "
              "magnitude is very small. A reduction of 0.0004 in cross-lineage error "
              "corresponds to approximately two or three images in a test partition of "
              "6,244. A statistically significant result of this size should be described "
              "as a reliably detected small effect, not as a clinically important "
              "improvement, and the low p-value reflects the consistency of the "
              "difference across seeds, not its size. Reporting it otherwise would be a "
              "misuse of the significance test.")
    para(doc,
         "The comparison also illustrates why the hierarchical decomposition of "
              "Equation (3.11) was worth computing. Total accuracy for the two arms is "
              "identical to three decimal places, at 0.9567 against 0.9565, and total "
              "error is likewise unchanged. Judged on accuracy alone the hierarchy would "
              "appear to do nothing whatsoever. The decomposition shows that it "
              "redistributes errors between categories of differing clinical severity "
              "while leaving their total count essentially unchanged, which is exactly "
              "the behaviour a scalar metric cannot express. As noted in Chapter 2, no "
              "reviewed study reports this decomposition, which means that comparable "
              "effects in existing systems would have been invisible to their authors.")
    para(doc,
         "An apparent anomaly in Table 4.4 deserves comment. The imbalance ablation "
              "records both the lowest cross-lineage error, at 0.0104, and the highest "
              "lineage accuracy, at 0.9896, despite being the weakest arm on every "
              "minority metric. This is not a contradiction. Cross-lineage error is "
              "computed over all test samples, so it is dominated by the majority "
              "classes, on which that arm concentrates its capacity. A model that serves "
              "the head of the distribution well and abandons the tail will produce few "
              "lineage-level errors in aggregate because the abandoned classes are rare. "
              "The measure is most informative when comparing arms of similar minority "
              "performance, as with the hierarchical and flat pair, but should not be "
              "read across arms with very different tail behaviour.")

    h3(doc, "4.3.4  Why the hierarchical effect is small")
    para(doc,
         "Diagnostic logging supplies a direct explanation for the modest size of "
              "the hierarchical contribution. Lineage accuracy reaches 98.9% (Table 4.4), "
              "and by the end of training both auxiliary loss terms have collapsed to "
              "approximately 0.0007. The lineage task is essentially solved early in "
              "training, after which the auxiliary head contributes almost no gradient to "
              "the shared trunk.")
    para(doc,
         "This is a finding about the design, not a tuning problem, and the "
              "distinction matters for what follows from it. Adjusting the loss weight "
              "would not help: multiplying a gradient of approximately zero by a larger "
              "coefficient yields a gradient of approximately zero. The auxiliary task "
              "regularises an axis the model has already mastered, so it cannot continue "
              "to shape the representation during the period when the hard "
              "discriminations are being learned.")
    para(doc,
         "The hard problem is discrimination within a lineage, between myeloid "
              "maturation stages and among the rare lymphoid types, and lineage "
              "supervision provides no signal whatever about those distinctions by "
              "construction, since every confusable pair shares a lineage label. A "
              "myelocyte and a metamyelocyte are both myeloid; the auxiliary head is "
              "indifferent between them, and correctly so. The architecture supplies "
              "extra supervision on the one axis that needed it least.")
    para(doc,
         "The result is consistent with the exploratory analysis reported in Chapter "
              "3. Frozen-feature analysis found k-nearest-neighbour agreement of 0.78 for "
              "lineage against 0.42 for fine cell type: the coarse level was already "
              "substantially recoverable from a generic ImageNet representation before "
              "any fine-tuning, whereas the fine level was not. An auxiliary task defined "
              "on the coarse level had limited headroom from the outset, and in hindsight "
              "the exploratory measurement predicted the size of the effect that was "
              "subsequently observed. The implication for future work, developed in "
              "Chapter 5, is that the auxiliary signal should target maturation stage "
              "within the myeloid branch, where the residual confusion demonstrably "
              "resides.")

    h3(doc, "4.3.5  Per-class analysis: where the hierarchy actually helps")
    para(doc,
         "The aggregate metrics of Table 4.2 average over eighteen classes of widely "
              "differing frequency and difficulty, and in doing so they conceal the "
              "structure of the effect. Table 4.5 reports per-class test F1 for each arm, "
              "averaged over the three seeds, together with the difference between the "
              "proposed model and each of its two comparators.")

    per_class_rows = [
        ["Typical lymphocytes", "830", "0.937", "0.934", "0.942", "0.934", "+0.003", "-0.005"],
        ["Hairy cells", "490", "0.982", "0.984", "0.986", "0.984", "-0.002", "-0.005"],
        ["Large granular lymphocytes", "278", "0.919", "0.918", "0.927", "0.920", "+0.000", "-0.008"],
        ["Atypical lymphocytes (plasma)", "248", "0.995", "0.995", "0.996", "0.994", "+0.000", "-0.001"],
        ["Smudge cells †", "148", "0.960", "0.964", "0.962", "0.959", "-0.004", "-0.002"],
        ["Neoplastic lymphocytes †", "27", "0.739", "0.689", "0.604", "0.680", "+0.050", "+0.136"],
        ["Reactive lymphocytes †", "5", "0.430", "0.360", "0.370", "0.407", "+0.070", "+0.060"],
        ["Myeloblasts", "1291", "0.986", "0.985", "0.989", "0.983", "+0.001", "-0.003"],
        ["Promyelocytes †", "112", "0.840", "0.853", "0.856", "0.849", "-0.013", "-0.016"],
        ["Atypical promyelocytes", "305", "0.988", "0.988", "0.990", "0.985", "+0.000", "-0.002"],
        ["Myelocytes †", "112", "0.740", "0.755", "0.760", "0.711", "-0.015", "-0.021"],
        ["Metamyelocytes †", "72", "0.677", "0.676", "0.705", "0.661", "+0.001", "-0.028"],
        ["Band neutrophils †", "103", "0.754", "0.760", "0.747", "0.744", "-0.006", "+0.007"],
        ["Segmented neutrophils", "1076", "0.984", "0.984", "0.984", "0.984", "-0.000", "+0.000"],
        ["Eosinophil granulocytes", "367", "0.991", "0.990", "0.993", "0.990", "+0.000", "-0.002"],
        ["Basophil granulocytes †", "92", "0.982", "0.982", "0.983", "0.976", "+0.000", "-0.002"],
        ["Monocytes", "377", "0.940", "0.939", "0.938", "0.936", "+0.000", "+0.001"],
        ["Normoblasts", "311", "0.983", "0.981", "0.986", "0.987", "+0.002", "-0.004"],
    ]
    table(doc,
          ["Cell type", "Test n", "Hier.", "Flat", "No imb.", "Stain",
           "Hier. − Flat", "Hier. − No imb."],
          per_class_rows,
          cap="Table 4.5  Per-class test F1 by arm, averaged over three seeds. Minority "
               "classes marked †. Rows follow class-index order, so adjacent myeloid rows "
               "are adjacent maturation stages",
          widths=[1.55, 0.5, 0.5, 0.5, 0.55, 0.5, 0.75, 0.85], font_size=8)
    figure(doc, "fig_per_class_dumbbell.png",
           "Figure 4.3  Per-class F1 for the proposed model against the flat baseline. "
                "Where the two dots sit on top of each other the arms agree; only two "
                "classes show a visible gap.", width_in=6.1)

    para(doc,
         "The first observation is that the hierarchical advantage is not diffuse. "
              "Of eighteen classes, fifteen show a difference against the flat baseline "
              "of 0.006 or less in absolute terms, which is within seed-to-seed noise. "
              "The aggregate improvement is carried almost entirely by two classes: "
              "reactive lymphocytes, which gain 0.070, and neoplastic lymphocytes, which "
              "gain 0.050. These are the two rarest lymphoid types in the corpus, with 33 "
              "and 180 images respectively.")
    para(doc,
         "This is the pattern the theoretical argument predicts, and it is a "
              "considerably more specific confirmation than the aggregate metrics "
              "provide. The mechanism claimed for hierarchical coupling is that rare "
              "classes borrow representational strength from abundant neighbours sharing "
              "their lineage. Reactive and neoplastic lymphocytes sit in a lineage that "
              "also contains typical lymphocytes at 5,532 images and hairy cells at "
              "3,265. There is a great deal of same-lineage evidence for them to borrow "
              "from, and the coupling lets them do so. The effect appears exactly where "
              "the mechanism says it should, and is absent where the mechanism offers no "
              "reason to expect it.")
    para(doc,
         "The second observation is the mirror image of the first. The hierarchy "
              "confers no measurable benefit on the myeloid maturation stages: across the "
              "six classes of the neutrophil chain the mean difference against the flat "
              "baseline is -0.005, against a typical seed-to-seed standard deviation of "
              "0.011, so the chain as a whole is unchanged within noise. The largest "
              "individual movements, myelocytes at -0.015 and promyelocytes at -0.013, "
              "are of the same order as their own seed variation and no claim of harm is "
              "made on that evidence. What can be said is that the gain observed on the "
              "rare lymphoid types is entirely absent here, and the reason is structural: "
              "these are classes whose confusable neighbours share their lineage label, "
              "so lineage supervision cannot discriminate among them even in principle.")
    para(doc,
         "The net effect is positive, which is why the aggregate favours the "
              "proposed model, but describing it as a uniform improvement would "
              "misrepresent the evidence. It is a redistribution, and the redistribution "
              "follows the structure of the taxonomy in a way that is predictable in "
              "advance, not accidental.")
    para(doc,
         "The third observation concerns the imbalance ablation, whose per-class "
              "profile is a still starker trade. Neoplastic lymphocytes lose 0.136 and "
              "reactive lymphocytes 0.060 when the class-balanced and focal terms are "
              "removed, which is the expected direction and a very large effect. But "
              "several mid-frequency classes improve without those terms: metamyelocytes "
              "gain 0.028, myelocytes 0.021 and promyelocytes 0.016. Class-balanced "
              "weighting redistributes capacity toward the extreme tail, and it takes "
              "that capacity from classes that are themselves uncommon but not rare "
              "enough to be protected. Three of the classes that lose ground are formally "
              "minority classes under the threshold adopted here, which shows the "
              "threshold is a simplification of a continuum, not a natural boundary.")
    para(doc,
         "The fourth observation concerns absolute difficulty, and it is independent "
              "of arm. The five worst classes under every configuration are reactive "
              "lymphocytes at 0.43, metamyelocytes at 0.68, neoplastic lymphocytes at "
              "0.74, myelocytes at 0.74 and band neutrophils at 0.75. Two of these are "
              "rare lymphoid types; three are consecutive stages of neutrophil "
              "maturation. Rarity alone does not explain the pattern, since basophil "
              "granulocytes at 92 test images reach 0.982 and smudge cells at 148 reach "
              "0.960, both comparable in frequency to the failing myeloid stages. What "
              "distinguishes the failing classes is that they have close morphological "
              "neighbours, whereas basophils are visually distinctive. Difficulty here is "
              "driven by morphological adjacency at least as much as by sample count, "
              "which is consistent with the argument advanced in Chapter 2 and is the "
              "strongest available justification for targeting the maturation axis in "
              "future work.")
    para(doc,
         "Reactive lymphocytes warrant separate comment. At an F1 of approximately "
              "0.43 across arms, the class is not usefully classified by any "
              "configuration tested. With 5 test images, a single prediction changes its "
              "recall by 0.2, so the seed-to-seed variance is necessarily large and no "
              "conclusion about the relative merits of the arms on that class alone would "
              "be safe. It is reported because omitting it would flatter the results, and "
              "because it establishes a floor: for a class with 23 training examples, "
              "neither the hierarchy nor the class-balanced objective as implemented here "
              "is sufficient.")

    h3(doc, "4.3.6  Errors follow the maturation continuum")
    para(doc,
         "The per-class results raise a question they cannot answer. Three of the "
              "five weakest classes are consecutive stages of neutrophil maturation, so "
              "it is worth asking what those classes are actually being confused with. "
              "Pooling the predictions of all three hierarchical seeds gives 18,732 test "
              "predictions and 811 errors, of which 305 fall between two classes that "
              "both belong to the maturation chain. Each of those errors can be scored by "
              "how many developmental stages separate the true class from the predicted "
              "one.")
    figure(doc, "fig_stage_distance.png",
           "Figure 4.4  How far along the maturation chain each error lands. Almost "
                "every mistake is a single stage in one direction or the other, and none is "
                "more than two.", width_in=5.6)
    para(doc,
         "Figure 4.4 gives an unusually clean answer. Of the 305 errors inside the "
              "chain, 294, or 96.4%, are off by exactly one stage. Not one is off by more "
              "than two. The individual classes behave the same way: promyelocytes are "
              "called myelocytes in 88% of their errors, metamyelocytes go to band "
              "neutrophils or myelocytes in 100% of theirs, and segmented neutrophils are "
              "called band neutrophils in 81%. These are neighbours on the developmental "
              "path, and the model is landing next door, not guessing at random.")
    para(doc,
         "Counted against every error the model makes, adjacent-stage confusions "
              "come to 294 of 811, or 36%. Over a third of all mistakes are the mildest "
              "kind available, and the training objective treats each of them as exactly "
              "as costly as calling a lymphocyte a myeloblast. Nothing in the loss, and "
              "nothing in macro F1, knows that these six classes lie in an order. Chapter "
              "5 returns to this, because it is the clearest opening the results leave "
              "for further work.")
    para(doc,
         "Myeloblasts are the exception that confirms the reading. Their errors do "
              "not travel along the chain at all: 47% go to typical lymphocytes and 22% "
              "to monocytes, both outside the myeloid lineage. A blast is not visually "
              "adjacent to a promyelocyte in the way a metamyelocyte is adjacent to a "
              "band neutrophil, and distinguishing blasts from lymphocytes is a different "
              "and clinically graver problem that ordinal reasoning would not touch.")

    h3(doc, "4.3.7  Stain normalisation did not help")
    para(doc,
         "The stain-normalised arm underperformed the proposed model on every metric "
              "in Table 4.2, with macro F1 lower by 0.008 and minority macro F1 lower by "
              "0.019. Neither difference is significant at three seeds, at p = 0.149 and "
              "p = 0.229, but the direction was consistent across seeds and cross-lineage "
              "error was higher at p = 0.069. The per-class view in Table 4.5 shows the "
              "loss concentrated on myelocytes, which fall from 0.740 to 0.711, and on "
              "neoplastic lymphocytes. The hypothesis that removing colour variation "
              "would aid classification is not supported here, and the balance of "
              "evidence mildly favours the opposite.")
    para(doc,
         "Two explanations are plausible and the experiment does not distinguish "
              "them. Colour may carry genuine diagnostic signal in Pappenheim-stained "
              "material, since stain uptake reflects nucleic acid and protein content, so "
              "normalising it away would throw away information, not noise. This would "
              "explain why the loss falls on classes distinguished partly by cytoplasmic "
              "staining character. Alternatively, MLL23 originates from a single "
              "laboratory with consistent preparation protocols, so there may be little "
              "inter-site variation for normalisation to correct, leaving only its cost "
              "in discarded signal.")
    para(doc,
         "The second explanation implies the arm might behave quite differently on "
              "multi-site data, where the variation stain normalisation exists to remove "
              "is actually present. The finding should not be generalised beyond the "
              "single-source setting tested, and it constitutes an argument for the "
              "cross-site evaluation proposed in Chapter 5, not a verdict on the "
              "technique.")

    h3(doc, "4.3.8  Saliency")
    figure(doc, "gradcam_fine.png",
           "Figure 4.5  Grad-CAM saliency for the fine classification head, with true "
                "and predicted labels", width_in=5.6)
    para(doc,
         "Figure 4.5 shows Grad-CAM maps for the fine head across a range of "
              "classes. Attention concentrates on the cell body, covering the nucleus, "
              "chromatin texture and cytoplasmic boundary, and not on the red cells "
              "around it or on empty background. This is the qualitative check the method "
              "was included to perform, and it passes: the network is not exploiting "
              "background staining artefacts or field position as a shortcut, which is "
              "the failure mode that would most obviously prevent transfer to another "
              "laboratory.")
    para(doc,
         "The reactive lymphocyte panel, which the model misclassified as a typical "
              "lymphocyte, is the most instructive. Saliency remains correctly localised "
              "on the cell, so the failure is not one of attention but of discrimination: "
              "the model examined the right evidence and drew the wrong conclusion from "
              "it. For a class with 23 training examples this is the expected failure "
              "mode, and it carries a practical implication. Further gains on that class "
              "will come from the training objective or from additional data, not from "
              "architectural changes that improve localisation, since localisation is "
              "already correct.")
    para(doc,
         "A methodological caveat attaches to producing these maps on a Vision "
              "Transformer. Because the classifier reads only the class token, patch "
              "tokens at the final block have exactly zero gradient, and hooking that "
              "layer yields a uniformly blank map that min-max normalisation renders as a "
              "plausible-looking figure. The measured gradient magnitude was 0.0 at the "
              "final block against 6.6 x 10^-2 at the input to its attention sublayer, "
              "which is the layer used here. This is worth recording because the failure "
              "is silent: the resulting figure looks like a result, and would pass visual "
              "inspection.")

    # ------------------------------------------------ 4.4
    h2(doc, "4.4  Comparison with Baseline Methods")
    para(doc,
         "The primary baseline is the flat single-head classifier trained under "
              "identical conditions, reported throughout Section 4.3. That comparison is "
              "internally controlled: identical partition, identical backbone, identical "
              "schedule, identical augmentation, differing only in the presence of the "
              "lineage head and its loss terms. It is the comparison on which the "
              "conclusions of this study rest, and it is the only one in which the "
              "confounds are fully accounted for.")
    para(doc,
         "It is tempting to place these numbers beside the accuracies quoted in "
              "Chapter 2, several of which reach 99% or 100%. No such table appears here, "
              "and the reason is that the comparison would be meaningless. Those figures "
              "come from problems with two to eight classes; this one has eighteen. They "
              "come from datasets that were balanced before training began; this one has "
              "261 times more myeloblasts than reactive lymphocytes. They report "
              "accuracy; this study reports macro F1 because Section 4.3.1 showed "
              "accuracy to be actively misleading on data of this shape. Three separate "
              "things differ at once, so any ranking drawn from such a table would say "
              "more about which benchmark each author chose than about which method "
              "works.")
    para(doc,
         "The size of the distortion can be measured inside this study's own "
              "results. The proposed model scores 0.957 on accuracy and 0.879 on macro F1 "
              "from the very same predictions. Nearly eight points separate the two "
              "numbers, and the only difference is whether rare classes are allowed to "
              "count as much as common ones. Quoting the higher figure alongside a "
              "published 99% would suggest this work falls short, when in truth the two "
              "numbers are measuring different things.")
    para(doc,
         "The per-class results make the same point more concretely. This study's "
              "models pass 0.98 F1 on eight of the eighteen classes, which matches "
              "anything in the published literature, and fall below 0.76 on five others. "
              "A system evaluated only on the easy eight would look excellent and would "
              "never meet the hard five at all. That is the shape of the problem, and it "
              "is why a like-for-like table cannot be built from the sources available.")
    para(doc,
         "One published system can be compared in a qualitative sense. Acevedo and "
              "colleagues [16] built the closest antecedent to this architecture, a "
              "two-stage design that decides the coarse group first and the fine type "
              "second. Their two stages learn from separate objectives, so a mistake at "
              "the fine level never teaches the coarse stage anything. The model here "
              "joins the levels through one differentiable loss, and Section 4.3.3 shows "
              "that this does move errors from the severe category to the mild one, if "
              "only by a little. Which design would win on the same data is unknown, "
              "because no dataset has been used by both.")

    h3(doc, "4.4.1  A methodological correction affecting these results")
    para(doc,
         "An earlier complete experimental matrix was run and subsequently "
              "discarded. It is reported here because the reason for discarding it bears "
              "directly on the reliability of the results above, and because the failure "
              "mode is one the reviewed literature does not discuss.")
    para(doc,
         "That first matrix enabled early stopping against a cosine schedule defined "
              "over a 50-epoch budget. Runs halted between epochs 12 and 37, which "
              "truncated the anneal and left models at up to 86% of peak learning rate, "
              "mid-training and not converged. The number of epochs a run survived then "
              "correlated with its final score at r = +0.87, so the ranking of arms "
              "substantially measured the stopping rule, not the method under test. A "
              "striking symptom was that the six-epoch screening runs, whose schedules "
              "completed, scored higher on test than any fifty-epoch run: training for "
              "longer made results worse, purely through this interaction.")
    para(doc,
         "Compounding this, epoch-to-epoch validation macro F1 varied with a "
              "standard deviation of 0.021, larger than the between-arm differences of "
              "0.008 to 0.028 the study set out to detect. Selecting a checkpoint on such "
              "a signal approximates a lottery, and patience-based stopping on it "
              "approximates a random stopping rule.")
    para(doc,
         "The consequence was not merely imprecision but inversion. That first "
              "matrix ranked the flat baseline above the hierarchical model on minority "
              "macro F1, at 0.819 against 0.791, the reverse of the ordering reported in "
              "Table 4.2. Had it been reported, this dissertation would have concluded "
              "that lineage awareness harms rare-class performance. No number of "
              "additional seeds could have rescued it, because the measurement noise "
              "exceeded the effect being measured.")
    para(doc,
         "The protocol was rebuilt around four changes. Early stopping was switched "
              "off and the budget fixed at 30 epochs, so every arm now receives the same "
              "anneal. Weight averaging was added, and the averaged weights are evaluated "
              "in place of the live ones. Predictions are averaged over four flip views "
              "at test time. Augmentation was strengthened to counter the memorisation "
              "visible when training loss fell to 0.005 while validation sat on a "
              "plateau. A fifth problem surfaced during the rebuild: the configuration "
              "field controlling augmentation policy never reached the dataset, so every "
              "run in the first matrix had trained under the basic policy no matter what "
              "its recorded settings claimed.")
    para(doc,
         "These changes reduced epoch-to-epoch validation variation from 0.0211 to "
              "0.0097, a reduction of 54%, measured on a matched configuration with "
              "identical data and seed. The accompanying accuracy gain was modest, "
              "approximately 0.011 in macro F1, and is not the point. Halving the "
              "measurement noise is what makes an effect of 0.005 to 0.017 detectable at "
              "all, and is the reason the cross-lineage comparison could reach "
              "significance on three seeds. The episode is reported because a study whose "
              "effects are of the same order as its measurement noise cannot be evaluated "
              "by its readers unless that noise is stated.")

    h3(doc, "4.4.2  Limitations")
    para(doc,
         "Five limitations qualify these results. First, three seeds provide two "
              "degrees of freedom, which is a thin basis for a paired t-test. Effect "
              "sizes and per-seed values are reported throughout so that conclusions do "
              "not rest on p-values alone, but several comparisons showing large effect "
              "sizes remain formally unresolved. The honest summary is that the "
              "hierarchical improvement in aggregate metrics is suggested, not "
              "demonstrated, and only the cross-lineage effect and the imbalance ablation "
              "are statistically supported.")
    para(doc,
         "Second, the rarest class contributes 23 training and 5 test images. Its "
              "per-class scores are correspondingly unstable, and minority-class "
              "aggregates computed over 671 test samples carry meaningful sampling "
              "uncertainty. The per-class differences discussed in Section 4.3.5 for "
              "individual rare classes should be read as indicative of the pattern, not "
              "as precise estimates.")
    para(doc,
         "Third, the backbone screen used a short budget and picked a backbone "
              "instead of comparing them, so this study licenses no claim about the "
              "relative merits of the four architectures. A full-length comparison "
              "remains outstanding, and the choice of Vision Transformer rests on weaker "
              "evidence than the rest of the design.")
    para(doc,
         "Fourth, MLL23 originates from a single laboratory. The generalisation "
              "problem identified in Chapter 2 as a central unsolved issue in this field "
              "is left untouched by this study, and the stain normalisation result in "
              "particular may not transfer to a multi-site setting.")
    para(doc,
         "Fifth, the minority threshold of 1,000 images is a simplification. Section "
              "4.3.5 shows that classes on either side of it behave along a continuum "
              "instead of splitting into two groups, and that the class-balanced "
              "objective redistributes capacity between classes that are both nominally "
              "minority. Aggregate minority statistics average over a mixed set, which "
              "the per-class table is included to make visible.")
    pagebreak(doc)
