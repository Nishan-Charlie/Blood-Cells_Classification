"""Two-level MLL23 label hierarchy.

Level 1 (``y1``) is the haematopoietic lineage, Level 2 (``y2``) the fine cell type.

Fine-class indices are NOT alphabetical. They are ordered lineage-by-lineage, and
within the myeloid branch they follow the maturation continuum
(myeloblast -> promyelocyte -> myelocyte -> metamyelocyte -> band -> segmented).
Confusion matrices plotted in index order therefore place biologically adjacent
maturation stages next to each other, which is where the clinically expected
errors land. Do not re-sort these.
"""

from __future__ import annotations

from dataclasses import dataclass

LINEAGES: tuple[str, ...] = ("Lymphoid", "Myeloid", "Erythroid")

LINEAGE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(LINEAGES)}


@dataclass(frozen=True)
class CellClass:
    """One fine-grained cell type.

    Attributes:
        idx: Level 2 class index in ``[0, 17]``.
        key: Stable snake_case identifier, also the Zenodo archive stem.
        name: Human-readable name as printed in the dissertation.
        lineage: Level 1 lineage name; must be a member of ``LINEAGES``.
        expected_count: Image count reported in the MLL23 data descriptor.
            Used to verify the download is complete and correctly mapped.
    """

    idx: int
    key: str
    name: str
    lineage: str
    expected_count: int

    @property
    def lineage_idx(self) -> int:
        return LINEAGE_TO_IDX[self.lineage]


# Ordered by lineage, then by maturation stage within the myeloid branch.
CLASSES: tuple[CellClass, ...] = (
    # --- Lymphoid -----------------------------------------------------------
    CellClass(0, "lymphocyte", "Typical lymphocytes", "Lymphoid", 5_532),
    CellClass(1, "hairy_cell", "Hairy cells", "Lymphoid", 3_265),
    CellClass(2, "lymphocyte_large_granular", "Large granular lymphocytes", "Lymphoid", 1_849),
    CellClass(3, "plasma_cell", "Atypical lymphocytes (plasma cells)", "Lymphoid", 1_658),
    CellClass(4, "smudge_cell", "Smudge cells", "Lymphoid", 988),
    CellClass(5, "lymphocyte_neoplastic", "Neoplastic lymphocytes", "Lymphoid", 180),
    CellClass(6, "lymphocyte_reactive", "Reactive lymphocytes", "Lymphoid", 33),
    # --- Myeloid, in maturation order ---------------------------------------
    CellClass(7, "myeloblast", "Myeloblasts", "Myeloid", 8_606),
    CellClass(8, "promyelocyte", "Promyelocytes", "Myeloid", 745),
    CellClass(9, "promyelocyte_atypical", "Atypical promyelocytes", "Myeloid", 2_033),
    CellClass(10, "myelocyte", "Myelocytes", "Myeloid", 747),
    CellClass(11, "metamyelocyte", "Metamyelocytes", "Myeloid", 483),
    CellClass(12, "neutrophil_band", "Band neutrophils", "Myeloid", 687),
    CellClass(13, "neutrophil_segmented", "Segmented neutrophils", "Myeloid", 7_170),
    CellClass(14, "eosinophil", "Eosinophil granulocytes", "Myeloid", 2_448),
    CellClass(15, "basophil", "Basophil granulocytes", "Myeloid", 616),
    CellClass(16, "monocyte", "Monocytes", "Myeloid", 2_510),
    # --- Erythroid ----------------------------------------------------------
    CellClass(17, "normoblast", "Normoblasts", "Erythroid", 2_071),
)

NUM_FINE_CLASSES = len(CLASSES)
NUM_LINEAGES = len(LINEAGES)
TOTAL_EXPECTED_IMAGES = sum(c.expected_count for c in CLASSES)

BY_KEY: dict[str, CellClass] = {c.key: c for c in CLASSES}
BY_IDX: dict[int, CellClass] = {c.idx: c for c in CLASSES}

#: Maps every Level 2 index to its Level 1 lineage index. Indexable by a tensor
#: of ``y2`` values to recover ``y1``, which the hierarchical loss relies on.
FINE_TO_LINEAGE: tuple[int, ...] = tuple(c.lineage_idx for c in CLASSES)

FINE_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)


def _validate() -> None:
    """Guard the invariants the rest of the pipeline assumes."""
    assert [c.idx for c in CLASSES] == list(range(NUM_FINE_CLASSES)), "indices must be 0..17 in order"
    assert len(BY_KEY) == NUM_FINE_CLASSES, "duplicate class key"
    assert set(c.lineage for c in CLASSES) <= set(LINEAGES), "unknown lineage"
    assert TOTAL_EXPECTED_IMAGES == 41_621, f"counts sum to {TOTAL_EXPECTED_IMAGES}, expected 41621"


_validate()
