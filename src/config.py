"""Central configuration.

Every path and hyper-parameter the pipeline depends on is declared here so the
four experiment configurations (flat baseline, full hierarchical, and the two
ablations) cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# The dataset lives off the system drive: C: sits at ~97% capacity and the
# extracted TIFFs need ~10 GB. Override with the MLL23_ROOT environment variable.
DATA_ROOT = Path(os.environ.get("MLL23_ROOT", r"D:\MLL23"))

RAW_DIR = DATA_ROOT / "raw"          # one folder of TIFFs per class key
ARCHIVE_DIR = DATA_ROOT / "archives" # transient; zips deleted after extraction
INTERIM_DIR = DATA_ROOT / "interim"  # manifest + splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"  # figures and tables for the write-up

MANIFEST_PATH = INTERIM_DIR / "manifest.csv"
SPLIT_PATH = INTERIM_DIR / "splits.csv"

# --- Data --------------------------------------------------------------------
ZENODO_RECORD_ID = "14277609"
ZENODO_DOI = "10.5281/zenodo.14277609"

SOURCE_IMAGE_SIZE = 288  # native MLL23 resolution, in pixels

#: Input resolution for *every* backbone including the ViT. Holding resolution
#: constant is what keeps the backbone comparison controlled; see CLAUDE.md.
IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Splitting ---------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SPLIT_SEED = 42  # the split must be deterministic across all experiments

# --- Augmentation ------------------------------------------------------------
MAX_ROTATION_DEGREES = 90.0
JITTER_STRENGTH = 0.1

#: Classes at or below this count receive the intensified minority regime.
#: Captures eight classes (smudge 988 down to reactive lymphocytes 33); the
#: next class up is atypical lymphocytes at 1,658, so the threshold sits in a
#: natural gap rather than splitting a cluster.
MINORITY_THRESHOLD = 1_000

#: Multiplier applied to rotation and jitter for minority classes.
MINORITY_AUG_MULTIPLIER = 2.0


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (RAW_DIR, ARCHIVE_DIR, INTERIM_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)
