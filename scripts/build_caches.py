"""Build both decoded-image caches. One-time, resumable, safe to re-run.

Run from the repository root:

    python scripts/build_caches.py

Re-running is cheap: :func:`cache.build_cache` verifies the sidecar first and
skips a cache that already matches the manifest.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache, config, splits  # noqa: E402


def main() -> None:
    df = splits.load()
    print(f"manifest rows: {len(df):,}")

    t0 = time.time()

    # --- raw cache: arms 1-4 ---------------------------------------------------
    cache.build_cache(df, cache.RAW_CACHE_PATH)
    print(f"raw cache elapsed: {(time.time() - t0) / 60:.1f} min\n")

    # --- Reinhard cache: arm 5 -------------------------------------------------
    # Reuse the reference notebook 02 fitted on the training split, so the
    # stain-normalised arm and the notebook-02 stain figure describe the same
    # transformation. Fitting on train only avoids leaking test colour statistics.
    ref_path = config.INTERIM_DIR / "reinhard_ref.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        print(f"using existing Reinhard reference from {ref_path.name}")
    else:
        ref = cache.fit_reinhard_reference(df[df.split == "train"])
        ref_path.write_text(json.dumps(ref))
        print(f"fitted new Reinhard reference -> {ref_path.name}")

    t1 = time.time()
    cache.build_cache(df, cache.REINHARD_CACHE_PATH, reinhard_ref=ref)
    print(f"reinhard cache elapsed: {(time.time() - t1) / 60:.1f} min")

    print(f"\ntotal: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
