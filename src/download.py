"""Fetch the MLL23 dataset from Zenodo.

Downloads one class archive at a time, verifies its MD5 against the checksum
Zenodo publishes, extracts it, then deletes the archive before moving on. Peak
disk usage is therefore the extracted corpus plus a single zip (~2 GB) rather
than the ~10 GB the archives would occupy all at once.

Every stage is resumable: an already-extracted class is skipped, and a partial
download resumes via an HTTP Range request.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from . import config
from .hierarchy import BY_KEY, CellClass

API_URL = f"https://zenodo.org/api/records/{config.ZENODO_RECORD_ID}"
_CHUNK = 1 << 20  # 1 MiB
_MAX_BACKOFF_SECONDS = 120


def fetch_record() -> dict:
    """Return the Zenodo record metadata as a dict."""
    with urllib.request.urlopen(API_URL, timeout=60) as resp:
        return json.load(resp)


def archive_index(record: dict) -> dict[str, dict]:
    """Map class key -> {url, size, md5} for the 18 class archives.

    Raises:
        ValueError: if the record's archives do not correspond exactly to the
            18 classes declared in :mod:`hierarchy`. A mismatch means the
            dataset was revised and the hierarchy needs revisiting, so we fail
            loudly rather than silently training on the wrong label set.
    """
    index: dict[str, dict] = {}
    for entry in record["files"]:
        filename = entry.get("key") or entry.get("filename", "")
        if not filename.endswith(".zip"):
            continue
        key = filename[: -len(".zip")]
        checksum = entry.get("checksum", "")
        index[key] = {
            "url": entry["links"]["self"],
            "size": entry.get("size", 0),
            "md5": checksum.split(":")[-1] if checksum else "",
        }

    missing = set(BY_KEY) - set(index)
    unexpected = set(index) - set(BY_KEY)
    if missing or unexpected:
        raise ValueError(
            f"Zenodo record does not match the declared hierarchy. "
            f"Missing archives: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
        )
    return index


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


class IncompleteDownload(Exception):
    """The transfer ended before the full archive arrived."""


class ChecksumMismatch(Exception):
    """The archive downloaded fully but does not match the published MD5."""


#: Transient faults worth retrying. A multi-GB transfer over several concurrent
#: connections will hit these; they say nothing about the data itself.
_RETRYABLE = (
    TimeoutError,
    urllib.error.URLError,
    ConnectionError,
    http.client.HTTPException,
    IncompleteDownload,
    ChecksumMismatch,
)


def download_archive(
    key: str,
    meta: dict,
    *,
    verify: bool = True,
    quiet: bool = False,
    attempts: int = 9,
) -> Path:
    """Download one archive with retries, resuming a partial file if present.

    Args:
        quiet: Suppress per-chunk progress. Set when several downloads run
            concurrently, since interleaved percentages are unreadable.
        attempts: Tries before giving up. Each retry resumes from the bytes
            already on disk rather than restarting, so a late failure on a 2 GB
            archive costs seconds, not the whole transfer.

            The default budget spans roughly six minutes of backoff, chosen to
            outlast a dropped Wi-Fi link rather than merely a slow response. An
            earlier 6-attempt/30 s-cap budget gave up after ~60 s and lost four
            archives to a transient outage.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(key, meta, verify=verify, quiet=quiet)
        except _RETRYABLE as exc:
            last = exc
            if attempt == attempts:
                break
            backoff = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
            print(
                f"  [retry] {key}: {type(exc).__name__} ({exc}); "
                f"attempt {attempt + 1}/{attempts} in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)

    raise RuntimeError(f"{key}: giving up after {attempts} attempts") from last


def _download_once(key: str, meta: dict, *, verify: bool, quiet: bool) -> Path:
    dest = config.ARCHIVE_DIR / f"{key}.zip"
    expected = meta["size"]

    if dest.exists() and dest.stat().st_size == expected:
        if not quiet:
            print(f"  archive already present ({_human(expected)})")
    else:
        start = dest.stat().st_size if dest.exists() else 0
        if start and start > expected:
            # Overshoot means a corrupt leftover; start clean.
            dest.unlink()
            start = 0

        req = urllib.request.Request(meta["url"])
        mode = "wb"
        if start:
            req.add_header("Range", f"bytes={start}-")
            mode = "ab"
            if not quiet:
                print(f"  resuming at {_human(start)} / {_human(expected)}")

        with urllib.request.urlopen(req, timeout=120) as resp, dest.open(mode) as fh:
            done = start
            last_pct = -5
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                pct = int(done * 100 / expected) if expected else 0
                if not quiet and pct >= last_pct + 5:
                    print(f"    {pct:3d}%  {_human(done)} / {_human(expected)}", flush=True)
                    last_pct = pct

    # A server that closes the stream early ends the read loop without raising,
    # leaving a short file. Treat that as retryable: the next attempt resumes
    # from whatever did land.
    actual_size = dest.stat().st_size
    if expected and actual_size < expected:
        raise IncompleteDownload(
            f"{key}.zip truncated at {_human(actual_size)} of {_human(expected)}"
        )

    if verify and meta["md5"]:
        actual = _md5(dest)
        if actual != meta["md5"]:
            # Delete so the retry restarts cleanly rather than resuming onto
            # bytes we already know are wrong.
            dest.unlink()
            raise ChecksumMismatch(
                f"MD5 mismatch for {key}.zip (got {actual}, want {meta['md5']}); deleted"
            )
        if not quiet:
            print("  md5 ok")
    return dest


def extract_archive(archive: Path, key: str) -> int:
    """Extract every image in ``archive`` into ``raw/<key>/``, flattening paths.

    Returns:
        Number of image files written.
    """
    target = config.RAW_DIR / key
    target.mkdir(parents=True, exist_ok=True)

    written = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                continue
            out = target / name
            if not out.exists():
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst, _CHUNK)
            written += 1
    return written


def class_is_complete(cls: CellClass) -> bool:
    """True if the class folder already holds the expected number of images."""
    folder = config.RAW_DIR / cls.key
    if not folder.is_dir():
        return False
    return sum(1 for _ in folder.iterdir()) == cls.expected_count


def _fetch_one(key: str, meta: dict, *, keep_archives: bool, quiet: bool) -> int:
    """Download -> verify -> extract -> delete for a single class."""
    cls = BY_KEY[key]
    if class_is_complete(cls):
        print(f"  [skip] {cls.name}: already extracted ({cls.expected_count:,})", flush=True)
        return cls.expected_count

    archive = download_archive(key, meta, quiet=quiet)
    n = extract_archive(archive, key)
    if not keep_archives:
        archive.unlink()

    status = "ok" if n == cls.expected_count else f"WARNING expected {cls.expected_count:,}"
    print(f"  [done] {cls.name}: {n:,} images  [{status}]", flush=True)
    return n


def download_all(
    keys: Iterable[str] | None = None,
    *,
    keep_archives: bool = False,
    workers: int = 4,
) -> dict[str, int]:
    """Download, verify, and extract the requested classes.

    Args:
        keys: Class keys to fetch; defaults to all 18.
        keep_archives: Retain the zips instead of deleting after extraction.
            Leave False unless disk space is plentiful.
        workers: Concurrent downloads. Zenodo throttles per connection rather
            than per client, so several connections multiply total throughput
            (measured ~45 MB/min on one, ~88 MB/min across three). Each worker
            holds at most one archive on disk at a time, so peak transient usage
            is roughly ``workers`` archives. Set to 1 for serial, readable logs.

    Returns:
        Mapping of class key to the number of images on disk.
    """
    config.ensure_dirs()
    index = archive_index(fetch_record())
    targets = list(keys) if keys is not None else [c.key for c in BY_KEY.values()]

    pending = [k for k in targets if not class_is_complete(BY_KEY[k])]
    done = {k: BY_KEY[k].expected_count for k in targets if k not in pending}
    if done:
        print(f"{len(done)} class(es) already extracted -- skipping\n")
    if not pending:
        return done

    # Largest first: the 1.9 GB archive should start immediately rather than
    # becoming a straggler that runs alone after the small ones finish.
    pending.sort(key=lambda k: index[k]["size"], reverse=True)

    total = sum(index[k]["size"] for k in pending)
    print(f"fetching {len(pending)} archives ({_human(total)}) with {workers} workers\n", flush=True)

    counts: dict[str, int] = dict(done)
    if workers == 1:
        for key in pending:
            counts[key] = _fetch_one(key, index[key], keep_archives=keep_archives, quiet=False)
        return counts

    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, k, index[k], keep_archives=keep_archives, quiet=True): k
            for k in pending
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                counts[key] = fut.result()
            except Exception as exc:  # noqa: BLE001 - one class must not sink the rest
                # Report at the end rather than aborting: the other archives are
                # independent, and re-running resumes only what is missing.
                failures[key] = exc
                print(f"  [FAIL] {BY_KEY[key].name}: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print(
            f"\n{len(failures)} class(es) failed: {sorted(failures)}"
            f"\nRe-run download_all() to resume just those.",
            flush=True,
        )
    return counts
