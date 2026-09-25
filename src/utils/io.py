"""
Atomic file I/O -- ported and adapted from an external repository audit
(one of the genuinely real files found there; most of that repository's
measurement layer was fabricated, but its utils/ layer was solid).

WHY THIS MATTERS HERE SPECIFICALLY: RUNNING.md describes multi-hour-to-
multi-day training runs. A crash, OOM kill, or disconnect partway
through a plain `f.write(json.dumps(...))` can leave a truncated,
corrupt results file on disk -- which is worse than losing the run
entirely, because a partially-written JSON file can look like a
legitimate (if oddly incomplete) result rather than an obvious failure.
atomic_write_text() writes to a temp file in the same directory and
only replaces the target file with a single atomic os.replace() once
the write has fully succeeded, so a result file on disk is always
either the previous complete version or the new complete version --
never a partial one.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Streams the file in chunks rather than reading it fully into
    memory -- relevant here since the raw Dukascopy/ForexSB CSVs this
    project works with run into the hundreds of MB to low GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Not atomic (pandas/pyarrow's parquet writer doesn't expose a
    temp-file-then-replace hook the way plain text writes do) -- for
    genuinely crash-safe large-array writes, prefer save_json for
    metadata/results and let the .npy arrays in data/processed/ be
    regenerated from scripts/prepare_data.py if a run is interrupted,
    rather than relying on this function's atomicity."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
