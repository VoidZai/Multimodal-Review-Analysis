"""Memory-safe streaming access to the raw JSON Lines source files.

The raw files (`data/raw/reviews_clothing.jsonl.gz`,
`data/raw/meta_clothing.jsonl.gz`) are multi-gigabyte gzip archives with
tens of millions of lines. Nothing in this module ever materializes a full
file in memory: `stream_records` is a generator that decodes one line at a
time, and every other helper here is built on top of it with an explicit
`limit` so callers opt in to how much they load.

This module intentionally does not filter, clean, or join anything (that's
T1.4/T1.5) — it only answers "can I read the files, and what's actually in
them," which must be settled before any filtering logic is written.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from cragb.utils.io import resolve_path


def _open_text(path: Path):
    """Open a file as text, transparently handling gzip by extension."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def stream_records(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per line, without loading the whole file.

    Args:
        path: path to a `.jsonl` or `.jsonl.gz` file, absolute or relative
            to the repo root.
        limit: if given, stop after yielding this many records. `None`
            streams the entire file — safe for the multi-GB source files
            precisely because nothing is buffered beyond one line at a
            time plus whatever the caller itself accumulates.

    Yields:
        One dict per non-empty line, in file order.

    Raises:
        json.JSONDecodeError: if a line is not valid JSON. Line contents
            are not swallowed silently — a malformed source file should
            fail loudly here rather than produce a silently-truncated
            corpus downstream.
    """
    resolved = resolve_path(path)
    with _open_text(resolved) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_records(path: str | Path) -> int:
    """Count records via a single streamed pass (no full-file buffering).

    Useful for sanity-checking a file's row count against the dataset
    card without ever holding more than one line in memory at a time.
    """
    resolved = resolve_path(path)
    n = 0
    with _open_text(resolved) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def sample_dataframe(path: str | Path, n: int) -> pd.DataFrame:
    """Materialize the first `n` records of a JSONL(.gz) file as a DataFrame.

    This is deliberately bounded: `n` records are read via `stream_records`
    and handed to `pandas.json_normalize` in one shot. It is meant for
    schema inspection and quick EDA on a sample, not for building the full
    corpus (see T1.7/T1.8 for the stratified, streamed build).

    Args:
        path: path to a `.jsonl` or `.jsonl.gz` file.
        n: number of leading records to materialize.

    Returns:
        A DataFrame with one row per record and one column per top-level
        JSON key seen in the sample (nested dict/list fields, e.g.
        `images` or `details`, are kept as Python objects rather than
        flattened, since their structure varies by record).
    """
    records = list(stream_records(path, limit=n))
    return pd.json_normalize(records, max_level=0)


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, null count/rate, and an example value.

    Args:
        df: a DataFrame as produced by `sample_dataframe`.

    Returns:
        A DataFrame indexed by column name with columns `dtype`,
        `null_count`, `null_pct`, and `example` (the first non-null value,
        repr'd and truncated so long review text doesn't blow up the
        display).
    """
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        example = repr(non_null.iloc[0])[:120] if not non_null.empty else None
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(100 * series.isna().mean(), 2),
                "example": example,
            }
        )
    return pd.DataFrame(rows).set_index("column")
