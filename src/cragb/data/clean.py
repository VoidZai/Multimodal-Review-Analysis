"""Pure, testable cleaning filters for review records (PLAN.md §3 E0).

Each function takes a DataFrame in and returns a filtered/transformed
DataFrame out — no I/O, no config reads, no global state beyond
`langdetect`'s deterministic seeding. That makes each one unit-testable in
isolation on small synthetic frames, which matters because a silently
wrong filter (e.g. one that drops 90% of rows by an off-by-one on
`min_chars`) would invalidate every downstream sample without any error
being raised.

Recommended application order (cheapest/least-destructive first):
    1. `normalize_whitespace` — collapse whitespace before anything else
       inspects text length or content, so "  " isn't mistaken for
       non-empty and language detection sees clean text.
    2. `drop_empty_text` — cheap, removes rows with nothing worth
       language-detecting.
    3. `dedup` — cheap, shrinks the frame before the expensive step.
    4. `filter_language` — the most expensive filter (runs a language
       classifier per row), so it runs last, over the smallest possible
       input.

This is a deliberate deviation from the listing order in PLAN.md §3 E0
("dedup, language filter, drop empty/near-empty text, normalise
whitespace") — that list describes *what* to do, not a required sequence,
and running language detection before whitespace normalization risks
misclassifying text that is only "empty" because of unstripped whitespace.
T1.8 (the end-to-end build) is what actually fixes the applied order; here
each filter is independent by design so that choice is free to make later.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger(__name__)

# langdetect's classifier draws pseudo-random n-gram samples internally;
# without a fixed seed, calling detect() on the same text twice can
# occasionally return different results. Seed once at import time so
# language filtering is reproducible.
DetectorFactory.seed = 0

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Collapse runs of whitespace (including newlines/tabs) to single
    spaces and strip leading/trailing whitespace, in the given columns.

    Non-string values (e.g. NaN) are left untouched.

    Args:
        df: input frame.
        columns: names of text columns to normalize.

    Returns:
        A new DataFrame (input is not mutated) with the given columns
        whitespace-normalized.
    """
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(
            lambda v: _WHITESPACE_RE.sub(" ", v).strip() if isinstance(v, str) else v
        )
    return out


def drop_empty_text(df: pd.DataFrame, text_col: str, min_chars: int) -> pd.DataFrame:
    """Drop rows whose `text_col` is missing or shorter than `min_chars`
    after stripping whitespace.

    Args:
        df: input frame. Call `normalize_whitespace` first if the column
            may contain unstripped whitespace, so a "   " value isn't
            mistaken for `min_chars` worth of content.
        text_col: column to check.
        min_chars: minimum stripped character length to keep a row.

    Returns:
        A new, filtered DataFrame (original index preserved on kept rows).
    """
    lengths = df[text_col].fillna("").astype(str).str.strip().str.len()
    keep = lengths >= min_chars
    dropped = int((~keep).sum())
    if dropped:
        logger.info("drop_empty_text: dropped %d/%d rows (< %d chars in %r)", dropped, len(df), min_chars, text_col)
    return df[keep]


def dedup(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Drop exact duplicate rows on `subset`, keeping the first occurrence.

    Args:
        df: input frame.
        subset: columns that jointly identify a duplicate (e.g.
            `[user_id, parent_asin, text]` — the same user posting the
            same text on the same product).

    Returns:
        A new, deduplicated DataFrame.
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(out)
    if dropped:
        logger.info("dedup: dropped %d/%d duplicate rows on %r", dropped, before, subset)
    return out


def filter_language(df: pd.DataFrame, text_col: str, keep_languages: list[str]) -> pd.DataFrame:
    """Keep only rows whose `text_col` is detected as one of `keep_languages`.

    Detection failures (e.g. text too short/ambiguous for `langdetect` to
    classify, which raises `LangDetectException`) are treated as "drop, not
    keep" — an unclassifiable review is not safely assumed to be English.

    Args:
        df: input frame. Run `drop_empty_text` first: detection on
            very short strings is unreliable and wastes the call.
        text_col: column to detect language on.
        keep_languages: ISO 639-1 codes to keep (e.g. `["en"]`).

    Returns:
        A new, filtered DataFrame.
    """
    keep_set = set(keep_languages)

    def _keep(text: object) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        try:
            return detect(text) in keep_set
        except LangDetectException:
            return False

    mask = df[text_col].map(_keep)
    dropped = int((~mask).sum())
    if dropped:
        logger.info("filter_language: dropped %d/%d rows not in %r", dropped, len(df), keep_languages)
    return df[mask]
