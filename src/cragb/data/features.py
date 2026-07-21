"""Per-review feature computation: token length and image parsing (T1.6).

Two feature families, both needed before stratified sampling (T1.7) can
run and before the EDA notebook (T1.9-11) can plot anything:

1. **Token length** — a cheap length proxy per review, for the
   review-length EDA figure and for later chunking decisions (E2).
2. **Image fields** — the `images` column (confirmed via
   `notebooks/00_schema_probe.ipynb` and ad-hoc inspection to always be a
   list of dicts with a consistent `{small,medium,large}_image_url` +
   `attachment_type` shape, empty when the review has no photos) is
   parsed into a flat list of usable URLs, a count, and a boolean flag —
   the last of which is exactly the "image-bearing" stratification axis
   PLAN.md §3 E0 requires to avoid Risk C (random sampling starving
   negative/defect questions, which correlate with the image-bearing
   minority).

Tokenization note: `compute_token_length` uses a simple regex
word-tokenizer (`\\w+` sequences and standalone punctuation), *not* a
model-specific tokenizer like `tiktoken`. At this stage we only need a
stable, cheap, dependency-light proxy for relative review length across
hundreds of thousands of rows (EDA distributions, stratification). The
*exact* token count for LLM context budgeting is a different, later
concern (E4/E5/E6), where it will be computed with whatever tokenizer
matches the actual generator model in use — pinning that choice now,
before a generator is chosen, would be false precision.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

# Preference order when picking a single representative URL per image.
_URL_KEYS_BY_PREFERENCE = ("large_image_url", "medium_image_url", "small_image_url")


def compute_token_length(text: object) -> int:
    """Approximate token count of `text` via a whitespace/punctuation regex.

    Args:
        text: review text (or title). Non-string / missing values count
            as zero tokens rather than raising.

    Returns:
        Number of word and punctuation tokens matched. `""`, `None`, and
        `NaN` all yield `0`.
    """
    if not isinstance(text, str):
        return 0
    return len(_TOKEN_RE.findall(text))


def parse_image_urls(images: object) -> list[str]:
    """Extract one representative URL per image entry, preferring the
    highest-resolution variant available.

    Args:
        images: the raw `images` field — expected to be a list of dicts
            each carrying some subset of `large_image_url`,
            `medium_image_url`, `small_image_url`. Any other shape
            (`None`, `NaN`, a plain list of strings, malformed dicts) is
            handled defensively rather than raising, since this runs over
            every row of an untrusted external file.

    Returns:
        A list of URL strings, one per parseable image entry, in the
        original order. Entries with none of the expected keys are
        skipped (not raised on), since a single malformed image record
        should not fail an entire review.
    """
    if not isinstance(images, list):
        return []

    urls: list[str] = []
    for entry in images:
        if isinstance(entry, str):
            urls.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        for key in _URL_KEYS_BY_PREFERENCE:
            value = entry.get(key)
            if isinstance(value, str) and value:
                urls.append(value)
                break
    return urls


def has_image(images: object) -> bool:
    """Whether a review has at least one parseable image.

    Args:
        images: the raw `images` field, as in `parse_image_urls`.

    Returns:
        `True` if `parse_image_urls` would return a non-empty list.
    """
    return len(parse_image_urls(images)) > 0


def add_review_features(
    df: pd.DataFrame,
    text_col: str = "text",
    images_col: str = "images",
) -> pd.DataFrame:
    """Attach `token_len`, `image_urls`, `n_images`, `has_image` columns.

    Args:
        df: reviews frame (raw or already cleaned).
        text_col: column to compute token length on.
        images_col: column holding the raw per-review image list.

    Returns:
        A new DataFrame (input not mutated) with the four feature
        columns appended.
    """
    out = df.copy()
    out["token_len"] = out[text_col].map(compute_token_length)
    out["image_urls"] = out[images_col].map(parse_image_urls)
    out["n_images"] = out["image_urls"].map(len)
    out["has_image"] = out["n_images"] > 0
    return out
