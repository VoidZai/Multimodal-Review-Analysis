"""Attribute-term vocabulary coverage check (T1.11; PLAN.md §3 E0).

Before any question taxonomy is drafted (E1), we need to know the corpus
actually contains enough size/fit/colour/fabric language to support
questions like "does this run small?" or "is the colour as pictured?" —
otherwise a taxonomy built on assumption could ask questions the corpus
has no evidence for, an error that would only surface much later during
CRAGB pooling (E1). This module answers that with a plain keyword search
rather than a lexicon library or an LLM call, on purpose: at this stage a
transparent, auditable word list is more useful than a black-box
classifier, and the check is meant to catch a *near-zero-coverage*
failure, not to produce a precise linguistic count.

Categories checked, per M1.md/T1.11: size, fit, colour, fabric. Keyword
lists are deliberately curated and small — see `ATTRIBUTE_KEYWORDS` for
the exact terms and why each is there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Curated per category. Terms may overlap across categories (e.g. "true
# to size" is both a sizing and a fit signal) — categories are coverage
# checks, not a mutually-exclusive classification, so overlap is fine.
ATTRIBUTE_KEYWORDS: dict[str, list[str]] = {
    "size": [
        "size", "sizing", "sized", "small", "large", "medium", "petite",
        "plus size", "true to size", "runs small", "runs large", "runs big",
        "size up", "size down", "true-to-size", "oversized", "undersized",
    ],
    "fit": [
        "fit", "fits", "fitted", "fitting", "loose", "tight", "snug",
        "comfortable", "comfy", "roomy", "narrow", "wide", "baggy",
        "true to size", "runs small", "runs large",
    ],
    "colour": [
        "color", "colors", "colour", "colours", "colored", "coloured",
        "shade", "hue", "faded", "fade", "fades", "dye", "colorfast",
        "colourfast", "as pictured", "picture", "photo",
    ],
    "fabric": [
        "fabric", "material", "materials", "cotton", "polyester", "wool",
        "silk", "leather", "spandex", "nylon", "denim", "linen", "fleece",
        "quality", "stitching", "stitch", "seam", "seams", "thread",
        "threads", "cheaply made", "well made",
    ],
}


def _compile_category_pattern(terms: list[str]) -> re.Pattern:
    """Compile a case-insensitive, word-boundary alternation over `terms`.

    Multi-word phrases (e.g. "true to size") work correctly here: `\\b`
    only requires a word/non-word transition at each end of the whole
    matched phrase, which `re.escape` + surrounding spaces preserves.
    """
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


@dataclass(frozen=True)
class CategoryCoverage:
    category: str
    n_reviews_matched: int
    pct_reviews_matched: float
    n_keyword_hits_total: int
    top_terms: list[tuple[str, int]]


def attribute_vocab_coverage(
    df: pd.DataFrame,
    text_col: str,
    keyword_lists: dict[str, list[str]] | None = None,
    top_n_terms: int = 5,
) -> pd.DataFrame:
    """Check what fraction of reviews mention each attribute category.

    Args:
        df: reviews frame.
        text_col: column containing review text.
        keyword_lists: category -> keyword list; defaults to
            `ATTRIBUTE_KEYWORDS`.
        top_n_terms: how many of each category's most-frequent matched
            terms to report, as a diagnostic for which specific words are
            actually carrying the coverage.

    Returns:
        A DataFrame indexed by category with columns `n_reviews_matched`,
        `pct_reviews_matched`, `n_keyword_hits_total`, `top_terms` (a list
        of `(term, count)` pairs, most frequent first).
    """
    keyword_lists = keyword_lists or ATTRIBUTE_KEYWORDS
    text = df[text_col].fillna("").astype(str)

    rows: list[CategoryCoverage] = []
    for category, terms in keyword_lists.items():
        term_pattern = _compile_category_pattern(terms)
        matched_mask = text.str.contains(term_pattern, regex=True)
        n_matched = int(matched_mask.sum())

        # Per-term counts (for the "top_terms" diagnostic): count reviews
        # containing each individual term, not overlapping match spans.
        term_counts: dict[str, int] = {}
        for term in terms:
            single_pattern = _compile_category_pattern([term])
            term_counts[term] = int(text.str.contains(single_pattern, regex=True).sum())
        top_terms = sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n_terms]

        rows.append(
            CategoryCoverage(
                category=category,
                n_reviews_matched=n_matched,
                pct_reviews_matched=round(100 * n_matched / len(df), 2) if len(df) else 0.0,
                n_keyword_hits_total=sum(term_counts.values()),
                top_terms=top_terms,
            )
        )

    out = pd.DataFrame([vars(r) for r in rows]).set_index("category")
    return out
