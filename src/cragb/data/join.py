"""Join review records to product metadata on `parent_asin` (PLAN.md §3 E0).

The proposal explicitly flags **silent join loss on `parent_asin`** as a
failure mode (Risk in PLAN.md §1.4, listed again under E0's "Failure
modes"): a left-join that quietly drops or fails to match rows would
shrink the corpus without anyone noticing, and — worse — could bias it
if the losses aren't random (e.g. discontinued products missing from a
metadata snapshot). Every function here is built so that loss is
*measured and returned*, never just implicit in a smaller output frame.

Two column-naming collisions exist between the two source schemas and
must be resolved before merging, not after:
- `title` — review title vs. product title.
- `images` — review-attached photos vs. product listing photos.
Both are handled by `prepare_metadata_for_join`, which renames the
metadata-side columns with a `product_` prefix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that exist in both schemas (see notebooks/00_schema_probe.ipynb)
# under the same name but mean different things; renamed on the metadata
# side before merging so neither is silently overwritten by a pandas
# merge suffix.
_METADATA_RENAME = {
    "title": "product_title",
    "images": "product_images",
}


@dataclass(frozen=True)
class JoinReport:
    """Summary of a reviews-to-metadata join, for logging and datasheets."""

    total_reviews: int
    matched_reviews: int
    unmatched_reviews: int
    match_rate: float
    metadata_rows_in: int
    metadata_duplicate_parent_asin_dropped: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_reviews": self.total_reviews,
            "matched_reviews": self.matched_reviews,
            "unmatched_reviews": self.unmatched_reviews,
            "match_rate": self.match_rate,
            "metadata_rows_in": self.metadata_rows_in,
            "metadata_duplicate_parent_asin_dropped": self.metadata_duplicate_parent_asin_dropped,
        }


def prepare_metadata_for_join(metadata_df: pd.DataFrame, on: str = "parent_asin") -> tuple[pd.DataFrame, int]:
    """Rename colliding columns and enforce one row per `on` key.

    Product metadata is expected to have at most one row per
    `parent_asin`; if the raw file contains duplicates (a data-quality
    issue seen upstream in some Amazon Reviews dumps), keeping the first
    occurrence is a deliberate, logged choice rather than an accidental
    one — a merge against a non-unique right-hand key silently multiplies
    review rows, which is a worse failure than dropping a few duplicate
    metadata records.

    Args:
        metadata_df: raw product metadata frame.
        on: the join key column.

    Returns:
        A tuple of (renamed + deduplicated metadata frame, number of
        duplicate-key rows dropped).
    """
    renamed = metadata_df.rename(columns=_METADATA_RENAME)
    before = len(renamed)
    deduped = renamed.drop_duplicates(subset=[on], keep="first")
    n_dropped = before - len(deduped)
    if n_dropped:
        logger.warning(
            "prepare_metadata_for_join: dropped %d/%d metadata rows with duplicate %r "
            "(kept first occurrence per key)",
            n_dropped,
            before,
            on,
        )
    return deduped, n_dropped


def join_reviews_metadata(
    reviews_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    on: str = "parent_asin",
) -> tuple[pd.DataFrame, JoinReport]:
    """Left-join reviews to metadata on `parent_asin`, measuring join loss.

    A left join is used deliberately: every review is kept even if its
    product metadata is missing (e.g. a delisted product), so join loss
    is visible in the output (`product_title` etc. will be NaN) and
    reported in `JoinReport` rather than the review silently vanishing.

    Args:
        reviews_df: review records; must contain `on`.
        metadata_df: product metadata; must contain `on`. Will be passed
            through `prepare_metadata_for_join` internally.
        on: the join key column.

    Returns:
        `(merged_df, report)` — `merged_df` has one row per input review
        (never more, never fewer) with metadata columns attached (NaN
        where unmatched); `report` is the loss summary.

    Raises:
        KeyError: if `on` is missing from either input frame.
    """
    if on not in reviews_df.columns:
        raise KeyError(f"reviews_df is missing join key {on!r}")
    if on not in metadata_df.columns:
        raise KeyError(f"metadata_df is missing join key {on!r}")

    metadata_clean, n_dup_dropped = prepare_metadata_for_join(metadata_df, on=on)

    total_reviews = len(reviews_df)
    merged = reviews_df.merge(metadata_clean, on=on, how="left", indicator=True, validate="m:1")

    matched_mask = merged["_merge"] == "both"
    matched = int(matched_mask.sum())
    unmatched = total_reviews - matched
    match_rate = matched / total_reviews if total_reviews else 0.0
    merged = merged.drop(columns=["_merge"])

    report = JoinReport(
        total_reviews=total_reviews,
        matched_reviews=matched,
        unmatched_reviews=unmatched,
        match_rate=round(match_rate, 4),
        metadata_rows_in=len(metadata_df),
        metadata_duplicate_parent_asin_dropped=n_dup_dropped,
    )

    if total_reviews and match_rate < 1.0:
        logger.info(
            "join_reviews_metadata: %d/%d reviews (%.2f%%) matched metadata on %r; "
            "%d unmatched (likely delisted/missing products)",
            matched,
            total_reviews,
            100 * match_rate,
            on,
            unmatched,
        )

    # merge() with how="left" must never change the review row count.
    assert len(merged) == total_reviews, (
        f"join invariant violated: {len(merged)} output rows for {total_reviews} input reviews "
        "(a supposedly-deduplicated metadata key must have matched more than once)"
    )

    return merged, report
