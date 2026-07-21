"""Unit tests for cragb.data.join.

The central risk this module guards against (PLAN.md: "silent join loss
on parent_asin") means these tests care less about the merge mechanics
pandas already guarantees, and more about: nothing is silently dropped,
the row count invariant holds, column collisions are actually resolved,
and the reported loss numbers match reality.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.data.join import join_reviews_metadata, prepare_metadata_for_join


def _reviews(rows):
    return pd.DataFrame(rows)


def _metadata(rows):
    return pd.DataFrame(rows)


class TestPrepareMetadataForJoin:
    def test_renames_colliding_columns(self):
        meta = _metadata([{"parent_asin": "P1", "title": "Product One", "images": ["img1"]}])
        out, _ = prepare_metadata_for_join(meta)
        assert "product_title" in out.columns
        assert "product_images" in out.columns
        assert "title" not in out.columns
        assert "images" not in out.columns

    def test_drops_duplicate_keys_keeping_first(self):
        meta = _metadata(
            [
                {"parent_asin": "P1", "title": "First version"},
                {"parent_asin": "P1", "title": "Second version (stale)"},
                {"parent_asin": "P2", "title": "Only version"},
            ]
        )
        out, n_dropped = prepare_metadata_for_join(meta)
        assert n_dropped == 1
        assert len(out) == 2
        kept = out.loc[out["parent_asin"] == "P1", "product_title"].iloc[0]
        assert kept == "First version"

    def test_no_duplicates_reports_zero_dropped(self):
        meta = _metadata([{"parent_asin": "P1", "title": "A"}, {"parent_asin": "P2", "title": "B"}])
        out, n_dropped = prepare_metadata_for_join(meta)
        assert n_dropped == 0
        assert len(out) == 2


class TestJoinReviewsMetadata:
    def test_every_review_matches_when_metadata_is_complete(self):
        reviews = _reviews(
            [
                {"parent_asin": "P1", "text": "great fit"},
                {"parent_asin": "P2", "text": "runs small"},
            ]
        )
        meta = _metadata([{"parent_asin": "P1", "title": "Shirt"}, {"parent_asin": "P2", "title": "Shoes"}])
        merged, report = join_reviews_metadata(reviews, meta)

        assert report.total_reviews == 2
        assert report.matched_reviews == 2
        assert report.unmatched_reviews == 0
        assert report.match_rate == 1.0
        assert list(merged["product_title"]) == ["Shirt", "Shoes"]

    def test_output_row_count_always_equals_input_review_count(self):
        # This is the core invariant: a left join must never grow or
        # shrink the review set, regardless of match/miss/duplicate noise.
        reviews = _reviews(
            [
                {"parent_asin": "P1", "text": "a"},
                {"parent_asin": "MISSING", "text": "b"},
                {"parent_asin": "P1", "text": "c"},
            ]
        )
        meta = _metadata(
            [
                {"parent_asin": "P1", "title": "X"},
                {"parent_asin": "P1", "title": "X duplicate"},  # will be deduped internally
            ]
        )
        merged, report = join_reviews_metadata(reviews, meta)
        assert len(merged) == 3
        assert report.total_reviews == 3

    def test_unmatched_reviews_are_kept_with_nan_metadata_not_dropped(self):
        reviews = _reviews([{"parent_asin": "P1", "text": "a"}, {"parent_asin": "GONE", "text": "b"}])
        meta = _metadata([{"parent_asin": "P1", "title": "X"}])
        merged, report = join_reviews_metadata(reviews, meta)

        assert report.matched_reviews == 1
        assert report.unmatched_reviews == 1
        assert report.match_rate == 0.5
        assert len(merged) == 2  # unmatched row kept, not dropped
        gone_row = merged.loc[merged["parent_asin"] == "GONE"]
        assert gone_row["product_title"].isna().all()

    def test_review_title_is_not_overwritten_by_product_title(self):
        reviews = _reviews([{"parent_asin": "P1", "text": "a", "title": "My Review Title"}])
        meta = _metadata([{"parent_asin": "P1", "title": "Product Listing Title"}])
        merged, _ = join_reviews_metadata(reviews, meta)

        assert merged["title"].iloc[0] == "My Review Title"
        assert merged["product_title"].iloc[0] == "Product Listing Title"

    def test_missing_join_key_in_reviews_raises(self):
        reviews = _reviews([{"text": "no key column"}])
        meta = _metadata([{"parent_asin": "P1", "title": "X"}])
        with pytest.raises(KeyError):
            join_reviews_metadata(reviews, meta)

    def test_missing_join_key_in_metadata_raises(self):
        reviews = _reviews([{"parent_asin": "P1", "text": "a"}])
        meta = _metadata([{"title": "no key column"}])
        with pytest.raises(KeyError):
            join_reviews_metadata(reviews, meta)

    def test_empty_reviews_frame_yields_empty_merge_and_zero_rate(self):
        reviews = _reviews([])
        reviews["parent_asin"] = pd.Series(dtype="object")
        meta = _metadata([{"parent_asin": "P1", "title": "X"}])
        merged, report = join_reviews_metadata(reviews, meta)

        assert len(merged) == 0
        assert report.total_reviews == 0
        assert report.match_rate == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
