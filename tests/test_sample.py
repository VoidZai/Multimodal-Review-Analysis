"""Unit tests for cragb.data.sample.

Covers: rating bucketing edge cases, that joint target allocation sums
exactly to target_size and matches configured proportions, that
stratified_sample reproduces bit-for-bit under a fixed seed, that it
degrades visibly (not silently) when a stratum is underpopulated, and
that a forced minimum share for a rare stratum (the whole point of T1.7,
per PLAN.md Risk C) actually holds on real proportions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cragb.data.sample import (
    SamplingReport,
    add_rating_bucket,
    bucket_rating,
    compute_joint_targets,
    stratified_sample,
)


class TestBucketRating:
    @pytest.mark.parametrize(
        "rating,expected",
        [(1.0, "1-2"), (2.0, "1-2"), (3.0, "3"), (4.0, "4-5"), (5.0, "4-5")],
    )
    def test_known_ratings_map_correctly(self, rating, expected):
        assert bucket_rating(rating) == expected

    def test_integer_input_is_coerced(self):
        assert bucket_rating(4) == "4-5"

    def test_out_of_range_returns_none(self):
        assert bucket_rating(0.0) is None
        assert bucket_rating(6.0) is None

    def test_nan_returns_none(self):
        assert bucket_rating(float("nan")) is None

    def test_non_numeric_returns_none(self):
        assert bucket_rating("five stars") is None
        assert bucket_rating(None) is None


class TestAddRatingBucket:
    def test_adds_expected_column(self):
        df = pd.DataFrame({"rating": [1.0, 3.0, 5.0]})
        out = add_rating_bucket(df)
        assert list(out["rating_bucket"]) == ["1-2", "3", "4-5"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"rating": [1.0]})
        add_rating_bucket(df)
        assert "rating_bucket" not in df.columns


class TestComputeJointTargets:
    def test_sums_exactly_to_target_size(self):
        weights = {
            "rating": {"1-2": 0.2, "3": 0.15, "4-5": 0.65},
            "has_image": {True: 0.15, False: 0.85},
        }
        targets = compute_joint_targets(weights, target_size=1000)
        assert sum(targets.values()) == 1000

    def test_sums_exactly_for_an_awkward_target_size(self):
        # 137 forces non-trivial rounding remainders across 6 cells.
        weights = {
            "rating": {"1-2": 0.2, "3": 0.15, "4-5": 0.65},
            "has_image": {True: 0.15, False: 0.85},
        }
        targets = compute_joint_targets(weights, target_size=137)
        assert sum(targets.values()) == 137

    def test_joint_weight_is_product_of_marginals(self):
        weights = {"rating": {"a": 0.5, "b": 0.5}, "has_image": {True: 0.5, False: 0.5}}
        targets = compute_joint_targets(weights, target_size=10_000)
        # Each of the 4 equally-likely cells should get ~2500 (0.5*0.5*10000).
        for count in targets.values():
            assert 2400 <= count <= 2600

    def test_rejects_marginals_not_summing_to_one(self):
        weights = {"rating": {"1-2": 0.2, "3": 0.15, "4-5": 0.5}, "has_image": {True: 0.15, False: 0.85}}
        with pytest.raises(ValueError):
            compute_joint_targets(weights, target_size=1000)

    def test_single_dimension_works(self):
        weights = {"has_image": {True: 0.15, False: 0.85}}
        targets = compute_joint_targets(weights, target_size=1000)
        assert sum(targets.values()) == 1000
        assert targets[(True,)] == 150
        assert targets[(False,)] == 850


def _make_synthetic_reviews(n_per_cell: dict[tuple[str, bool], int]) -> pd.DataFrame:
    """Build a synthetic reviews frame with an exact, known population per
    (rating_bucket, has_image) cell, so expected sample proportions are
    known exactly rather than approximately."""
    rows = []
    for (bucket, has_img), n in n_per_cell.items():
        rating = {"1-2": 1.0, "3": 3.0, "4-5": 5.0}[bucket]
        for i in range(n):
            rows.append({"rating": rating, "has_image": has_img, "text": f"{bucket}-{has_img}-{i}"})
    return pd.DataFrame(rows)


WEIGHTS = {
    "rating_bucket": {"1-2": 0.2, "3": 0.15, "4-5": 0.65},
    "has_image": {True: 0.15, False: 0.85},
}


class TestStratifiedSample:
    def test_forced_minimum_share_holds_when_population_allows(self):
        # A plentiful, evenly-populated source pool: the rare "1-2 star with
        # image" combination is intentionally *not* rare in the source here,
        # so we can check the sampler actually hits the configured 20%*15%
        # share for it rather than reproducing the source's own imbalance.
        population = _make_synthetic_reviews(
            {(b, img): 5000 for b in ("1-2", "3", "4-5") for img in (True, False)}
        )
        df = add_rating_bucket(population)

        sampled, report = stratified_sample(df, WEIGHTS, target_size=3000, seed=42)

        assert report.actual_size == 3000
        assert report.shortfall_total == 0

        share_1_2_image = ((sampled["rating_bucket"] == "1-2") & (sampled["has_image"])).mean()
        expected_share = 0.2 * 0.15  # 3%
        assert abs(share_1_2_image - expected_share) < 0.02

    def test_reproducible_with_same_seed(self):
        population = _make_synthetic_reviews(
            {(b, img): 2000 for b in ("1-2", "3", "4-5") for img in (True, False)}
        )
        df = add_rating_bucket(population)

        sampled1, _ = stratified_sample(df, WEIGHTS, target_size=500, seed=7)
        sampled2, _ = stratified_sample(df, WEIGHTS, target_size=500, seed=7)

        pd.testing.assert_frame_equal(sampled1, sampled2)

    def test_different_seeds_give_different_samples(self):
        population = _make_synthetic_reviews(
            {(b, img): 2000 for b in ("1-2", "3", "4-5") for img in (True, False)}
        )
        df = add_rating_bucket(population)

        sampled1, _ = stratified_sample(df, WEIGHTS, target_size=500, seed=1)
        sampled2, _ = stratified_sample(df, WEIGHTS, target_size=500, seed=2)

        assert not sampled1.index.equals(sampled2.index)

    def test_shortfall_is_reported_not_silently_padded(self):
        # Starve the "1-2 star + has_image" cell deliberately: only 1 row
        # available where the target would demand many more.
        population = _make_synthetic_reviews(
            {
                ("1-2", True): 1,
                ("1-2", False): 5000,
                ("3", True): 5000,
                ("3", False): 5000,
                ("4-5", True): 5000,
                ("4-5", False): 5000,
            }
        )
        df = add_rating_bucket(population)

        sampled, report = stratified_sample(df, WEIGHTS, target_size=3000, seed=42)

        starved_outcome = report.per_stratum[("1-2", True)]
        assert starved_outcome.available == 1
        assert starved_outcome.sampled == 1
        assert starved_outcome.target > 1  # target demanded more than existed
        assert report.shortfall_total == report.target_size - report.actual_size
        assert report.actual_size < report.target_size

    def test_output_row_order_is_shuffled_not_grouped_by_stratum(self):
        population = _make_synthetic_reviews(
            {(b, img): 500 for b in ("1-2", "3", "4-5") for img in (True, False)}
        )
        df = add_rating_bucket(population)
        sampled, _ = stratified_sample(df, WEIGHTS, target_size=1200, seed=42)

        # If strata were laid out in contiguous blocks, the bucket labels
        # of consecutive rows would rarely change; shuffled output should
        # change often.
        bucket_changes = (sampled["rating_bucket"].values[1:] != sampled["rating_bucket"].values[:-1]).sum()
        assert bucket_changes > len(sampled) * 0.3

    def test_rejects_mismatched_strata_weight_keys(self):
        population = _make_synthetic_reviews({("1-2", True): 10})
        df = add_rating_bucket(population)
        bad_weights = {"has_image": {True: 1.0}, "rating_bucket": {"1-2": 1.0}}  # wrong order
        with pytest.raises(ValueError):
            stratified_sample(df, bad_weights, target_size=5, seed=1, strata_cols=("rating_bucket", "has_image"))

    def test_report_serializes_to_plain_dict(self):
        population = _make_synthetic_reviews({("1-2", True): 100, ("3", False): 100})
        df = add_rating_bucket(population)
        _, report = stratified_sample(
            df,
            {"rating_bucket": {"1-2": 0.5, "3": 0.5}, "has_image": {True: 0.5, False: 0.5}},
            target_size=50,
            seed=1,
        )
        d = report.as_dict()
        assert d["target_size"] == 50
        assert isinstance(d["per_stratum"], dict)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
