"""Unit tests for cragb.eval.bootstrap (T3.3; M3.md T3.3).

Covers the four validation checks M3.md specifies for this task: CI
width shrinks as sample size grows, CI on identical repeated scores
collapses to a point, a known synthetic A>B case yields p < 0.05 under
`paired_significance`, and a seeded `rng` makes `bootstrap_ci`
reproducible — plus argument validation and the zero-difference edge
case `paired_significance` handles explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cragb.eval.bootstrap import bootstrap_ci, paired_significance


class TestBootstrapCi:
    def test_identical_repeated_scores_collapse_to_a_point(self):
        lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_boot=2000, rng=np.random.default_rng(0))
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_single_score_collapses_to_that_value(self):
        lo, hi = bootstrap_ci([0.7], n_boot=2000, rng=np.random.default_rng(0))
        assert lo == pytest.approx(0.7)
        assert hi == pytest.approx(0.7)

    def test_ci_bounds_the_sample_mean(self):
        scores = [0.1, 0.4, 0.6, 0.9, 0.2, 0.8]
        lo, hi = bootstrap_ci(scores, n_boot=5000, rng=np.random.default_rng(1))
        assert lo <= np.mean(scores) <= hi

    def test_width_shrinks_as_sample_size_grows(self):
        # Same underlying distribution, drawn once at two sample sizes;
        # standard error of the mean shrinks as ~1/sqrt(n), so a larger
        # sample should yield a visibly tighter CI at fixed n_boot/alpha.
        draw_rng = np.random.default_rng(42)
        small_sample = draw_rng.normal(loc=0.6, scale=0.2, size=5).clip(0, 1).tolist()
        large_sample = draw_rng.normal(loc=0.6, scale=0.2, size=200).clip(0, 1).tolist()

        small_lo, small_hi = bootstrap_ci(small_sample, n_boot=5000, rng=np.random.default_rng(2))
        large_lo, large_hi = bootstrap_ci(large_sample, n_boot=5000, rng=np.random.default_rng(3))

        assert (large_hi - large_lo) < (small_hi - small_lo)

    def test_seeded_rng_is_exactly_reproducible(self):
        scores = [0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.55]
        result_1 = bootstrap_ci(scores, n_boot=3000, rng=np.random.default_rng(123))
        result_2 = bootstrap_ci(scores, n_boot=3000, rng=np.random.default_rng(123))
        assert result_1 == result_2

    def test_unseeded_calls_are_not_required_to_match(self):
        scores = [0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.55]
        result_1 = bootstrap_ci(scores, n_boot=3000)
        result_2 = bootstrap_ci(scores, n_boot=3000)
        # Extremely unlikely to coincide by chance with no shared seed;
        # guards against an accidental hidden global-seed dependency.
        assert result_1 != result_2

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_ci([], n_boot=100)

    def test_non_positive_n_boot_raises(self):
        with pytest.raises(ValueError, match="n_boot"):
            bootstrap_ci([0.1, 0.2], n_boot=0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_ci([0.1, 0.2], n_boot=100, alpha=alpha)


class TestPairedSignificance:
    def test_known_synthetic_a_greater_than_b_is_significant(self):
        scores_a = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        scores_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p_value = paired_significance(scores_a, scores_b)
        assert p_value < 0.05

    def test_identical_arrays_return_p_one(self):
        scores = [0.3, 0.5, 0.7, 0.2]
        assert paired_significance(scores, scores) == 1.0

    def test_no_real_difference_is_not_significant(self):
        rng = np.random.default_rng(7)
        base = rng.uniform(0, 1, size=30)
        noise = rng.normal(0, 0.01, size=30)
        scores_a = base.tolist()
        scores_b = (base + noise).tolist()
        p_value = paired_significance(scores_a, scores_b)
        assert p_value > 0.05

    def test_p_value_is_between_zero_and_one(self):
        scores_a = [0.9, 0.1, 0.8, 0.2, 0.95]
        scores_b = [0.1, 0.9, 0.2, 0.8, 0.05]
        p_value = paired_significance(scores_a, scores_b)
        assert 0.0 <= p_value <= 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            paired_significance([0.1, 0.2], [0.1])

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            paired_significance([], [])
