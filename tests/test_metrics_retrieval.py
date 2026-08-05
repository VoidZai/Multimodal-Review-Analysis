"""Unit tests for cragb.eval.metrics_retrieval (T3.2; M3.md T3.2).

Every metric is checked against a hand-computed value first (the whole
point of this task, per PLAN.md, is that a wrong Recall implementation
invalidates every downstream RQ2 number), then against edge cases
(relevant doc absent, empty relevant_ids, k larger than the ranked
list) and the Recall-is-monotone-in-k invariant.
"""

from __future__ import annotations

import math

import pytest

from cragb.eval.metrics_retrieval import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_all,
)


class TestRecallAtK:
    def test_relevant_doc_at_rank_one_is_full_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0

    def test_relevant_doc_outside_top_k_is_zero_recall(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_partial_recall_computed_correctly(self):
        # 1 of 2 relevant docs present in top-3 -> 0.5
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_all_relevant_docs_found_is_full_recall(self):
        assert recall_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0

    def test_k_larger_than_ranked_list_does_not_crash(self):
        assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5

    def test_monotonically_non_decreasing_in_k(self):
        ranked = ["x", "a", "y", "b", "z"]
        relevant = {"a", "b"}
        scores = [recall_at_k(ranked, relevant, k) for k in range(1, 6)]
        assert scores == sorted(scores)

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError, match="empty"):
            recall_at_k(["a", "b"], set(), k=3)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="positive"):
            recall_at_k(["a"], {"a"}, k=k)


class TestHitAtK:
    def test_relevant_doc_present_is_a_hit(self):
        assert hit_at_k(["x", "a", "y"], {"a"}, k=3) == 1.0

    def test_relevant_doc_absent_is_not_a_hit(self):
        assert hit_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_relevant_doc_outside_top_k_is_not_a_hit(self):
        assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_multiple_relevant_docs_still_cap_at_one(self):
        assert hit_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError, match="empty"):
            hit_at_k(["a"], set(), k=1)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="positive"):
            hit_at_k(["a"], {"a"}, k=k)


class TestNdcgAtK:
    def test_relevant_doc_at_rank_one_is_perfect_ndcg(self):
        # single relevant doc, found first -> DCG == IDCG
        assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)

    def test_relevant_doc_absent_is_zero_ndcg(self):
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_hand_computed_two_relevant_one_out_of_order(self):
        # relevant = {a, b}; ranked = [a, x, b]
        # DCG@3   = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@3  = 1/log2(2) + 1/log2(3)      = 1.0 + 0.6309... = 1.6309...
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(expected)

    def test_worse_ranking_scores_lower_than_better_ranking(self):
        relevant = {"a", "b"}
        better = ndcg_at_k(["a", "b", "x"], relevant, k=3)
        worse = ndcg_at_k(["x", "b", "a"], relevant, k=3)
        assert better > worse

    def test_ideal_ranking_scores_exactly_one(self):
        assert ndcg_at_k(["a", "b", "x", "y"], {"a", "b"}, k=4) == pytest.approx(1.0)

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ndcg_at_k(["a"], set(), k=1)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="positive"):
            ndcg_at_k(["a"], {"a"}, k=k)


class TestMrr:
    def test_relevant_doc_at_rank_one_is_reciprocal_one(self):
        assert mrr(["a", "x", "y"], {"a"}, k=3) == 1.0

    def test_relevant_doc_at_rank_three_is_one_third(self):
        assert mrr(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / 3)

    def test_relevant_doc_outside_top_k_is_zero(self):
        assert mrr(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_uses_first_relevant_hit_not_best_possible(self):
        # two relevant docs; first one encountered is at rank 2
        assert mrr(["x", "b", "a"], {"a", "b"}, k=3) == pytest.approx(1 / 2)

    def test_no_relevant_doc_present_is_zero(self):
        assert mrr(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mrr(["a"], set(), k=1)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="positive"):
            mrr(["a"], {"a"}, k=k)


class TestScoreAll:
    def test_returns_one_entry_per_k(self):
        result = score_all(["a", "x", "b"], {"a", "b"}, k_values=[1, 2, 3])
        assert set(result.keys()) == {1, 2, 3}

    def test_each_entry_has_all_four_metrics(self):
        result = score_all(["a", "x", "b"], {"a", "b"}, k_values=[2])
        assert set(result[2].keys()) == {"recall", "hit", "ndcg", "mrr"}

    def test_matches_individually_computed_metrics(self):
        ranked, relevant, k = ["a", "x", "b"], {"a", "b"}, 2
        result = score_all(ranked, relevant, k_values=[k])
        assert result[k]["recall"] == recall_at_k(ranked, relevant, k)
        assert result[k]["hit"] == hit_at_k(ranked, relevant, k)
        assert result[k]["ndcg"] == ndcg_at_k(ranked, relevant, k)
        assert result[k]["mrr"] == mrr(ranked, relevant, k)

    def test_empty_k_values_raises(self):
        with pytest.raises(ValueError, match="k_values"):
            score_all(["a"], {"a"}, k_values=[])

    def test_empty_relevant_ids_raises(self):
        with pytest.raises(ValueError, match="empty"):
            score_all(["a"], set(), k_values=[1])
