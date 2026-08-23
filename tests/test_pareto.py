"""Unit tests for cragb.eval.pareto (T5.6; M5.md T5.6).

Covers M5.md T5.6's own validation checks: `pareto_frontier` on a hand-built 4-point
case including a tie; every configuration in the quality tables appears exactly once in
the joined table (no silent inner-join row loss, verified by row-count assertions before
and after `join_quality_cost`, plus tests that it actively raises rather than silently
dropping/duplicating rows on a mismatch).
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.eval.pareto import (
    join_quality_cost,
    pareto_frontier,
    prepare_quality_df,
    retrieval_quality_cost_rows,
)


class TestParetoFrontier:
    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="non-empty"):
            pareto_frontier(pd.DataFrame(columns=["quality", "cost"]), "quality", "cost")

    def test_single_row_is_always_on_the_frontier(self):
        df = pd.DataFrame({"quality": [3.0], "cost": [1.0]})
        result = pareto_frontier(df, "quality", "cost")
        assert result.tolist() == [True]

    def test_hand_built_four_point_case_with_a_tie(self):
        # A: cheap, low quality -- non-dominated (cheapest).
        # B: expensive, highest quality -- non-dominated (best quality).
        # C: same cost as B, strictly worse quality -- dominated by B.
        # D: same quality as A, more expensive -- dominated by A.
        df = pd.DataFrame(
            {
                "config": ["A", "B", "C", "D"],
                "quality": [3.0, 4.0, 3.5, 3.0],
                "cost": [1.0, 5.0, 5.0, 2.0],
            }
        )
        result = pareto_frontier(df, "quality", "cost")
        assert result.tolist() == [True, True, False, False]

    def test_exact_ties_are_both_kept(self):
        df = pd.DataFrame({"quality": [3.0, 3.0], "cost": [1.0, 1.0]})
        result = pareto_frontier(df, "quality", "cost")
        assert result.tolist() == [True, True]

    def test_result_is_aligned_to_a_non_default_index(self):
        df = pd.DataFrame({"quality": [3.0, 4.0], "cost": [1.0, 2.0]}, index=[10, 20])
        result = pareto_frontier(df, "quality", "cost")
        assert list(result.index) == [10, 20]

    def test_strictly_dominated_point_is_excluded(self):
        # Worse quality AND higher cost than another point -- clearly dominated.
        df = pd.DataFrame({"quality": [3.0, 4.0], "cost": [5.0, 1.0]})
        result = pareto_frontier(df, "quality", "cost")
        assert result.tolist() == [False, True]


def make_rq_table(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "arm", "mean", "ci_lo", "ci_hi"]).assign(n=60, wilcoxon_p=0.1)


class TestPrepareQualityDf:
    def test_raises_if_metric_missing_from_both_tables(self):
        rq0 = make_rq_table([("correctness", "closed_book", 1.5, 1.2, 1.8)])
        rq1 = make_rq_table([("correctness", "rag_small", 3.6, 3.3, 4.0)])
        with pytest.raises(ValueError, match="not found"):
            prepare_quality_df(rq0, rq1, "completeness")

    def test_one_row_per_arm_across_both_tables(self):
        rq0 = make_rq_table(
            [
                ("correctness", "closed_book", 1.5, 1.2, 1.8),
                ("correctness", "rag_small", 3.6, 3.3, 4.0),
            ]
        )
        rq1 = make_rq_table(
            [
                ("correctness", "rag_small", 3.67, 3.3, 4.02),
                ("correctness", "rag_large", 3.77, 3.4, 4.1),
            ]
        )
        result = prepare_quality_df(rq0, rq1, "correctness")
        assert set(result["arm"]) == {"closed_book", "rag_small", "rag_large"}
        assert len(result) == 3

    def test_rag_small_duplicate_prefers_rq1_version(self):
        rq0 = make_rq_table([("correctness", "rag_small", 111.0, 1.0, 1.0)])
        rq1 = make_rq_table([("correctness", "rag_small", 222.0, 2.0, 2.0)])
        result = prepare_quality_df(rq0, rq1, "correctness").set_index("arm")
        assert result.loc["rag_small", "mean"] == 222.0

    def test_filters_to_requested_metric_only(self):
        rq0 = make_rq_table(
            [
                ("correctness", "closed_book", 1.5, 1.2, 1.8),
                ("faithfulness", "closed_book", 4.5, 4.0, 4.9),
            ]
        )
        rq1 = make_rq_table([("correctness", "rag_large", 3.77, 3.4, 4.1)])
        result = prepare_quality_df(rq0, rq1, "faithfulness")
        assert list(result["arm"]) == ["closed_book"]


def make_quality_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["closed_book", "rag_small", "rag_large"],
            "n": [60, 60, 60],
            "mean": [1.52, 3.67, 3.77],
            "ci_lo": [1.23, 3.30, 3.40],
            "ci_hi": [1.85, 4.02, 4.10],
        }
    )


def make_cost_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["closed_book", "rag_small", "rag_large"],
            "model": ["openai/gpt-oss-20b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            "mean_usd_per_query": [0.0000385, 0.0000777, 0.0001537],
            "usd_per_query_ci_lo": [0.0000351, 0.0000743, 0.0001474],
            "usd_per_query_ci_hi": [0.0000428, 0.0000814, 0.0001601],
        }
    )


def make_latency_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["closed_book", "rag_small", "rag_large"],
            "e2e_ms_p50": [3473.9, 10922.9, 9207.3],
            "e2e_ms_p95": [4503.0, 13873.8, 13733.2],
        }
    )


class TestJoinQualityCost:
    def test_one_row_per_arm_in_quality_df(self):
        joined = join_quality_cost(make_quality_df(), make_cost_df(), make_latency_df())
        assert len(joined) == 3
        assert set(joined["arm"]) == {"closed_book", "rag_small", "rag_large"}

    def test_row_count_unchanged_by_the_join(self):
        quality = make_quality_df()
        joined = join_quality_cost(quality, make_cost_df(), make_latency_df())
        assert len(joined) == len(quality)

    def test_expected_columns_present(self):
        joined = join_quality_cost(make_quality_df(), make_cost_df(), make_latency_df())
        assert set(joined.columns) == {
            "arm",
            "model",
            "n",
            "quality_mean",
            "quality_ci_lo",
            "quality_ci_hi",
            "usd_per_query",
            "usd_per_query_ci_lo",
            "usd_per_query_ci_hi",
            "e2e_ms_p50",
            "e2e_ms_p95",
        }

    def test_values_carried_through_correctly(self):
        joined = join_quality_cost(make_quality_df(), make_cost_df(), make_latency_df()).set_index("arm")
        assert joined.loc["rag_large", "quality_mean"] == 3.77
        assert joined.loc["rag_large", "usd_per_query"] == pytest.approx(0.0001537)
        assert joined.loc["rag_large", "e2e_ms_p50"] == pytest.approx(9207.3)

    def test_arm_missing_from_cost_df_raises(self):
        cost_df = make_cost_df()
        cost_df = cost_df[cost_df["arm"] != "rag_large"]
        with pytest.raises(ValueError, match="missing from cost_df"):
            join_quality_cost(make_quality_df(), cost_df, make_latency_df())

    def test_arm_missing_from_latency_df_raises(self):
        latency_df = make_latency_df()
        latency_df = latency_df[latency_df["arm"] != "closed_book"]
        with pytest.raises(ValueError, match="missing from latency_df"):
            join_quality_cost(make_quality_df(), make_cost_df(), latency_df)

    def test_duplicate_arm_in_cost_df_raises_rather_than_fanning_out(self):
        cost_df = pd.concat([make_cost_df(), make_cost_df().iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate arms"):
            join_quality_cost(make_quality_df(), cost_df, make_latency_df())


def make_retrieval_eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "retriever": ["bm25", "dense", "bm25", "dense"],
            "k": [1, 1, 5, 5],
            "n_questions": [58, 58, 58, 58],
            "recall_mean": [0.059, 0.055, 0.50, 0.48],
            "recall_ci_lo": [0.045, 0.042, 0.44, 0.42],
            "recall_ci_hi": [0.078, 0.075, 0.56, 0.54],
        }
    )


def make_retrieval_cost_latency_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "retriever": ["bm25", "dense"],
            "latency_p50_ms": [850.5, 12.3],
        }
    )


class TestRetrievalQualityCostRows:
    def test_raises_on_missing_k(self):
        with pytest.raises(ValueError, match="k=10"):
            retrieval_quality_cost_rows(make_retrieval_eval_df(), make_retrieval_cost_latency_df(), k=10)

    def test_two_rows_at_k5(self):
        result = retrieval_quality_cost_rows(make_retrieval_eval_df(), make_retrieval_cost_latency_df(), k=5)
        assert set(result["retriever"]) == {"bm25", "dense"}
        assert len(result) == 2

    def test_dense_faster_but_lower_recall_neither_dominates(self):
        # bm25: higher recall, higher latency. dense: lower recall, lower latency.
        # Neither dominates the other -- both should be on the frontier.
        result = retrieval_quality_cost_rows(make_retrieval_eval_df(), make_retrieval_cost_latency_df(), k=5)
        assert result["on_frontier"].all()

    def test_missing_retriever_in_cost_latency_raises(self):
        cost_latency = make_retrieval_cost_latency_df()
        cost_latency = cost_latency[cost_latency["retriever"] != "dense"]
        with pytest.raises(ValueError, match="missing from retrieval_cost_latency_df"):
            retrieval_quality_cost_rows(make_retrieval_eval_df(), cost_latency, k=5)

    def test_values_carried_through_correctly(self):
        result = retrieval_quality_cost_rows(make_retrieval_eval_df(), make_retrieval_cost_latency_df(), k=5).set_index(
            "retriever"
        )
        assert result.loc["bm25", "quality_mean"] == pytest.approx(0.50)
        assert result.loc["bm25", "latency_ms_p50"] == pytest.approx(850.5)
