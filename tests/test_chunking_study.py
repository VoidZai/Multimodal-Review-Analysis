"""Unit tests for cragb.eval.chunking_study (T3.4; M3.md T3.4).

`collapse_chunk_ranking_to_parents` is tested directly against a hand-
built chunk->parent mapping (the trickiest, most bug-prone piece of the
chunking study — a wrong collapse would silently structurally penalize
multi-chunk schemes, exactly the failure mode PLAN.md's "wrong Recall
implementation invalidates everything" warning is about). The rest is
exercised end-to-end on a tiny synthetic corpus with a real
`BM25Retriever` — small enough to be fast and deterministic, but real
enough to catch integration bugs a mocked retriever would hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cragb.eval.chunking_study import (
    collapse_chunk_ranking_to_parents,
    run_chunking_study,
    run_scheme_recall,
    summarize_recall,
)
from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.retrieval.chunking import ChunkingConfig


class TestCollapseChunkRankingToParents:
    def test_dedups_keeping_first_occurrence_rank(self):
        mapping = {"c1": "p1", "c2": "p1", "c3": "p2", "c4": "p3"}
        ranked = ["c1", "c3", "c2", "c4"]
        assert collapse_chunk_ranking_to_parents(ranked, mapping) == ["p1", "p2", "p3"]

    def test_single_chunk_per_parent_is_a_pure_passthrough(self):
        mapping = {"p1": "p1", "p2": "p2", "p3": "p3"}
        ranked = ["p2", "p1", "p3"]
        assert collapse_chunk_ranking_to_parents(ranked, mapping) == ["p2", "p1", "p3"]

    def test_empty_ranking_returns_empty_list(self):
        assert collapse_chunk_ranking_to_parents([], {"c1": "p1"}) == []

    def test_unknown_chunk_id_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            collapse_chunk_ranking_to_parents(["c_missing"], {"c1": "p1"})


def make_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "This runs small, definitely size up before you order.",
                "Great fabric, held up after a dozen washes with no pilling.",
                "The colour matched the listing photo exactly, very happy.",
            ]
        }
    )


def make_questions() -> list[RetrievalQuestion]:
    return [
        RetrievalQuestion(
            id="fit_q",
            type="fit_sizing",
            question="does this run small",
            is_negative=False,
            relevant_ids=frozenset({"0"}),
        ),
        RetrievalQuestion(
            id="fabric_q",
            type="fabric_quality",
            question="does the fabric hold up after washing",
            is_negative=False,
            relevant_ids=frozenset({"1"}),
        ),
    ]


class TestRunSchemeRecall:
    def test_whole_review_finds_the_obviously_relevant_doc_at_k1(self):
        result = run_scheme_recall(
            make_corpus(),
            ChunkingConfig(scheme="whole_review"),
            make_questions(),
            k_values=[1],
            show_progress=False,
        )
        fit_row = result.loc[result["question_id"] == "fit_q"].iloc[0]
        assert fit_row["recall"] == 1.0

    def test_output_has_one_row_per_question_per_k(self):
        result = run_scheme_recall(
            make_corpus(),
            ChunkingConfig(scheme="whole_review"),
            make_questions(),
            k_values=[1, 2],
            show_progress=False,
        )
        assert len(result) == len(make_questions()) * 2
        assert set(result.columns) == {"question_id", "type", "k", "recall"}

    def test_recall_scores_are_in_valid_range(self):
        result = run_scheme_recall(
            make_corpus(),
            ChunkingConfig(scheme="whole_review"),
            make_questions(),
            k_values=[1, 2, 3],
            show_progress=False,
        )
        assert result["recall"].between(0.0, 1.0).all()

    def test_extreme_fixed_token_chunking_still_recovers_the_right_parent(self):
        # token_size=1 shatters every review into single-word chunks;
        # collapse_chunk_ranking_to_parents + a generous chunk_search_multiplier
        # must still surface the correct parent review at k=1.
        result = run_scheme_recall(
            make_corpus(),
            ChunkingConfig(scheme="fixed_token", fixed_token_size=1),
            make_questions(),
            k_values=[1],
            chunk_search_multiplier=20,
            show_progress=False,
        )
        fit_row = result.loc[result["question_id"] == "fit_q"].iloc[0]
        assert fit_row["recall"] == 1.0


class TestSummarizeRecall:
    def test_produces_expected_columns_and_one_row_per_k(self):
        per_question = pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q1", "q2"],
                "type": ["fit_sizing"] * 4,
                "k": [1, 1, 3, 3],
                "recall": [1.0, 0.0, 1.0, 0.5],
            }
        )
        summary = summarize_recall(per_question, scheme="whole_review", n_boot=500, rng=np.random.default_rng(0))
        assert list(summary.columns) == [
            "scheme", "k", "recall_mean", "recall_ci_lo", "recall_ci_hi", "n_questions",
        ]
        assert set(summary["k"]) == {1, 3}
        assert (summary["scheme"] == "whole_review").all()

    def test_recall_mean_matches_manual_average(self):
        per_question = pd.DataFrame(
            {"question_id": ["q1", "q2"], "type": ["value"] * 2, "k": [1, 1], "recall": [1.0, 0.0]}
        )
        summary = summarize_recall(per_question, scheme="x", n_boot=500, rng=np.random.default_rng(0))
        assert summary.loc[0, "recall_mean"] == pytest.approx(0.5)
        assert summary.loc[0, "n_questions"] == 2


class TestRunChunkingStudy:
    def test_returns_summary_and_per_question_frames(self):
        summary, per_question = run_chunking_study(
            make_corpus(),
            {"whole_review": ChunkingConfig(scheme="whole_review")},
            make_questions(),
            k_values=[1, 2],
            n_boot=500,
            rng=np.random.default_rng(0),
            show_progress=False,
        )
        assert set(summary.columns) == {
            "scheme", "k", "recall_mean", "recall_ci_lo", "recall_ci_hi", "n_questions",
        }
        assert set(per_question.columns) == {"scheme", "question_id", "type", "k", "recall"}

    def test_scheme_label_is_the_dict_key_not_config_scheme(self):
        # Two labels both use scheme="fixed_token" at different sizes;
        # the output must distinguish them by label, not collapse them.
        summary, _ = run_chunking_study(
            make_corpus(),
            {
                "fixed_token_1": ChunkingConfig(scheme="fixed_token", fixed_token_size=1),
                "fixed_token_2": ChunkingConfig(scheme="fixed_token", fixed_token_size=2),
            },
            make_questions(),
            k_values=[1],
            chunk_search_multiplier=20,
            n_boot=500,
            rng=np.random.default_rng(0),
            show_progress=False,
        )
        assert set(summary["scheme"]) == {"fixed_token_1", "fixed_token_2"}

    def test_reproducible_with_seeded_rng(self):
        kwargs = dict(
            corpus=make_corpus(),
            scheme_configs={"whole_review": ChunkingConfig(scheme="whole_review")},
            questions=make_questions(),
            k_values=[1, 2],
            n_boot=1000,
            show_progress=False,
        )
        summary_1, _ = run_chunking_study(**kwargs, rng=np.random.default_rng(42))
        summary_2, _ = run_chunking_study(**kwargs, rng=np.random.default_rng(42))
        pd.testing.assert_frame_equal(summary_1, summary_2)
