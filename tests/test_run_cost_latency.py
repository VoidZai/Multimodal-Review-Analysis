"""Unit tests for cragb.eval.run_cost_latency (T5.5; M5.md T5.5).

`stratified_question_sample`, `run_one_closed_book_question`,
`run_one_grounded_question`, and `summarize_e2e_latency` are all pure/injected-dependency
functions and are unit-tested directly. `run_e2e_latency`/`main` wire together this
project's real `configs/*.yaml` files and construct real `GroqClient`s (mirroring
`cragb.generate.grounded_qa`'s own `main`, which this project's test suite does not unit
test either) — exercising them means either hitting the real disk cache/network or
extensively monkeypatching config loading, neither of which tests real logic beyond what
`_client_and_template` already covers via its callers below.

Covers M5.md T5.5's own validation checks: every row's `cached` reflects what the
`usage_fn` actually returned (a fake always returning `cached=False` proves the pipeline
propagates it rather than hard-coding it); `e2e_ms >= retrieval_ms + generate_ms`;
`p95 >= p50` in the summary (via `latency_stats`, already covered by `test_timing.py`,
re-verified here at this module's call site).
"""

from __future__ import annotations

from string import Template

import numpy as np
import pandas as pd
import pytest

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.run_cost_latency import (
    run_one_closed_book_question,
    run_one_grounded_question,
    stratified_question_sample,
    summarize_e2e_latency,
)
from cragb.generate.api_clients import CompletionResult
from cragb.generate.context_builder import build_corpus_lookup, index_bm25_retriever
from cragb.retrieval.chunking import ChunkingConfig


def make_question(id_: str, type_: str, question: str = "does this run small?") -> RetrievalQuestion:
    return RetrievalQuestion(
        id=id_, type=type_, question=question, is_negative=False, relevant_ids=frozenset()
    )


def make_pool(counts: dict[str, int]) -> list[RetrievalQuestion]:
    pool = []
    for type_, n in counts.items():
        for i in range(n):
            pool.append(make_question(f"{type_}_{i}", type_))
    return pool


class TestStratifiedQuestionSample:
    def test_rejects_non_positive_n(self):
        pool = make_pool({"a": 5, "b": 5})
        with pytest.raises(ValueError, match="n must be positive"):
            stratified_question_sample(pool, 0, np.random.default_rng(0))

    def test_rejects_n_exceeding_pool(self):
        pool = make_pool({"a": 5, "b": 5})
        with pytest.raises(ValueError, match="exceeds"):
            stratified_question_sample(pool, 11, np.random.default_rng(0))

    def test_returns_exactly_n_questions(self):
        # Matches CRAGB v1's real distribution (60 total, 7 types).
        pool = make_pool(
            {"fit_sizing": 12, "fabric_quality": 10, "colour_appearance": 8, "durability": 8, "defects": 8, "occasion": 8, "value": 6}
        )
        sample = stratified_question_sample(pool, 15, np.random.default_rng(0))
        assert len(sample) == 15

    def test_returned_questions_are_unique_and_from_the_pool(self):
        pool = make_pool({"a": 12, "b": 10, "c": 8})
        sample = stratified_question_sample(pool, 15, np.random.default_rng(0))
        ids = [q.id for q in sample]
        assert len(set(ids)) == len(ids)
        assert set(ids) <= {q.id for q in pool}

    def test_per_type_quota_is_independent_of_seed(self):
        pool = make_pool(
            {"fit_sizing": 12, "fabric_quality": 10, "colour_appearance": 8, "durability": 8, "defects": 8, "occasion": 8, "value": 6}
        )
        sample_a = stratified_question_sample(pool, 15, np.random.default_rng(1))
        sample_b = stratified_question_sample(pool, 15, np.random.default_rng(999))

        def type_counts(sample):
            counts: dict[str, int] = {}
            for q in sample:
                counts[q.type] = counts.get(q.type, 0) + 1
            return counts

        assert type_counts(sample_a) == type_counts(sample_b)
        # But different seeds draw different specific questions within a type.
        assert {q.id for q in sample_a} != {q.id for q in sample_b}

    def test_different_seeds_are_reproducible(self):
        pool = make_pool({"a": 12, "b": 10, "c": 8})
        sample_1 = stratified_question_sample(pool, 15, np.random.default_rng(42))
        sample_2 = stratified_question_sample(pool, 15, np.random.default_rng(42))
        assert [q.id for q in sample_1] == [q.id for q in sample_2]

    def test_n_equal_pool_size_returns_everything(self):
        pool = make_pool({"a": 3, "b": 4})
        sample = stratified_question_sample(pool, 7, np.random.default_rng(0))
        assert {q.id for q in sample} == {q.id for q in pool}


CLOSED_BOOK_TEMPLATE = Template("Q: $question (no context)")
GROUNDED_TEMPLATE = Template("Q: $question\nContext:\n$context_block")


def make_fake_usage_fn(text: str = "an answer", model: str = "openai/gpt-oss-20b", cached: bool = False):
    def usage_fn(messages):
        return CompletionResult(
            text=text, prompt_tokens=10, completion_tokens=3, latency_s=0.01, cached=cached, model=model
        )

    return usage_fn


class TestRunOneClosedBookQuestion:
    def test_row_shape_and_zero_retrieval_ms(self):
        question = make_question("q0", "fit_sizing")
        row = run_one_closed_book_question(question, CLOSED_BOOK_TEMPLATE, make_fake_usage_fn())

        assert row.arm == "closed_book"
        assert row.question_id == "q0"
        assert row.model == "openai/gpt-oss-20b"
        assert row.retrieval_ms == 0.0
        assert row.generate_ms >= 0
        assert row.e2e_ms >= row.generate_ms
        assert row.cached is False

    def test_cached_flag_is_propagated_from_usage_fn(self):
        question = make_question("q0", "fit_sizing")
        row = run_one_closed_book_question(question, CLOSED_BOOK_TEMPLATE, make_fake_usage_fn(cached=True))
        assert row.cached is True

    def test_to_dict_has_expected_keys(self):
        question = make_question("q0", "fit_sizing")
        row = run_one_closed_book_question(question, CLOSED_BOOK_TEMPLATE, make_fake_usage_fn())
        assert set(row.to_dict()) == {
            "arm",
            "question_id",
            "model",
            "retrieval_ms",
            "generate_ms",
            "e2e_ms",
            "cached",
            "run_timestamp",
        }


def make_indexed_context(corpus: pd.DataFrame):
    retriever, chunk_to_parent = index_bm25_retriever(corpus, ChunkingConfig(scheme="whole_review"))
    lookup = build_corpus_lookup(corpus, image_flag_col="has_image")
    return retriever, chunk_to_parent, lookup


def make_synthetic_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "This runs small, definitely size up.",
                "Great fabric, held up after washes.",
                "The colour matched the listing exactly.",
                "Comfortable shoes, true to size.",
            ],
            "has_image": [False, True, False, False],
        }
    )


class TestRunOneGroundedQuestion:
    def test_e2e_ms_is_at_least_retrieval_plus_generate(self):
        corpus = make_synthetic_corpus()
        retriever, chunk_to_parent, lookup = make_indexed_context(corpus)
        question = make_question("q0", "fit_sizing", question="does this run small")

        row = run_one_grounded_question(
            "rag_small", question, GROUNDED_TEMPLATE, retriever, chunk_to_parent, lookup, k=2,
            usage_fn=make_fake_usage_fn(),
        )

        assert row.e2e_ms >= row.retrieval_ms + row.generate_ms
        assert row.retrieval_ms > 0  # a real BM25 search, not the closed-book zero
        assert row.arm == "rag_small"
        assert row.question_id == "q0"

    def test_arm_label_is_carried_through(self):
        corpus = make_synthetic_corpus()
        retriever, chunk_to_parent, lookup = make_indexed_context(corpus)
        question = make_question("q0", "fit_sizing", question="does this run small")

        row = run_one_grounded_question(
            "rag_large", question, GROUNDED_TEMPLATE, retriever, chunk_to_parent, lookup, k=2,
            usage_fn=make_fake_usage_fn(model="openai/gpt-oss-120b"),
        )
        assert row.arm == "rag_large"
        assert row.model == "openai/gpt-oss-120b"

    def test_cached_flag_is_propagated_from_usage_fn(self):
        corpus = make_synthetic_corpus()
        retriever, chunk_to_parent, lookup = make_indexed_context(corpus)
        question = make_question("q0", "fit_sizing", question="does this run small")

        row = run_one_grounded_question(
            "rag_small", question, GROUNDED_TEMPLATE, retriever, chunk_to_parent, lookup, k=2,
            usage_fn=make_fake_usage_fn(cached=True),
        )
        assert row.cached is True


def make_per_question_rows() -> pd.DataFrame:
    # 5 questions per arm, deliberately non-uniform ms so p50 != p95 != mean.
    rows = []
    for arm, model, base_ms in [
        ("closed_book", "openai/gpt-oss-20b", 100),
        ("rag_small", "openai/gpt-oss-20b", 800),
        ("rag_large", "openai/gpt-oss-120b", 1500),
    ]:
        for i in range(5):
            ms = base_ms + i * 50
            rows.append(
                {
                    "arm": arm,
                    "question_id": f"{arm}_{i}",
                    "model": model,
                    "retrieval_ms": 0.0 if arm == "closed_book" else 20.0 + i,
                    "generate_ms": float(ms),
                    "e2e_ms": float(ms) + (0.0 if arm == "closed_book" else 20.0 + i) + 1.0,
                    "cached": False,
                    "run_timestamp": "2026-08-23T12:00:00+00:00",
                }
            )
    return pd.DataFrame(rows)


class TestSummarizeE2ELatency:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            summarize_e2e_latency(pd.DataFrame(columns=["arm"]))

    def test_one_row_per_arm(self):
        summary = summarize_e2e_latency(make_per_question_rows())
        assert set(summary["arm"]) == {"closed_book", "rag_small", "rag_large"}
        assert len(summary) == 3

    def test_p95_is_at_least_p50(self):
        summary = summarize_e2e_latency(make_per_question_rows())
        assert (summary["e2e_ms_p95"] >= summary["e2e_ms_p50"]).all()

    def test_cache_bypassed_true_when_every_row_is_uncached(self):
        summary = summarize_e2e_latency(make_per_question_rows())
        assert summary["cache_bypassed"].all()

    def test_cache_bypassed_false_if_any_row_was_a_cache_hit(self):
        rows = make_per_question_rows()
        rows.loc[rows["arm"] == "rag_small", "cached"] = [False, False, True, False, False]
        summary = summarize_e2e_latency(rows).set_index("arm")
        assert summary.loc["rag_small", "cache_bypassed"] == False  # noqa: E712
        assert summary.loc["closed_book", "cache_bypassed"] == True  # noqa: E712

    def test_closed_book_retrieval_p50_is_zero(self):
        summary = summarize_e2e_latency(make_per_question_rows()).set_index("arm")
        assert summary.loc["closed_book", "retrieval_ms_p50"] == 0.0

    def test_n_matches_row_count_per_arm(self):
        summary = summarize_e2e_latency(make_per_question_rows()).set_index("arm")
        assert summary.loc["rag_small", "n"] == 5
