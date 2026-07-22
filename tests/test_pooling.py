"""Unit tests for cragb.bench.pooling.

`pool_question`/`pool_all_questions` only depend on the `Retriever`
interface (T2.4), not any concrete implementation, so these tests use a
lightweight `FakeRetriever` stub throughout — no BM25 index, no
sentence-transformers model, no network. This keeps the whole file fast
and dependency-free, matching `cragb.bench.pooling`'s own design goal of
staying import-time-independent of the heavy dense-retrieval stack.

Covers: question loading, union-of-top-k pooling (including that only
`max(k_values)` is actually queried), best-rank sorting, retriever-
overlap counting, the zero-pool-vs-low-diversity distinction in
`summarize_pools` (and that negatives are excluded from the low-
diversity flag), and JSONL round-tripping.
"""

from __future__ import annotations

import json

import pytest

from cragb.bench.pooling import (
    PoolHit,
    QuestionPool,
    QuestionRecord,
    load_questions,
    pool_all_questions,
    pool_question,
    summarize_pools,
    write_pools_jsonl,
)
from cragb.retrieval.base import Retriever, SearchResult


class FakeRetriever(Retriever):
    """Returns a fixed, pre-scripted result list for any query, truncated to k.

    Also records every `(query, k)` call it receives, so tests can
    assert *how* pooling queried it (e.g. only at k_max, not at every
    configured k).
    """

    def __init__(self, results: list[SearchResult]):
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def index(self, corpus, text_col="text", id_col=None) -> None:  # pragma: no cover - unused in these tests
        pass

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.calls.append((query, k))
        return self._results[:k]


def make_question(id_="q1", category="fit_sizing", question="Does this run small?", is_negative=False) -> QuestionRecord:
    return QuestionRecord(id=id_, category=category, question=question, is_negative=is_negative)


class TestLoadQuestions:
    def test_round_trips_jsonl_ignoring_image_target(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        path.write_text(
            '{"id": "fit_sizing_000", "category": "fit_sizing", "question": "Does this run small?", '
            '"is_negative": false, "image_target": true}\n',
            encoding="utf-8",
        )
        questions = load_questions(path)
        assert questions == [QuestionRecord("fit_sizing_000", "fit_sizing", "Does this run small?", False)]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        path.write_text(
            '{"id": "a", "category": "fit_sizing", "question": "Q1?", "is_negative": false, "image_target": false}\n\n'
            '{"id": "b", "category": "fit_sizing", "question": "Q2?", "is_negative": true, "image_target": false}\n',
            encoding="utf-8",
        )
        questions = load_questions(path)
        assert len(questions) == 2
        assert questions[1].is_negative is True


class TestPoolQuestion:
    def test_single_retriever_pool_matches_its_results(self):
        results = [SearchResult("d1", 3.0, 1), SearchResult("d2", 2.0, 2)]
        retriever = FakeRetriever(results)
        pool = pool_question(make_question(), {"bm25": retriever}, k_values=[2])

        assert pool.doc_ids == ["d1", "d2"]
        assert pool.pool_size == 2
        assert pool.n_retrievers_used == 1
        assert pool.overlap_count == 0

    def test_union_across_two_retrievers_with_partial_overlap(self):
        bm25 = FakeRetriever([SearchResult("d1", 3.0, 1), SearchResult("shared", 2.0, 2)])
        dense = FakeRetriever([SearchResult("shared", 0.9, 1), SearchResult("d3", 0.8, 2)])

        pool = pool_question(make_question(), {"bm25": bm25, "dense": dense}, k_values=[2])

        assert set(pool.doc_ids) == {"d1", "shared", "d3"}
        assert pool.pool_size == 3
        assert pool.overlap_count == 1  # "shared" found by both
        assert pool.n_retrievers_used == 2

    def test_only_queries_at_k_max_not_every_configured_k(self):
        retriever = FakeRetriever([SearchResult(f"d{i}", float(-i), i) for i in range(1, 11)])
        pool_question(make_question(), {"bm25": retriever}, k_values=[5, 10])

        assert retriever.calls == [("Does this run small?", 10)]

    def test_doc_ids_sorted_by_best_rank_across_retrievers(self):
        bm25 = FakeRetriever([SearchResult("late", 1.0, 5), SearchResult("early", 0.9, 1)])
        dense = FakeRetriever([SearchResult("late", 0.99, 1)])  # "late" is rank 1 via dense

        pool = pool_question(make_question(), {"bm25": bm25, "dense": dense}, k_values=[5])

        # "late" has best_rank=1 (from dense), "early" has best_rank=1 too (from bm25);
        # tie broken alphabetically -> "early" before "late".
        assert pool.doc_ids == ["early", "late"]

    def test_hits_carry_full_provenance(self):
        retriever = FakeRetriever([SearchResult("d1", 3.0, 1)])
        pool = pool_question(make_question(), {"bm25": retriever}, k_values=[1])
        assert pool.hits == [PoolHit(doc_id="d1", retriever="bm25", rank=1, score=3.0)]

    def test_empty_k_values_raises(self):
        with pytest.raises(ValueError, match="k_values must be"):
            pool_question(make_question(), {"bm25": FakeRetriever([])}, k_values=[])

    def test_non_positive_k_values_raises(self):
        with pytest.raises(ValueError, match="k_values must be"):
            pool_question(make_question(), {"bm25": FakeRetriever([])}, k_values=[0])

    def test_carries_question_metadata(self):
        question = make_question(id_="fit_sizing_neg_000", category="fit_sizing", is_negative=True)
        pool = pool_question(question, {"bm25": FakeRetriever([])}, k_values=[5])
        assert pool.question_id == "fit_sizing_neg_000"
        assert pool.category == "fit_sizing"
        assert pool.is_negative is True


class TestPoolAllQuestions:
    def test_pools_every_question_in_order(self):
        questions = [make_question("q1"), make_question("q2"), make_question("q3")]
        retriever = FakeRetriever([SearchResult("d1", 1.0, 1)])
        pools = pool_all_questions(questions, {"bm25": retriever}, k_values=[5])

        assert [p.question_id for p in pools] == ["q1", "q2", "q3"]


def make_pool(question_id, pool_size, is_negative=False) -> QuestionPool:
    doc_ids = [f"d{i}" for i in range(pool_size)]
    hits = [PoolHit(doc_id=d, retriever="bm25", rank=i + 1, score=1.0) for i, d in enumerate(doc_ids)]
    return QuestionPool(
        question_id=question_id,
        category="fit_sizing",
        is_negative=is_negative,
        doc_ids=doc_ids,
        hits=hits,
        pool_size=pool_size,
        n_retrievers_used=1,
        overlap_count=0,
    )


class TestSummarizePools:
    def test_zero_pool_is_a_hard_failure(self):
        pools = [make_pool("q1", pool_size=0), make_pool("q2", pool_size=15)]
        report = summarize_pools(pools, min_pool_size=12)
        assert report.ok is False
        assert report.zero_pools == ("q1",)

    def test_low_diversity_pool_is_a_soft_flag_not_a_failure(self):
        pools = [make_pool("q1", pool_size=10)]  # <= min_pool_size, but > 0
        report = summarize_pools(pools, min_pool_size=12)
        assert report.ok is True
        assert report.low_diversity_pools == ("q1",)

    def test_negative_question_excluded_from_low_diversity_flag(self):
        pools = [make_pool("q1_neg", pool_size=1, is_negative=True)]
        report = summarize_pools(pools, min_pool_size=12)
        assert report.ok is True  # pool_size > 0, so not a zero-pool failure either
        assert report.low_diversity_pools == ()

    def test_min_max_mean_computed_correctly(self):
        pools = [make_pool("q1", pool_size=10), make_pool("q2", pool_size=20)]
        report = summarize_pools(pools, min_pool_size=12)
        assert report.min_size == 10
        assert report.max_size == 20
        assert report.mean_size == 15.0

    def test_healthy_pools_produce_a_clean_message(self):
        pools = [make_pool("q1", pool_size=18)]
        report = summarize_pools(pools, min_pool_size=12)
        assert report.ok is True
        assert report.zero_pools == ()
        assert report.low_diversity_pools == ()
        assert "No empty or low-diversity pools." in report.messages


class TestWritePoolsJsonl:
    def test_round_trips_expected_schema(self, tmp_path):
        pool = make_pool("q1", pool_size=2)
        out_path = tmp_path / "pools.jsonl"
        write_pools_jsonl([pool], out_path)

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["question_id"] == "q1"
        assert row["pool_size"] == 2
        assert row["doc_ids"] == ["d0", "d1"]
        assert row["n_retrievers_used"] == 1
        assert row["overlap_count"] == 0
        assert len(row["hits"]) == 2
        assert row["hits"][0] == {"doc_id": "d0", "retriever": "bm25", "rank": 1, "score": 1.0}


class TestPoolingEndToEnd:
    """Exercises pool_all_questions + summarize_pools together, simulating
    two retrievers of differing quality (as BM25 vs dense would be)."""

    def test_realistic_two_retriever_run(self):
        questions = [
            make_question("fit_sizing_000", is_negative=False),
            make_question("fit_sizing_neg_000", is_negative=True),
        ]
        # bm25: lexical match finds 10 docs for the real question, nothing for the negative
        bm25 = FakeRetriever([SearchResult(f"bm25_{i}", float(10 - i), i) for i in range(1, 11)])
        # dense: semantic match finds a partially overlapping set for the real question,
        # and — realistically — still returns *something* for the "negative" (top-k
        # always returns k docs; none of them need to actually be relevant)
        dense = FakeRetriever(
            [SearchResult("bm25_1", 0.95, 1)] + [SearchResult(f"dense_{i}", 0.9 - i * 0.01, i + 1) for i in range(1, 10)]
        )

        retrievers = {"bm25": bm25, "dense": dense}
        pools = pool_all_questions(questions, retrievers, k_values=[5, 10])
        report = summarize_pools(pools, min_pool_size=12)

        # FakeRetriever ignores query text (fixed script), so both questions
        # get the same union here — the point is exercising the full
        # pool_all_questions -> summarize_pools pipeline, not query-dependent
        # results (that's covered by the real-corpus tests in test_retrieval.py).
        for question_id in ("fit_sizing_000", "fit_sizing_neg_000"):
            pool = next(p for p in pools if p.question_id == question_id)
            assert pool.pool_size == 19  # 10 + 10 - 1 shared ("bm25_1")
            assert pool.overlap_count == 1

        assert report.ok is True
        assert report.zero_pools == ()
