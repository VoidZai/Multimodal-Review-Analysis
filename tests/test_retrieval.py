"""Unit tests for cragb.retrieval.base and cragb.retrieval.bm25.

Covers: the `Retriever` ABC actually enforces its interface (can't
instantiate directly, incomplete subclasses fail); `tokenize`'s
lowercase/alphanumeric behavior; `BM25Retriever` on a small synthetic
fixture (a known query returns the expected top document, results are
score-sorted, ranks are 1-indexed and sequential); its guard rails
(search-before-index, non-positive k, empty corpus, duplicate ids); and,
if `corpus_v1.parquet` is present locally, a smoke test against a sample
of the real corpus using 3 real CRAGB questions (per M2.md T2.4's own
verification bar) — skipped rather than failed in a fresh checkout,
since `data/processed/` is git-ignored (T1.1's `.gitignore`).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.retrieval.base import Retriever, SearchResult
from cragb.retrieval.bm25 import BM25Retriever, tokenize
from cragb.utils.io import resolve_path

CORPUS_PATH = resolve_path("data/processed/corpus_v1.parquet")
QUESTIONS_PATH = resolve_path("benchmark/cragb_questions_v1.jsonl")


def make_synthetic_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "This shirt runs small, I had to size up.",
                "Completely unrelated review about a kitchen blender.",
                "The blender works great and is very durable.",
                "Sizing is accurate here, true to size for this shirt.",
            ]
        }
    )


class TestRetrieverInterface:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Retriever()

    def test_incomplete_subclass_raises(self):
        class MissingSearch(Retriever):
            def index(self, corpus, text_col="text", id_col=None):
                pass

        with pytest.raises(TypeError):
            MissingSearch()

    def test_search_result_fields(self):
        result = SearchResult(doc_id="42", score=1.5, rank=1)
        assert result.doc_id == "42"
        assert result.score == 1.5
        assert result.rank == 1


class TestTokenize:
    def test_lowercases_and_splits_on_non_alphanumeric(self):
        assert tokenize("Runs Small! Size-Up.") == ["runs", "small", "size", "up"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_numbers_are_kept_as_tokens(self):
        assert tokenize("size 10.5 US") == ["size", "10", "5", "us"]


class TestBM25RetrieverSyntheticFixture:
    def test_known_query_returns_expected_top_document(self):
        corpus = make_synthetic_corpus()
        retriever = BM25Retriever()
        retriever.index(corpus, text_col="text")

        results = retriever.search("does this shirt run small", k=2)
        assert results[0].doc_id == "0"  # the one review mentioning "runs small" + "shirt"
        assert len(results) == 2

    def test_results_sorted_by_score_descending(self):
        corpus = make_synthetic_corpus()
        retriever = BM25Retriever()
        retriever.index(corpus)

        results = retriever.search("blender durable", k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_1_indexed_and_sequential(self):
        corpus = make_synthetic_corpus()
        retriever = BM25Retriever()
        retriever.index(corpus)

        results = retriever.search("shirt", k=3)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_k_larger_than_corpus_returns_all_documents(self):
        corpus = make_synthetic_corpus()
        retriever = BM25Retriever()
        retriever.index(corpus)

        results = retriever.search("shirt", k=100)
        assert len(results) == len(corpus)

    def test_custom_id_col_used_as_doc_id(self):
        corpus = pd.DataFrame({"text": ["shirt runs small"], "docid": ["review_42"]})
        retriever = BM25Retriever()
        retriever.index(corpus, text_col="text", id_col="docid")

        results = retriever.search("shirt", k=1)
        assert results[0].doc_id == "review_42"

    def test_default_id_uses_dataframe_index(self):
        corpus = make_synthetic_corpus().set_index(pd.Index(["a", "b", "c", "d"]))
        retriever = BM25Retriever()
        retriever.index(corpus)

        results = retriever.search("shirt", k=1)
        assert results[0].doc_id in {"a", "d"}


class TestBM25RetrieverGuardRails:
    def test_search_before_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match=r"before \.index\(\)"):
            retriever.search("shirt", k=3)

    def test_k_zero_raises(self):
        retriever = BM25Retriever()
        retriever.index(make_synthetic_corpus())
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.search("shirt", k=0)

    def test_k_negative_raises(self):
        retriever = BM25Retriever()
        retriever.index(make_synthetic_corpus())
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.search("shirt", k=-1)

    def test_empty_corpus_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="empty corpus"):
            retriever.index(pd.DataFrame({"text": []}))

    def test_duplicate_ids_raise(self):
        corpus = pd.DataFrame({"text": ["a review", "another review"], "docid": ["x", "x"]})
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="not unique"):
            retriever.index(corpus, text_col="text", id_col="docid")

    def test_missing_text_treated_as_empty_string(self):
        corpus = pd.DataFrame({"text": ["shirt runs small", None]})
        retriever = BM25Retriever()
        retriever.index(corpus)  # must not raise on the None row
        results = retriever.search("shirt", k=2)
        assert len(results) == 2


@pytest.mark.skipif(
    not CORPUS_PATH.is_file(),
    reason="data/processed/corpus_v1.parquet not present locally (git-ignored; run T1.8's build first)",
)
class TestBM25RetrieverAgainstRealCorpus:
    """Smoke test against a sample of the real corpus, per M2.md T2.4's own bar."""

    @staticmethod
    @pytest.fixture(scope="class")
    def indexed_retriever() -> BM25Retriever:
        corpus = pd.read_parquet(CORPUS_PATH)
        # A 5,000-row sample keeps this test fast; indexing the full
        # 200k-row corpus_v1 takes ~5s (measured), which is fine for a
        # one-off run but not for every `pytest` invocation.
        sample = corpus.sample(n=min(5000, len(corpus)), random_state=42)
        retriever = BM25Retriever()
        retriever.index(sample, text_col="text")
        return retriever

    def _real_questions(self, n: int = 3) -> list[str]:
        if not QUESTIONS_PATH.is_file():
            pytest.skip("benchmark/cragb_questions_v1.jsonl not present locally")
        lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()[:n]
        return [json.loads(line)["question"] for line in lines]

    def test_real_questions_return_nonempty_sensibly_ranked_results(self, indexed_retriever):
        for question in self._real_questions():
            results = indexed_retriever.search(question, k=5)
            assert len(results) > 0
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
            assert [r.rank for r in results] == list(range(1, len(results) + 1))
