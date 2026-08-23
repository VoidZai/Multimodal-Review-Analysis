"""Unit tests for cragb.generate.context_builder (T4a.2; M4a.md T4a.2).

Covers: `build_corpus_lookup`'s id-resolution/validation contract, `build_context`'s
retrieval-to-rendered-text pipeline (rank order, photo flags, truncation, empty/error
cases), and `index_bm25_retriever`'s end-to-end wiring against a real (small, in-memory)
`BM25Retriever` and the locked `whole_review` chunking scheme — no network, no GPU, no
API key required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.generate.context_builder import (
    build_context,
    build_corpus_lookup,
    index_bm25_retriever,
)
from cragb.retrieval.base import Retriever, SearchResult
from cragb.retrieval.chunking import ChunkingConfig


def make_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "These run small, I had to size up two sizes to get a comfortable fit.",
                "Great fabric quality, held up after a dozen washes with no pilling.",
                "Colour matched the product photo exactly, very happy with the purchase.",
                "Runs a little big, order down half a size for a snug fit.",
            ],
            "has_image": [True, False, True, False],
        },
        index=pd.Index(["101", "202", "303", "404"]),
    )


# --------------------------------------------------------------------------
# build_corpus_lookup
# --------------------------------------------------------------------------


class TestBuildCorpusLookup:
    def test_default_id_uses_dataframe_index(self):
        lookup = build_corpus_lookup(make_corpus())
        assert set(lookup.text_by_id) == {"101", "202", "303", "404"}

    def test_text_by_id_matches_source_rows(self):
        lookup = build_corpus_lookup(make_corpus())
        assert lookup.text_by_id["101"].startswith("These run small")

    def test_has_photo_by_id_reflects_image_flag_col(self):
        lookup = build_corpus_lookup(make_corpus())
        assert lookup.has_photo_by_id["101"] is True
        assert lookup.has_photo_by_id["202"] is False

    def test_custom_id_col_used_as_doc_id(self):
        corpus = make_corpus().reset_index(drop=True)
        corpus["review_id"] = ["a", "b", "c", "d"]
        lookup = build_corpus_lookup(corpus, id_col="review_id")
        assert set(lookup.text_by_id) == {"a", "b", "c", "d"}

    def test_missing_has_image_values_default_to_false(self):
        corpus = make_corpus()
        # object dtype (not the column's native bool) so a `None` can be
        # assigned at all — mirrors how a real corpus load could surface
        # a missing flag as null rather than True/False.
        corpus["has_image"] = corpus["has_image"].astype(object)
        corpus.loc["101", "has_image"] = None
        lookup = build_corpus_lookup(corpus)
        assert lookup.has_photo_by_id["101"] is False

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_corpus_lookup(pd.DataFrame({"text": [], "has_image": []}))

    def test_duplicate_ids_raise(self):
        corpus = make_corpus()
        corpus["review_id"] = ["dup", "dup", "x", "y"]
        with pytest.raises(ValueError, match="not unique"):
            build_corpus_lookup(corpus, id_col="review_id")


# --------------------------------------------------------------------------
# build_context
# --------------------------------------------------------------------------


class _StubRetriever(Retriever):
    """A retriever with a fixed, hand-authored ranking — isolates `build_context`'s
    own logic (collapsing, rendering, truncation) from BM25's actual scoring."""

    def __init__(self, ranking: list[SearchResult]) -> None:
        self._ranking = ranking

    def index(self, corpus, text_col="text", id_col=None) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        return self._ranking[:k]

    def index_size_bytes(self) -> int:  # pragma: no cover - unused
        raise NotImplementedError


def _hit(doc_id: str, rank: int, score: float = 1.0) -> SearchResult:
    return SearchResult(doc_id=doc_id, score=score, rank=rank)


class TestBuildContext:
    def test_k_must_be_positive(self):
        lookup = build_corpus_lookup(make_corpus())
        retriever = _StubRetriever([_hit("101", 1)])
        with pytest.raises(ValueError, match="k must be positive"):
            build_context("q", retriever, {"101": "101"}, lookup, k=0)

    def test_doc_ids_in_rank_order(self):
        lookup = build_corpus_lookup(make_corpus())
        ranking = [_hit("303", 1), _hit("101", 2), _hit("202", 3), _hit("404", 4)]
        retriever = _StubRetriever(ranking)
        chunk_to_parent = {d: d for d in ("101", "202", "303", "404")}
        block = build_context("does colour match", retriever, chunk_to_parent, lookup, k=2)
        assert block.doc_ids == ("303", "101")

    def test_multi_chunk_hits_collapse_to_unique_parents(self):
        # Two chunks of review "101" rank ahead of review "202"; build_context
        # must collapse them to one entry for "101", not eat the k budget twice.
        lookup = build_corpus_lookup(make_corpus())
        ranking = [_hit("101::0", 1), _hit("101::1", 2), _hit("202", 3)]
        retriever = _StubRetriever(ranking)
        chunk_to_parent = {"101::0": "101", "101::1": "101", "202": "202"}
        block = build_context("q", retriever, chunk_to_parent, lookup, k=2)
        assert block.doc_ids == ("101", "202")

    def test_text_contains_id_and_photo_flag(self):
        lookup = build_corpus_lookup(make_corpus())
        retriever = _StubRetriever([_hit("101", 1)])
        block = build_context("q", retriever, {"101": "101"}, lookup, k=1)
        assert "[101] has_photo: yes" in block.text
        assert "These run small" in block.text

    def test_photo_flags_match_lookup(self):
        lookup = build_corpus_lookup(make_corpus())
        ranking = [_hit("101", 1), _hit("202", 2)]
        retriever = _StubRetriever(ranking)
        chunk_to_parent = {"101": "101", "202": "202"}
        block = build_context("q", retriever, chunk_to_parent, lookup, k=2)
        assert block.photo_flags == {"101": True, "202": False}

    def test_excerpt_truncated_to_max_chars(self):
        corpus = make_corpus()
        corpus.loc["101", "text"] = "x" * 1000
        lookup = build_corpus_lookup(corpus)
        retriever = _StubRetriever([_hit("101", 1)])
        block = build_context("q", retriever, {"101": "101"}, lookup, k=1, max_excerpt_chars=50)
        assert "x" * 51 not in block.text
        assert "x" * 50 in block.text

    def test_no_hits_returns_placeholder_context(self):
        lookup = build_corpus_lookup(make_corpus())
        retriever = _StubRetriever([])
        block = build_context("q", retriever, {}, lookup, k=3)
        assert block.doc_ids == ()
        assert block.photo_flags == {}
        assert "no review excerpts" in block.text

    def test_doc_id_missing_from_lookup_raises_keyerror(self):
        lookup = build_corpus_lookup(make_corpus())
        retriever = _StubRetriever([_hit("999", 1)])
        with pytest.raises(KeyError):
            build_context("q", retriever, {"999": "999"}, lookup, k=1)


# --------------------------------------------------------------------------
# index_bm25_retriever (integration: real BM25Retriever + whole_review chunking)
# --------------------------------------------------------------------------


class TestIndexBm25Retriever:
    def test_returns_indexed_retriever_and_identity_chunk_map(self):
        corpus = make_corpus()
        config = ChunkingConfig(scheme="whole_review")
        retriever, chunk_to_parent = index_bm25_retriever(corpus, config)
        # whole_review: chunk_id == parent_doc_id, so the map is the identity.
        assert chunk_to_parent == {"101": "101", "202": "202", "303": "303", "404": "404"}
        hits = retriever.search("does this run small", k=2)
        assert len(hits) == 2

    def test_end_to_end_build_context_smoke(self):
        corpus = make_corpus()
        config = ChunkingConfig(scheme="whole_review")
        retriever, chunk_to_parent = index_bm25_retriever(corpus, config)
        lookup = build_corpus_lookup(corpus)
        block = build_context("does this run small or big", retriever, chunk_to_parent, lookup, k=2)
        assert len(block.doc_ids) == 2
        # The two sizing-related reviews (101, 404) should outrank the
        # fabric/colour reviews (202, 303) for this query.
        assert set(block.doc_ids) == {"101", "404"}
