"""BM25 sparse retriever (T2.4; PLAN.md §3 E3).

The industrial sparse baseline named in PLAN.md §3-E3 (TF-IDF is only a
weaker optional reference; BM25 is what's actually compared against
dense retrieval for RQ2). Built on `rank_bm25.BM25Okapi`, the standard
lightweight pure-Python implementation — no external search engine
process, which matters on the 3050 laptop GPU/CPU budget noted in
M2.md's hardware section, since indexing is CPU-only regardless of GPU.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd
from rank_bm25 import BM25Okapi

from cragb.retrieval.base import Retriever, SearchResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokenizer.

    Deliberately simple — no stemming, no stopword removal. BM25's own
    term-frequency/IDF weighting already discounts very common words, and
    a transparent tokenizer keeps the sparse baseline easy to reason
    about and debug, in the spirit of PLAN.md §12 ("simplicity beats
    complexity" wherever the data doesn't demand otherwise).
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Retriever(Retriever):
    """BM25 over whole-review documents, via `Retriever`."""

    _bm25: BM25Okapi | None = field(default=None, init=False, repr=False)
    _doc_ids: list[str] = field(default_factory=list, init=False, repr=False)

    def index(
        self,
        corpus: pd.DataFrame,
        text_col: str = "text",
        id_col: str | None = None,
    ) -> None:
        if corpus.empty:
            raise ValueError("Cannot index an empty corpus.")

        text = corpus[text_col].fillna("").astype(str)
        doc_ids = (
            corpus[id_col].astype(str).tolist()
            if id_col is not None
            else corpus.index.astype(str).tolist()
        )
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError(
                "Document ids are not unique; pass a unique `id_col` or "
                "reset the corpus index before indexing."
            )

        tokenized_corpus = [tokenize(t) for t in text]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._doc_ids = doc_ids
        logger.info("BM25Retriever indexed %d documents", len(self._doc_ids))

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search() called before .index().")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        scores = self._bm25.get_scores(tokenize(query))
        # Sort by score descending; ties broken by original document order
        # (stable, deterministic — no dependence on argsort's own tie
        # behavior, which numpy does not guarantee across versions).
        ranked_idx = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]

        return [
            SearchResult(doc_id=self._doc_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ranked_idx, start=1)
        ]
