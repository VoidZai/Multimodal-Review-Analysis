"""Swappable retriever interface (T2.4; PLAN.md §3 E3, §5).

RQ2 (PLAN.md §2/§3-E3) compares BM25 against a dense embedding retriever
on identical questions, chunking, and k. That comparison is only clean if
both retrievers are called through the same shape of code — otherwise a
difference in *how* each is invoked, not the retrieval method itself,
could explain a result gap. `Retriever` is that single shape: `.index()`
builds whatever internal structure a given method needs, `.search()`
returns a ranked list of `SearchResult`s, and nothing about the caller
(pooling in T2.6, the eval harness in E3/E5) needs to know or care which
concrete retriever it's holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit for a query.

    `rank` is 1-indexed and redundant with list position by construction
    (`results[i].rank == i + 1`), but carrying it explicitly means a
    caller that pulls a single `SearchResult` out of context (e.g. into
    a pooled-candidates row, T2.6) doesn't have to reconstruct its
    position from the list it came from.
    """

    doc_id: str
    score: float
    rank: int


class Retriever(ABC):
    """Common interface every retrieval method (BM25, dense, ...) implements."""

    @abstractmethod
    def index(
        self,
        corpus: pd.DataFrame,
        text_col: str = "text",
        id_col: str | None = None,
    ) -> None:
        """Build the retriever's index over `corpus`.

        Args:
            corpus: one row per document/chunk to be searchable. T2.4
                treats each corpus row (a whole review) as one document;
                the chunking study (E2) may revisit this.
            text_col: column containing the searchable text.
            id_col: column to use as each document's id. If `None`, the
                DataFrame's index is used (converted to `str`) — stable
                as long as the underlying corpus file doesn't change,
                which `corpus_v1`'s manifest hash (T1.8) guarantees.

        Raises:
            ValueError: if `corpus` is empty or resulting document ids
                are not unique.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return the top-`k` documents for `query`, best match first.

        Args:
            query: free-text query.
            k: number of results to return (must be positive). If fewer
                than `k` documents exist, all of them are returned.

        Returns:
            Up to `k` `SearchResult`s, sorted best-first (rank 1 first).

        Raises:
            RuntimeError: if called before `.index()`.
            ValueError: if `k` is not positive.
        """
        raise NotImplementedError
