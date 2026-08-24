"""Context-block builder for grounded QA (T4a.2; PLAN.md §3 E4, M4a.md T4a.2).

T4a.1's prompt (`prompts/grounded_qa_v1.md`) expects one thing from its caller: a
`$context_block` string listing retrieved reviews, each labeled with the exact id the
model must cite back (`[doc_id]`) and whether it has a photo (`has_photo: yes/no`). This
module is the single place that turns "a question + an indexed retriever" into that
string, so T4a.3's generation pipeline never has to know how retrieval or id-resolution
work — only that `build_context(...)` returns something ready to drop into the template.

Two id-resolution details this module must get right, both inherited from decisions
already locked earlier in the project rather than invented here:

- **doc_ids must match what CRAGB's ground truth uses.** `relevant_ids` in
  `benchmark/cragb_v1.jsonl` and the citations in `cragb.bench.reference_answers` are
  review ids — the same ids `Retriever.index`/`cragb.retrieval.chunking.chunk_corpus`
  derive from `corpus.index.astype(str)` when no explicit `id_col` is given
  (`cragb.retrieval.base.Retriever.index`). `build_corpus_lookup` reuses that exact
  contract so a `doc_id` returned by a retriever always resolves back to a real review.
- **a retrieved hit is a chunk id, not necessarily a review id.** `configs/chunking.yaml`
  currently locks `scheme: whole_review` (T3.4's decision), under which chunk_id ==
  parent_doc_id and this distinction is moot — but `build_context` still collapses
  through `chunk_to_parent` unconditionally via
  `cragb.eval.chunking_study.collapse_chunk_ranking_to_parents`, the same helper T3.6's
  eval harness uses, so nothing here silently breaks if the chunking config is ever
  swapped to `fixed_token`/`sentence_window`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cragb.eval.chunking_study import collapse_chunk_ranking_to_parents
from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import ChunkingConfig, chunk_corpus

# Truncate long reviews so a k-review context block stays a bounded prompt
# size regardless of how verbose any single review is. 600 chars is
# generous relative to corpus_v1's review lengths (T1.9's EDA: the large
# majority of reviews are well under this) while still capping the rare
# long outlier.
DEFAULT_MAX_EXCERPT_CHARS = 600

# Hits are searched `k * DEFAULT_SEARCH_MULTIPLIER` deep before collapsing
# chunk ids to unique parent reviews, so that (once chunking schemes other
# than the locked `whole_review` are in play) several top-ranked chunks
# belonging to the same review don't starve the collapsed result below
# `k` distinct reviews. Mirrors `cragb.eval.chunking_study.run_scheme_recall`'s
# `chunk_search_multiplier` pattern.
DEFAULT_SEARCH_MULTIPLIER = 5


@dataclass(frozen=True)
class CorpusLookup:
    """Precomputed per-review text/photo lookups, keyed by doc_id (str).

    Built once per corpus (`build_corpus_lookup`) and reused across many
    `build_context` calls — the pilot run (T4a.5) calls `build_context`
    once per CRAGB question, and re-scanning all of `corpus_v1` on every
    call would be wasteful.
    """

    text_by_id: dict[str, str]
    has_photo_by_id: dict[str, bool]

    def __post_init__(self) -> None:
        if set(self.text_by_id) != set(self.has_photo_by_id):
            raise ValueError(
                "CorpusLookup's text_by_id and has_photo_by_id must cover the same doc_ids."
            )


def build_corpus_lookup(
    corpus: pd.DataFrame,
    text_col: str = "text",
    image_flag_col: str = "has_image",
    id_col: str | None = None,
) -> CorpusLookup:
    """Precompute a `CorpusLookup` for `corpus`.

    Args:
        corpus: one row per review (e.g. `corpus_v1`).
        text_col: column containing the review text.
        image_flag_col: boolean column marking whether a review has a
            photo attached (`corpus_v1`'s `has_image` column).
        id_col: column to use as each review's doc_id. If `None`, the
            DataFrame's index is used (`corpus.index.astype(str)`) — the
            same default `Retriever.index`/`chunk_corpus` use, so a
            `doc_id` a retriever returns always resolves here as long as
            the retriever was indexed over this same corpus.

    Returns:
        A `CorpusLookup` covering every row of `corpus`.

    Raises:
        ValueError: if `corpus` is empty, or resulting doc_ids are not
            unique.
    """
    if corpus.empty:
        raise ValueError("Cannot build a corpus lookup from an empty corpus.")

    doc_ids = (
        corpus[id_col].astype(str).tolist()
        if id_col is not None
        else corpus.index.astype(str).tolist()
    )
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(
            "Document ids are not unique; pass a unique `id_col` or reset "
            "the corpus index before building a lookup."
        )

    texts = corpus[text_col].fillna("").astype(str).tolist()
    has_photo = corpus[image_flag_col].fillna(False).astype(bool).tolist()

    return CorpusLookup(
        text_by_id=dict(zip(doc_ids, texts)),
        has_photo_by_id=dict(zip(doc_ids, has_photo)),
    )


@dataclass(frozen=True)
class ContextBlock:
    """A rendered `$context_block` value, plus the bookkeeping later stages need.

    `doc_ids` and `photo_flags` let T4a.4's citation-validity checker
    confirm a generated answer only cites ids that were actually shown to
    the model, without re-parsing `text` itself.
    """

    text: str
    doc_ids: tuple[str, ...]  # rank-ordered, deduped, best-match-first
    photo_flags: dict[str, bool]  # doc_id -> whether that review has a photo


def render_excerpt(doc_id: str, text: str, has_photo: bool, max_chars: int) -> str:
    """One excerpt block, matching the schema `grounded_qa_v1.md` documents to the model.

    Public (not module-private) deliberately: `cragb.finetune.sample_contexts` (T7.2)
    renders training-context excerpts in this exact same `[doc_id] has_photo: yes/no\\n
    <snippet>` shape, so a sampler-built context is indistinguishable, byte-for-byte, from
    one `build_context` would have retrieved. Reimplementing this format string a second
    time would risk exactly the training/inference skew T7.1's prompt-parity guarantee
    exists to prevent.
    """
    snippet = text.strip()[:max_chars]
    photo_flag = "yes" if has_photo else "no"
    return f"[{doc_id}] has_photo: {photo_flag}\n{snippet}"


def build_context(
    question: str,
    retriever: Retriever,
    chunk_to_parent: dict[str, str],
    lookup: CorpusLookup,
    k: int,
    search_multiplier: int = DEFAULT_SEARCH_MULTIPLIER,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> ContextBlock:
    """Retrieve the top-`k` reviews for `question` and render them as a context block.

    Args:
        question: free-text query — the CRAGB question being answered.
        retriever: an already-indexed `Retriever` (e.g. from
            `index_bm25_retriever`).
        chunk_to_parent: maps every chunk id the retriever can return to
            its parent review id (as produced by
            `cragb.retrieval.chunking.chunk_corpus`'s `chunk_id`/
            `parent_doc_id` columns). Under the locked `whole_review`
            scheme this is the identity map, but it is always applied so
            nothing here depends on that scheme staying locked.
        lookup: a `CorpusLookup` built over the same corpus the retriever
            was indexed on.
        k: number of distinct reviews to include in the context.
        search_multiplier: how much deeper than `k` to search before
            collapsing chunk hits to unique parent reviews (see module
            docstring).
        max_excerpt_chars: per-review truncation length.

    Returns:
        A `ContextBlock`. If retrieval returns no results at all (only
        possible if `retriever` was indexed over an empty corpus), `text`
        is a placeholder noting no excerpts were found and `doc_ids`/
        `photo_flags` are empty — this is a legitimate input to the
        grounded-QA prompt, which is instructed to abstain when its
        context is empty.

    Raises:
        ValueError: if `k` is not positive.
        KeyError: if a doc_id the retriever/`chunk_to_parent` produced
            has no entry in `lookup` — indicates the retriever and the
            lookup were built from different corpora.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    hits = retriever.search(question, k=k * search_multiplier)
    ranked_chunk_ids = [hit.doc_id for hit in hits]
    parent_ids = collapse_chunk_ranking_to_parents(ranked_chunk_ids, chunk_to_parent)
    doc_ids = tuple(parent_ids[:k])

    if not doc_ids:
        return ContextBlock(
            text="(no review excerpts were found for this question)", doc_ids=(), photo_flags={}
        )

    missing = [doc_id for doc_id in doc_ids if doc_id not in lookup.text_by_id]
    if missing:
        raise KeyError(
            f"doc_id(s) {missing} returned by the retriever have no entry in the corpus "
            "lookup; retriever and lookup must be built from the same corpus."
        )

    excerpts = [
        render_excerpt(
            doc_id, lookup.text_by_id[doc_id], lookup.has_photo_by_id[doc_id], max_excerpt_chars
        )
        for doc_id in doc_ids
    ]
    photo_flags = {doc_id: lookup.has_photo_by_id[doc_id] for doc_id in doc_ids}
    return ContextBlock(text="\n\n".join(excerpts), doc_ids=doc_ids, photo_flags=photo_flags)


def index_bm25_retriever(
    corpus: pd.DataFrame, chunking_config: ChunkingConfig
) -> tuple[BM25Retriever, dict[str, str]]:
    """Chunk `corpus` under `chunking_config` and index a fresh `BM25Retriever`.

    A thin convenience wrapper for `build_context`'s two BM25-specific
    inputs (an indexed retriever and its chunk->parent map). Kept
    separate from `cragb.eval.run_retrieval_eval.build_all_retrievers`
    deliberately: that function also builds the dense retriever, logs an
    `IndexBuildReport`, and requires a smoke query — machinery T4a.1's
    config-locked `retrieval.retriever: bm25` choice (PLAN.md §14.1: no
    GPU/venv dependency) doesn't need here.

    Args:
        corpus: `corpus_v1`-shaped DataFrame (one row per review).
        chunking_config: which scheme to chunk `corpus` under (T4a.1's
            config points this at `configs/chunking.yaml`, currently
            locked to `whole_review`).

    Returns:
        `(retriever, chunk_to_parent)`, ready to pass into `build_context`.
    """
    chunks = chunk_corpus(corpus, chunking_config)
    retriever = BM25Retriever()
    retriever.index(chunks, text_col="text", id_col="chunk_id")
    chunk_to_parent = dict(zip(chunks["chunk_id"], chunks["parent_doc_id"]))
    return retriever, chunk_to_parent
