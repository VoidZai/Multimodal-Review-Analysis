"""Chunking study: which chunking scheme retrieves best, on evidence (T3.4; PLAN.md §3 E2).

E2's question is decided here: whole-review vs fixed-token(128/256) vs
sentence-window, scored by Recall@k on CRAGB, using BM25 only (dense is
deliberately excluded — this is a chunking decision, not a retriever
comparison, and BM25 is enough to see whether splitting reviews helps or
hurts; RQ2's BM25-vs-dense comparison is T3.6+, after the chunking
scheme this study picks is locked in).

The one genuinely tricky piece is scoring: CRAGB's `relevant_ids`
(T2.6/T2.7's pooled labels) are **review-level** ids, but a retriever
indexed under `fixed_token`/`sentence_window` returns **chunk-level**
hits — several of which can belong to the same parent review.
`collapse_chunk_ranking_to_parents` turns a chunk-level ranking into a
review-level one (first occurrence of a parent wins its best rank,
later duplicate chunks from the same parent are dropped), so
Recall/Hit/nDCG/MRR@k always mean "top-k *reviews*", comparably across
every scheme including `whole_review` (where the two rankings are
identical by construction, since one chunk == one review there).

That collapsing step is also why each search here asks a retriever for
more than `k` raw chunks: if a review is split into several chunks, `k`
chunk-level hits can collapse into far fewer than `k` *unique* reviews,
which would structurally penalize multi-chunk schemes for a reason that
has nothing to do with retrieval quality. `chunk_search_multiplier`
(§`run_scheme_recall`) exists specifically to give collapsing enough raw
material to be fair to every scheme being compared.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from cragb.eval.bootstrap import bootstrap_ci
from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.metrics_retrieval import recall_at_k
from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import ChunkingConfig, chunk_corpus

logger = logging.getLogger(__name__)


def collapse_chunk_ranking_to_parents(
    ranked_chunk_ids: list[str], chunk_to_parent: dict[str, str]
) -> list[str]:
    """Ranked chunk ids -> ranked, de-duplicated parent (review) ids.

    Each parent keeps the rank of its *first* (best-scoring) chunk;
    later chunks from an already-seen parent are dropped rather than
    re-inserted or re-ranked. Relative order of first appearances is
    preserved, so the result is still best-match-first.

    Args:
        ranked_chunk_ids: chunk ids in best-first order, e.g. from
            `Retriever.search()`.
        chunk_to_parent: maps every chunk id to its parent review id
            (as produced by `cragb.retrieval.chunking.chunk_corpus`'s
            `chunk_id`/`parent_doc_id` columns).

    Returns:
        Parent ids in best-first order, each appearing at most once.

    Raises:
        KeyError: if a chunk id has no entry in `chunk_to_parent` —
            indicates the ranking came from a different index than the
            mapping, which would silently corrupt every score downstream
            if left uncaught.
    """
    parents: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranked_chunk_ids:
        try:
            parent_id = chunk_to_parent[chunk_id]
        except KeyError as exc:
            raise KeyError(
                f"chunk_id {chunk_id!r} not found in chunk_to_parent mapping; "
                "ranking and mapping must come from the same chunked index."
            ) from exc
        if parent_id not in seen:
            seen.add(parent_id)
            parents.append(parent_id)
    return parents


def run_scheme_recall(
    corpus: pd.DataFrame,
    chunking_config: ChunkingConfig,
    questions: list[RetrievalQuestion],
    k_values: list[int],
    text_col: str = "text",
    id_col: str | None = None,
    chunk_search_multiplier: int = 10,
    retriever: Retriever | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Recall@k for every question, under one chunking scheme, via BM25.

    Args:
        corpus: `corpus_v1`-shaped DataFrame (one row per review).
        chunking_config: which scheme to chunk `corpus` under.
        questions: questions to score. Must already be filtered to
            scorable questions (`cragb.eval.cragb_questions.filter_scorable`)
            — a question with empty `relevant_ids` makes `recall_at_k`
            raise, by design (see that module's docstring).
        k_values: the `k`s to score Recall at, e.g. `[1, 3, 5, 10]`.
        text_col: column containing review text.
        id_col: column to use as each review's id; the DataFrame index
            is used if `None`.
        chunk_search_multiplier: raw chunks requested per query is
            `max(k_values) * chunk_search_multiplier`, before collapsing
            to unique parents — see module docstring for why this
            exists. BM25 scoring cost is dominated by corpus size, not
            `k` (`BM25Retriever.search` scores every document regardless
            of `k`), so a generous multiplier here is effectively free.
        retriever: a `Retriever` to index and search with; defaults to a
            fresh `BM25Retriever()`. Exposed for testing (a stub
            retriever with known output) rather than expected to vary in
            production use.
        show_progress: show a `tqdm` progress bar over questions.

    Returns:
        Long-format `[question_id, type, k, recall]`, one row per
        (question, k) pair.
    """
    chunks = chunk_corpus(corpus, chunking_config, text_col=text_col, id_col=id_col)
    chunk_to_parent = dict(zip(chunks["chunk_id"], chunks["parent_doc_id"]))

    retriever = retriever if retriever is not None else BM25Retriever()
    retriever.index(chunks, text_col="text", id_col="chunk_id")
    logger.info(
        "scheme=%s: indexed %d chunks from %d reviews",
        chunking_config.scheme,
        len(chunks),
        corpus.shape[0],
    )

    chunk_search_k = max(k_values) * chunk_search_multiplier

    rows: list[dict[str, object]] = []
    iterator = tqdm(questions, desc=f"scoring [{chunking_config.scheme}]", disable=not show_progress)
    for question in iterator:
        chunk_hits = retriever.search(question.question, k=chunk_search_k)
        ranked_parents = collapse_chunk_ranking_to_parents(
            [hit.doc_id for hit in chunk_hits], chunk_to_parent
        )
        for k in k_values:
            rows.append(
                {
                    "question_id": question.id,
                    "type": question.type,
                    "k": k,
                    "recall": recall_at_k(ranked_parents, question.relevant_ids, k),
                }
            )

    return pd.DataFrame(rows)


def summarize_recall(
    per_question: pd.DataFrame,
    scheme: str,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng=None,
) -> pd.DataFrame:
    """Per-`k` mean Recall + bootstrap CI, from `run_scheme_recall`'s long output.

    Args:
        per_question: output of `run_scheme_recall` (one scheme).
        scheme: label to attach as the `scheme` column (kept separate
            from `run_scheme_recall` so a caller can re-summarize the
            same raw scores under a different label if needed).
        n_boot, alpha, rng: forwarded to `cragb.eval.bootstrap.bootstrap_ci`.

    Returns:
        `[scheme, k, recall_mean, recall_ci_lo, recall_ci_hi, n_questions]`,
        one row per distinct `k` in `per_question`.
    """
    rows: list[dict[str, object]] = []
    for k, group in per_question.groupby("k"):
        scores = group["recall"].tolist()
        lo, hi = bootstrap_ci(scores, n_boot=n_boot, alpha=alpha, rng=rng)
        rows.append(
            {
                "scheme": scheme,
                "k": k,
                "recall_mean": sum(scores) / len(scores),
                "recall_ci_lo": lo,
                "recall_ci_hi": hi,
                "n_questions": len(scores),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def run_chunking_study(
    corpus: pd.DataFrame,
    scheme_configs: dict[str, ChunkingConfig],
    questions: list[RetrievalQuestion],
    k_values: list[int],
    text_col: str = "text",
    id_col: str | None = None,
    chunk_search_multiplier: int = 10,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng=None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every scheme in `scheme_configs` and summarize Recall@k for each.

    The single entry point T3.4's notebook calls: build+index each
    chunking scheme, score every question, summarize with CIs.

    Args:
        corpus: `corpus_v1`-shaped DataFrame.
        scheme_configs: `{label: ChunkingConfig}`, e.g.
            `{"whole_review": ChunkingConfig(scheme="whole_review"),
              "fixed_token_128": ChunkingConfig(scheme="fixed_token", fixed_token_size=128)}`.
            `label` (not `config.scheme`) is what ends up in the output
            `scheme` column, so `fixed_token` at two different sizes can
            appear as two distinct rows.
        questions: pre-filtered scorable questions (see `run_scheme_recall`).
        k_values: the `k`s to score Recall at.
        text_col, id_col, chunk_search_multiplier: forwarded to
            `run_scheme_recall` for every scheme.
        n_boot, alpha, rng: forwarded to `summarize_recall` for every
            scheme. Pass a seeded `rng` for a fully reproducible study.
        show_progress: show a `tqdm` progress bar per scheme.

    Returns:
        `(summary, per_question)`:
        - `summary`: `[scheme, k, recall_mean, recall_ci_lo, recall_ci_hi, n_questions]`,
          concatenated across all schemes — ready to write to
          `results/tables/chunking_study_v1.csv` and plot.
        - `per_question`: `[scheme, question_id, type, k, recall]`,
          concatenated across all schemes — the raw scores behind
          `summary`, useful for per-type follow-up or debugging a
          surprising result.
    """
    summaries: list[pd.DataFrame] = []
    per_question_frames: list[pd.DataFrame] = []

    for label, config in scheme_configs.items():
        per_question = run_scheme_recall(
            corpus,
            config,
            questions,
            k_values,
            text_col=text_col,
            id_col=id_col,
            chunk_search_multiplier=chunk_search_multiplier,
            show_progress=show_progress,
        )
        per_question.insert(0, "scheme", label)
        per_question_frames.append(per_question)

        summaries.append(
            summarize_recall(per_question, scheme=label, n_boot=n_boot, alpha=alpha, rng=rng)
        )

    summary = pd.concat(summaries, ignore_index=True)
    per_question_all = pd.concat(per_question_frames, ignore_index=True)
    return summary, per_question_all
