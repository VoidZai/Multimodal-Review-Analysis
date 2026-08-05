"""Retrieval-quality metrics: Recall@k, Hit@k, nDCG@k, MRR (T3.2; PLAN.md §3 E3, §8 G1).

RQ2 (BM25 vs dense, M3.md T3.6/T3.7) and the chunking study (T3.4) both
stand entirely on these four numbers. PLAN.md calls out, explicitly,
that a wrong Recall implementation invalidates everything downstream —
so this module exists on its own, ahead of any real eval run, and is
tested against hand-computed toy cases rather than trusted by
construction.

CRAGB's relevance judgments (`benchmark/relevance_labels_v1.jsonl`,
built by pooling in M2/T2.7) are **binary** — a pooled candidate is
either relevant or it isn't, there is no graded 0-3 scale. All four
metrics here assume binary relevance accordingly (nDCG's gain is 1.0 for
a relevant hit, 0.0 otherwise; a graded variant would need a different
gain function, which is not needed by anything in this project).

Every function shares one signature shape:
    metric(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float
so a caller (the eval harness in T3.5+) can loop retrievers x k values x
metrics without special-casing any one of them. `ranked_doc_ids` is
assumed best-match-first (rank 1 at index 0) — exactly what
`Retriever.search()` (`cragb.retrieval.base`) already returns.

A question with **no relevant documents at all** (CRAGB's two
genuinely-empty negatives, M2.md/PLAN.md §14.2 — `fabric_quality_neg_000`
and `defects_neg_000`) makes every one of these four metrics undefined
(each has `relevant_ids`, or a count derived from it, in a denominator or
as the thing being ranked for). Rather than silently returning `0.0` or
`NaN` and risking it being averaged in as if it meant something, every
function raises `ValueError` on empty `relevant_ids` — callers must
explicitly filter such questions out before scoring retrieval quality
(they belong in the abstention-accuracy metric, E4/E5, not here).
"""

from __future__ import annotations

import math

MetricScores = dict[str, float]


def _validate(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant_ids:
        raise ValueError(
            "relevant_ids is empty; retrieval metrics are undefined for a "
            "question with no relevant documents (filter out abstention/"
            "negative questions before scoring retrieval quality)."
        )


def recall_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of `relevant_ids` present in the top-`k` of `ranked_doc_ids`.

    `Recall@k = |top-k ∩ relevant| / |relevant|`. Monotonically
    non-decreasing in `k` (a larger prefix can only add hits, never
    remove them).

    Raises:
        ValueError: if `k` is not positive or `relevant_ids` is empty.
    """
    _validate(ranked_doc_ids, relevant_ids, k)
    top_k = set(ranked_doc_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def hit_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant document appears in the top-`k`, else 0.0.

    Unlike Recall@k, Hit@k does not care *how many* relevant documents
    were found — only whether the top-`k` is useful at all. This is the
    "did the user get at least one good result" view of G1.

    Raises:
        ValueError: if `k` is not positive or `relevant_ids` is empty.
    """
    _validate(ranked_doc_ids, relevant_ids, k)
    top_k = set(ranked_doc_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at `k`, binary relevance.

    `DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)` (1-indexed rank `i`,
    `rel_i in {0, 1}`), normalized by the ideal DCG — the DCG of the best
    possible ranking, i.e. all relevant documents front-loaded into the
    first `min(k, |relevant_ids|)` positions. Rewards ranking relevant
    documents *higher*, not just including them somewhere in the top-`k`
    (which is all Recall@k / Hit@k measure).

    Raises:
        ValueError: if `k` is not positive or `relevant_ids` is empty.
    """
    _validate(ranked_doc_ids, relevant_ids, k)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    n_ideal = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg


def mrr(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top-`k`.

    `1 / rank` of the first hit (1-indexed), or `0.0` if no relevant
    document appears in the top-`k` at all. Answers "how far down did the
    user have to scroll for their first good result?" — the metric G1's
    `MRR` refers to (PLAN.md §3 E3, §8).

    Raises:
        ValueError: if `k` is not positive or `relevant_ids` is empty.
    """
    _validate(ranked_doc_ids, relevant_ids, k)
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def score_all(
    ranked_doc_ids: list[str], relevant_ids: set[str], k_values: list[int]
) -> dict[int, MetricScores]:
    """Recall/Hit/nDCG/MRR at every `k` in `k_values`, in one call.

    Convenience wrapper for the eval harness (T3.5+), which otherwise
    has to call all four metric functions at every `k` for every
    (retriever, question) pair.

    Args:
        ranked_doc_ids: retrieved document ids, best match first.
        relevant_ids: ground-truth relevant document ids for the query.
        k_values: the `k`s to score at, e.g. `[1, 3, 5, 10]`.

    Returns:
        `{k: {"recall": ..., "hit": ..., "ndcg": ..., "mrr": ...}}`, one
        entry per `k` in `k_values`.

    Raises:
        ValueError: if `k_values` is empty, any `k` is not positive, or
            `relevant_ids` is empty.
    """
    if not k_values:
        raise ValueError("k_values must be non-empty.")
    return {
        k: {
            "recall": recall_at_k(ranked_doc_ids, relevant_ids, k),
            "hit": hit_at_k(ranked_doc_ids, relevant_ids, k),
            "ndcg": ndcg_at_k(ranked_doc_ids, relevant_ids, k),
            "mrr": mrr(ranked_doc_ids, relevant_ids, k),
        }
        for k in k_values
    }
