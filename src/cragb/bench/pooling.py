"""CRAGB pooling: union of retriever top-k per question (T2.6; PLAN.md §3 E1, §1.4 risk A).

Hand-labeling relevance over the full ~200k-row `corpus_v1` is
impossible; PLAN.md's fix (risk A) is the standard TREC **pooling**
remedy: for every CRAGB question, run *every retriever under study*,
take the **union** of their top-k results, and only ask a human (T2.7)
to judge that pool. Recall/Hit@k then mean what they say — with respect
to the systems being compared — instead of silently counting unlabeled
true positives as misses.

A note on what "pool size" actually measures here: `search(query, k)`
always returns exactly `k` documents (the `k` highest-scoring ones,
whether or not they're actually relevant — that's the definition of
top-k retrieval, not a relevance filter). So with 2 retrievers at
`k_max`, a pool's size is mathematically bounded in `[k_max, 2*k_max]`:
it can never be empty, and it only hits the floor of exactly `k_max`
when both retrievers return the *identical* top-k set — i.e. the second
retriever added zero diversity. That floor case, not an "empty pool", is
the meaningful thing to flag (`min_pool_size` in `configs/pooling.yaml`
defaults just above `k_max` for this reason). A genuinely empty pool
(`pool_size == 0`) can still happen defensively — an empty corpus, or a
misconfigured `k`/no retrievers — and is treated as a hard failure.

`pool_question`/`pool_all_questions` take retrievers via the `Retriever`
interface (T2.4/T2.5), not concrete classes, so this module has no
import-time dependency on the heavy dense-retrieval stack
(`torch`/`sentence-transformers`/`faiss`) — only `main()`, which needs a
real `DenseRetriever`, imports it, and does so lazily.

Usage:
    python -m cragb.bench.pooling --config configs/pooling.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionRecord:
    id: str
    category: str
    question: str
    is_negative: bool


def load_questions(path: str | Path) -> list[QuestionRecord]:
    """Load T2.3's `cragb_questions_v1.jsonl` into `QuestionRecord`s.

    `image_target` (also present in that file) isn't needed for pooling
    and is deliberately not carried into `QuestionRecord` — this module
    only needs enough to run a query and know whether to expect a
    non-trivial pool.
    """
    records: list[QuestionRecord] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                QuestionRecord(
                    id=obj["id"],
                    category=obj["category"],
                    question=obj["question"],
                    is_negative=bool(obj["is_negative"]),
                )
            )
    return records


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolHit:
    """One retriever's hit on one document, for provenance."""

    doc_id: str
    retriever: str
    rank: int
    score: float


@dataclass(frozen=True)
class QuestionPool:
    question_id: str
    category: str
    is_negative: bool
    doc_ids: list[str]
    hits: list[PoolHit]
    pool_size: int
    n_retrievers_used: int
    overlap_count: int  # documents found by more than one retriever


def pool_question(
    question: QuestionRecord,
    retrievers: dict[str, Retriever],
    k_values: list[int],
) -> QuestionPool:
    """Union the top-k_max results from every retriever in `retrievers`.

    Args:
        question: the question to pool candidates for.
        retrievers: name -> already-indexed `Retriever`.
        k_values: pooling depths; only `max(k_values)` is actually
            queried (see module docstring for why that's equivalent to
            querying at every value separately).

    Returns:
        A `QuestionPool` with the union of doc ids, sorted by best
        (lowest) rank any retriever gave them, and full hit provenance.

    Raises:
        ValueError: if `k_values` is empty or non-positive.
    """
    if not k_values or max(k_values) <= 0:
        raise ValueError(f"k_values must be a non-empty list of positive ints, got {k_values}")
    k_max = max(k_values)

    hits: list[PoolHit] = []
    for name, retriever in retrievers.items():
        for result in retriever.search(question.question, k=k_max):
            hits.append(PoolHit(doc_id=result.doc_id, retriever=name, rank=result.rank, score=result.score))

    retrievers_by_doc: dict[str, set[str]] = {}
    best_rank: dict[str, int] = {}
    for hit in hits:
        retrievers_by_doc.setdefault(hit.doc_id, set()).add(hit.retriever)
        if hit.doc_id not in best_rank or hit.rank < best_rank[hit.doc_id]:
            best_rank[hit.doc_id] = hit.rank

    doc_ids = sorted(best_rank, key=lambda doc_id: (best_rank[doc_id], doc_id))
    overlap_count = sum(1 for names in retrievers_by_doc.values() if len(names) > 1)

    return QuestionPool(
        question_id=question.id,
        category=question.category,
        is_negative=question.is_negative,
        doc_ids=doc_ids,
        hits=hits,
        pool_size=len(doc_ids),
        n_retrievers_used=len(retrievers),
        overlap_count=overlap_count,
    )


def pool_all_questions(
    questions: list[QuestionRecord],
    retrievers: dict[str, Retriever],
    k_values: list[int],
) -> list[QuestionPool]:
    """Pool every question in `questions`, in order."""
    return [pool_question(q, retrievers, k_values) for q in questions]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolingReport:
    ok: bool
    n_questions: int
    min_size: int
    max_size: int
    mean_size: float
    zero_pools: tuple[str, ...]  # hard failure: pool_size == 0
    low_diversity_pools: tuple[str, ...]  # soft flag: pool_size <= min_pool_size
    messages: tuple[str, ...]


def summarize_pools(pools: list[QuestionPool], min_pool_size: int) -> PoolingReport:
    """Summarize pool sizes and flag anomalies.

    Two distinct anomaly classes (see module docstring for why they're
    different, not degrees of the same thing):

    - `zero_pools` (hard failure, `ok=False`): a pool with literally no
      candidates — a defensive check, since `search(q, k>0)` on a
      non-empty corpus always returns `k` results, but a misconfigured
      `k_values`/empty corpus/no retrievers should still surface loudly
      rather than produce a silently-empty benchmark row.
    - `low_diversity_pools` (soft flag, `ok` unaffected): a non-negative
      question whose pool sits at or near `k_max` — the retrievers
      agreed almost perfectly, adding little to no diversity beyond a
      single retriever's own top-k. Worth a look (raise k, or the
      retrievers may genuinely agree because the question is easy), but
      not itself evidence of a broken pipeline.

    Negative (deliberately-unanswerable) questions are excluded from
    `low_diversity_pools`: a tiny/overlapping pool for those is not a
    concern the way it is for an answerable question.
    """
    sizes = {p.question_id: p.pool_size for p in pools}

    zero_pools = tuple(p.question_id for p in pools if p.pool_size == 0)
    low_diversity = tuple(
        p.question_id
        for p in pools
        if not p.is_negative and p.pool_size > 0 and p.pool_size <= min_pool_size
    )

    messages: list[str] = []
    if zero_pools:
        messages.append(f"{len(zero_pools)} question(s) retrieved an EMPTY pool (hard failure): {list(zero_pools)}")
    if low_diversity:
        messages.append(
            f"{len(low_diversity)} answerable question(s) have a low-diversity pool "
            f"(size <= {min_pool_size}): {list(low_diversity)}"
        )
    if not messages:
        messages.append("No empty or low-diversity pools.")

    return PoolingReport(
        ok=not zero_pools,
        n_questions=len(pools),
        min_size=min(sizes.values()) if sizes else 0,
        max_size=max(sizes.values()) if sizes else 0,
        mean_size=round(sum(sizes.values()) / len(sizes), 2) if sizes else 0.0,
        zero_pools=zero_pools,
        low_diversity_pools=low_diversity,
        messages=tuple(messages),
    )


# --------------------------------------------------------------------------
# I/O + orchestration
# --------------------------------------------------------------------------


def write_pools_jsonl(pools: list[QuestionPool], out_path: str | Path) -> Path:
    """Write `pools` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for pool in pools:
            row = {
                "question_id": pool.question_id,
                "category": pool.category,
                "is_negative": pool.is_negative,
                "pool_size": pool.pool_size,
                "n_retrievers_used": pool.n_retrievers_used,
                "overlap_count": pool.overlap_count,
                "doc_ids": pool.doc_ids,
                "hits": [asdict(h) for h in pool.hits],
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return resolved


def _build_retrievers(corpus: pd.DataFrame, cfg: dict) -> dict[str, Retriever]:
    """Index BM25 + dense retrievers over `corpus`.

    `DenseRetriever` is imported here, not at module top-level, so that
    `cragb.bench.pooling` stays importable (and its retriever-agnostic
    logic testable) in environments without the heavy dense-retrieval
    stack installed — only actually running pooling requires it.
    """
    from cragb.retrieval.dense import DenseRetriever

    text_col = cfg["corpus"]["text_col"]

    bm25 = BM25Retriever()
    bm25.index(corpus, text_col=text_col)

    dense_cfg = cfg["dense"]
    dense = DenseRetriever(
        model_name=dense_cfg["model_name"],
        batch_size=dense_cfg["batch_size"],
        device=dense_cfg.get("device"),
    )
    dense.index(corpus, text_col=text_col)

    return {"bm25": bm25, "dense": dense}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pooling.yaml", help="Path to pooling config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    questions = load_questions(cfg["paths"]["questions_in"])
    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))

    retrievers = _build_retrievers(corpus, cfg)
    pools = pool_all_questions(questions, retrievers, cfg["pooling"]["k_values"])

    report = summarize_pools(pools, cfg["pooling"]["min_pool_size"])
    out_path = write_pools_jsonl(pools, cfg["paths"]["pools_out"])

    logger.info("Wrote %d pools to %s", len(pools), out_path)
    logger.info(
        "pool sizes: min=%d max=%d mean=%.2f", report.min_size, report.max_size, report.mean_size
    )
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
