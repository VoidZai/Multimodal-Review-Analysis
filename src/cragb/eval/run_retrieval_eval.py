"""Retrieval eval harness: config + index build (T3.5; PLAN.md §3 E3).

T3.5's scope is narrower than the file name suggests: this module's job
right now is to stand up `configs/retrieval_eval.yaml`, build BM25 and
dense indexes over `corpus_v1` under T3.4's locked chunking scheme, and
confirm both are actually queryable — logging build time as it goes, so
that number is available for the cost/latency table later (E6/M5). It
does **not** yet run the full CRAGB question set through both retrievers
or compute Recall/nDCG/MRR — that is T3.6's "eval-running portion",
added to this same file rather than a new one, since it shares this
module's config and indexes.

Windows note (PLAN.md §14.1): `DenseRetriever` needs `torch` +
`sentence-transformers` + `faiss`, which fail to install into this
project's main Python environment on Windows (MAX_PATH). Run this
module's `main()` via the short-path venv instead:

    C:\\venv\\cragb\\Scripts\\python.exe -m cragb.eval.run_retrieval_eval

`DenseRetriever`/`sentence-transformers`/`torch` are imported lazily,
inside `build_all_retrievers`, not at module import time — mirroring
`cragb.bench.pooling`'s precedent — so this module (and its BM25-only
tests) still import cleanly in the main environment, where that stack
is unavailable.

GPU: `DenseRetriever` auto-detects CUDA via `torch.cuda.is_available()`
when `retrievers.dense.device` in the config is `null` (the default) —
on this project's dev machine that resolves to the RTX 3050 Laptop GPU
inside the venv above, a ~10-20x encoding speedup over CPU for
`corpus_v1`'s ~200k reviews (PLAN.md §1.3 hardware note).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import chunk_corpus, load_chunking_config
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseRetrieverConfig:
    """Resolved `retrievers.dense` block of `configs/retrieval_eval.yaml`."""

    model_name: str
    batch_size: int
    device: str | None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("dense.model_name must be non-empty.")
        if self.batch_size <= 0:
            raise ValueError(f"dense.batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class RetrievalEvalConfig:
    """Resolved `configs/retrieval_eval.yaml`."""

    seed: int
    corpus_in: str
    questions_in: str
    chunking_config_path: str
    k_values: tuple[int, ...]
    build_report_out: str
    dense: DenseRetrieverConfig

    def __post_init__(self) -> None:
        if not self.k_values:
            raise ValueError("k_values must be non-empty.")
        if any(k <= 0 for k in self.k_values):
            raise ValueError(f"all k_values must be positive, got {self.k_values}")


def load_retrieval_eval_config(
    path: str | Path = "configs/retrieval_eval.yaml",
) -> RetrievalEvalConfig:
    """Load and validate `configs/retrieval_eval.yaml` (or an equivalent file).

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if a required key is missing.
        ValueError: if `k_values` is empty/non-positive, or the `dense`
            block fails `DenseRetrieverConfig`'s own validation.
    """
    raw = load_config(path)
    dense_raw = raw["retrievers"]["dense"]
    return RetrievalEvalConfig(
        seed=raw["seed"],
        corpus_in=raw["paths"]["corpus_in"],
        questions_in=raw["paths"]["questions_in"],
        chunking_config_path=raw["chunking_config"],
        k_values=tuple(raw["k_values"]),
        build_report_out=raw["paths"]["build_report_out"],
        dense=DenseRetrieverConfig(
            model_name=dense_raw["model_name"],
            batch_size=dense_raw["batch_size"],
            device=dense_raw.get("device"),
        ),
    )


# --------------------------------------------------------------------------
# Index build
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeHit:
    """One smoke-query result, kept human-readable for a manual sanity check."""

    doc_id: str
    score: float
    snippet: str


@dataclass(frozen=True)
class IndexBuildReport:
    """What happened when a retriever was indexed, for the eventual E6 cost table."""

    retriever: str
    n_chunks: int
    n_parent_docs: int
    build_seconds: float
    device: str | None
    smoke_query: str
    smoke_hits: tuple[SmokeHit, ...]

    def to_dict(self) -> dict:
        return {
            "retriever": self.retriever,
            "n_chunks": self.n_chunks,
            "n_parent_docs": self.n_parent_docs,
            "build_seconds": round(self.build_seconds, 3),
            "device": self.device,
            "smoke_query": self.smoke_query,
            "smoke_hits": [
                {"doc_id": h.doc_id, "score": round(h.score, 4), "snippet": h.snippet}
                for h in self.smoke_hits
            ],
        }


def build_retriever(
    name: str,
    retriever: Retriever,
    chunks: pd.DataFrame,
    smoke_query: str,
    device: str | None = None,
) -> tuple[Retriever, IndexBuildReport]:
    """Index `retriever` over `chunks`, smoke-test it, and report the build.

    Args:
        name: label for this retriever in the report (e.g. `"bm25"`).
        retriever: an un-indexed `Retriever`.
        chunks: `[chunk_id, parent_doc_id, text]`, as produced by
            `cragb.retrieval.chunking.chunk_corpus`.
        smoke_query: a free-text query to confirm the freshly-built
            index actually returns something sensible.
        device: recorded in the report only (e.g. `"cuda"`/`"cpu"` for a
            dense retriever); has no effect on indexing itself here.

    Returns:
        `(retriever, report)` — the now-indexed retriever, and a report
        of how long indexing took and what the smoke query returned.

    Raises:
        RuntimeError: if the smoke query returns zero results — the
            index would technically exist but be useless, and this is
            the cheapest point in the pipeline to catch that rather than
            discover it deep into a later eval run.
    """
    t0 = time.monotonic()
    retriever.index(chunks, text_col="text", id_col="chunk_id")
    build_seconds = time.monotonic() - t0

    hits = retriever.search(smoke_query, k=3)
    if not hits:
        raise RuntimeError(
            f"{name} smoke query {smoke_query!r} returned no results after "
            f"indexing {len(chunks)} chunks; index appears broken."
        )

    text_by_chunk_id = dict(zip(chunks["chunk_id"], chunks["text"]))
    smoke_hits = tuple(
        SmokeHit(doc_id=hit.doc_id, score=hit.score, snippet=text_by_chunk_id[hit.doc_id][:120])
        for hit in hits
    )

    report = IndexBuildReport(
        retriever=name,
        n_chunks=len(chunks),
        n_parent_docs=int(chunks["parent_doc_id"].nunique()),
        build_seconds=build_seconds,
        device=device,
        smoke_query=smoke_query,
        smoke_hits=smoke_hits,
    )
    logger.info(
        "%s: indexed %d chunks (%d reviews) in %.2fs; smoke query %r top-3 doc_ids: %s",
        name,
        report.n_chunks,
        report.n_parent_docs,
        build_seconds,
        smoke_query,
        [h.doc_id for h in smoke_hits],
    )
    return retriever, report


def build_all_retrievers(
    corpus: pd.DataFrame,
    config: RetrievalEvalConfig,
    smoke_query: str = "does this run true to size",
) -> dict[str, tuple[Retriever, IndexBuildReport]]:
    """Chunk `corpus` under T3.4's locked scheme and build both BM25 and dense indexes.

    Args:
        corpus: `corpus_v1`-shaped DataFrame (one row per review, must
            have a `text` column).
        config: a `RetrievalEvalConfig` (see `load_retrieval_eval_config`).
        smoke_query: forwarded to `build_retriever` for both retrievers.

    Returns:
        `{"bm25": (retriever, report), "dense": (retriever, report)}`.

    Raises:
        RuntimeError: propagated from `build_retriever` if either
            retriever's smoke query comes back empty.
        ImportError: if `sentence-transformers`/`torch`/`faiss` are not
            importable in the running Python (see module docstring —
            run via the venv on Windows).
    """
    chunking_config = load_chunking_config(config.chunking_config_path)
    chunks = chunk_corpus(corpus, chunking_config)
    logger.info(
        "chunked corpus_v1 under scheme=%s: %d chunks from %d reviews",
        chunking_config.scheme,
        len(chunks),
        len(corpus),
    )

    results: dict[str, tuple[Retriever, IndexBuildReport]] = {}

    results["bm25"] = build_retriever(
        "bm25", BM25Retriever(), chunks, smoke_query, device=None
    )

    # Imported lazily (see module docstring): the dense stack is heavy
    # and Windows-MAX_PATH-sensitive, so importing it only when a dense
    # index is actually about to be built keeps the rest of this module
    # (and its BM25-only tests) usable without that stack installed.
    import torch

    from cragb.retrieval.dense import DenseRetriever

    resolved_device = config.dense.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dense_retriever = DenseRetriever(
        model_name=config.dense.model_name,
        batch_size=config.dense.batch_size,
        device=config.dense.device,
    )
    results["dense"] = build_retriever(
        "dense", dense_retriever, chunks, smoke_query, device=resolved_device
    )

    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 + dense retrieval indexes over corpus_v1 (T3.5)."
    )
    parser.add_argument("--config", default="configs/retrieval_eval.yaml")
    parser.add_argument("--smoke-query", default="does this run true to size")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_retrieval_eval_config(args.config)
    corpus = pd.read_parquet(resolve_path(config.corpus_in), columns=["text"])
    logger.info("loaded corpus_in=%s: %d reviews", config.corpus_in, len(corpus))

    built = build_all_retrievers(corpus, config, smoke_query=args.smoke_query)

    out_path = resolve_path(config.build_report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports = [report.to_dict() for _retriever, report in built.values()]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    logger.info("wrote build report (%d retriever(s)) to %s", len(reports), out_path)


if __name__ == "__main__":
    main()
