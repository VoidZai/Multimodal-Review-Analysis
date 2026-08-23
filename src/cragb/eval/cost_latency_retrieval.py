"""Retrieval cost/latency harness (T5.3; PLAN.md §3 E6, §8 G4, M5.md T5.3).

The retrieval half of G4: for the *exact* BM25/dense configuration whose
Recall@k is already reported (`results/tables/retrieval_index_build_v1.json`
and `retrieval_eval_v1.csv`, T3.5/T3.6), measure what it actually costs to
build and to query — index build time and on-disk size, plus search
throughput (QPS) and p50/p90/p95 latency.

Two things this module is careful about, both load-bearing for a fair
BM25-vs-dense comparison and for M5.md's own validation checks:

- **Warm-up is mandatory, not optional.** A dense retriever's first query
  after `.index()` pays for CUDA context setup / cuBLAS kernel selection
  that steady-state queries don't; timing it in would make dense look
  artificially slower than it is in practice. `cragb.utils.timing.time_calls`
  runs `n_warmup` untimed passes over every question first, then times
  `n_repeats` more passes — see `configs/cost_latency.yaml`.
- **Index size means "what you'd have to persist to reuse this index",
  not "how big is the Python object holding it".** For `DenseRetriever`
  that excludes the loaded embedding model's weights (shared, reusable —
  not part of *this* index); see `Retriever.index_size_bytes` (T5.3's
  addition to `cragb.retrieval.base`) for why that needed a real
  interface method rather than reading a private attribute from here.

Everything here reuses T3.5/T3.6's building blocks unchanged
(`build_retriever`, `load_chunking_config`, `chunk_corpus`,
`load_retrieval_questions`) rather than re-deriving index construction —
the whole point of this task is to cost *that* configuration, not a
slightly different one.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cragb.eval.cragb_questions import load_retrieval_questions
from cragb.eval.run_retrieval_eval import (
    DenseRetrieverConfig,
    build_retriever,
)
from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import chunk_corpus, load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.timing import latency_stats, peak_memory, time_calls

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostLatencyConfig:
    """Resolved `configs/cost_latency.yaml` (or an equivalent file)."""

    seed: int
    corpus_in: str
    questions_in: str
    chunking_config_path: str
    out_path: str
    k: int
    n_warmup: int
    n_repeats: int
    dense: DenseRetrieverConfig

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.n_warmup < 0:
            raise ValueError(f"n_warmup must be >= 0, got {self.n_warmup}")
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")


def load_cost_latency_config(path: str | Path = "configs/cost_latency.yaml") -> CostLatencyConfig:
    """Load and validate `configs/cost_latency.yaml` (or an equivalent file).

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if a required key is missing.
        ValueError: if `k`/`n_warmup`/`n_repeats` fail `CostLatencyConfig`'s
            own validation, or the `dense` block fails `DenseRetrieverConfig`'s.
    """
    raw = load_config(path)
    dense_raw = raw["retrievers"]["dense"]
    return CostLatencyConfig(
        seed=raw["seed"],
        corpus_in=raw["paths"]["corpus_in"],
        questions_in=raw["paths"]["questions_in"],
        chunking_config_path=raw["chunking_config"],
        out_path=raw["paths"]["out"],
        k=raw["k"],
        n_warmup=raw["n_warmup"],
        n_repeats=raw["n_repeats"],
        dense=DenseRetrieverConfig(
            model_name=dense_raw["model_name"],
            batch_size=dense_raw["batch_size"],
            device=dense_raw.get("device"),
        ),
    )


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrieverCostLatency:
    """One retriever's row in `results/tables/retrieval_cost_latency_v1.csv`."""

    retriever: str
    n_chunks: int
    build_seconds: float
    index_bytes: int
    peak_rss_mb: float
    peak_vram_mb: float | None
    device: str | None
    n_queries: int
    qps: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_mean_ms: float

    def to_dict(self) -> dict:
        return {
            "retriever": self.retriever,
            "n_chunks": self.n_chunks,
            "build_seconds": round(self.build_seconds, 3),
            "index_bytes": self.index_bytes,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_vram_mb": (round(self.peak_vram_mb, 1) if self.peak_vram_mb is not None else None),
            "device": self.device,
            "n_queries": self.n_queries,
            "qps": round(self.qps, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p90_ms": round(self.latency_p90_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
        }


def measure_retriever_cost_latency(
    name: str,
    retriever: Retriever,
    chunks: pd.DataFrame,
    questions: list[str],
    k: int,
    n_warmup: int,
    n_repeats: int,
    device: str | None = None,
) -> RetrieverCostLatency:
    """Build `retriever` over `chunks`, then time `search(q, k)` across `questions`.

    Args:
        name: label for this retriever in the result row (e.g. `"bm25"`).
        retriever: an un-indexed `Retriever`.
        chunks: `[chunk_id, parent_doc_id, text]`, as produced by
            `cragb.retrieval.chunking.chunk_corpus`.
        questions: query strings to search with — every CRAGB v1 question,
            in `main`'s usage (latency does not care whether a question is
            answerable, only how long the query itself takes).
        k: `search(query, k)`'s `k`.
        n_warmup: untimed passes over `questions` before timing starts.
        n_repeats: timed passes over `questions`.
        device: recorded in the row only (e.g. `"cuda"`/`"cpu"`); has no
            effect on indexing or search here.

    Returns:
        A `RetrieverCostLatency` with build time, index size, peak
        memory, and latency/QPS statistics over `n_repeats * len(questions)`
        timed queries.

    Raises:
        RuntimeError: propagated from `build_retriever` if the smoke query
            (used only to confirm the freshly-built index is queryable)
            returns zero results.
    """
    retriever, _report = build_retriever(name, retriever, chunks, questions[0], device=device)
    build_seconds = _report.build_seconds
    index_bytes = retriever.index_size_bytes()

    args_list = [(q, k) for q in questions]
    durations_s = time_calls(retriever.search, args_list, repeats=n_repeats, warmup=n_warmup)
    stats = latency_stats(durations_s)
    mem = peak_memory()

    logger.info(
        "%s: build=%.2fs index=%.1fMB n_queries=%d qps=%.1f p50=%.1fms p95=%.1fms",
        name,
        build_seconds,
        index_bytes / 1e6,
        stats["n"],
        stats["qps"],
        stats["p50"] * 1000,
        stats["p95"] * 1000,
    )

    return RetrieverCostLatency(
        retriever=name,
        n_chunks=len(chunks),
        build_seconds=build_seconds,
        index_bytes=index_bytes,
        peak_rss_mb=mem["rss_mb"],
        peak_vram_mb=mem["vram_mb"],
        device=device,
        n_queries=stats["n"],
        qps=stats["qps"],
        latency_p50_ms=stats["p50"] * 1000,
        latency_p90_ms=stats["p90"] * 1000,
        latency_p95_ms=stats["p95"] * 1000,
        latency_mean_ms=stats["mean"] * 1000,
    )


def run_cost_latency_eval(corpus: pd.DataFrame, config: CostLatencyConfig) -> pd.DataFrame:
    """Measure both BM25 and dense under `config` and return the two-row result table.

    Args:
        corpus: `corpus_v1`-shaped DataFrame (one row per review, must
            have a `text` column).
        config: a `CostLatencyConfig` (see `load_cost_latency_config`).

    Returns:
        A `[retriever, n_chunks, build_seconds, index_bytes, peak_rss_mb,
        peak_vram_mb, device, n_queries, qps, latency_p50_ms,
        latency_p90_ms, latency_p95_ms, latency_mean_ms]` DataFrame, one
        row per retriever.

    Raises:
        ImportError: if `sentence-transformers`/`torch`/`faiss` are not
            importable in the running Python (run via the `C:\\venv\\cragb`
            short-path venv, PLAN.md §14.1).
    """
    chunking_config = load_chunking_config(config.chunking_config_path)
    chunks = chunk_corpus(corpus, chunking_config)
    logger.info(
        "chunked corpus_v1 under scheme=%s: %d chunks from %d reviews",
        chunking_config.scheme,
        len(chunks),
        len(corpus),
    )

    questions = [q.question for q in load_retrieval_questions(config.questions_in)]
    logger.info("measuring cost/latency over %d CRAGB v1 questions", len(questions))

    rows: list[RetrieverCostLatency] = []

    rows.append(
        measure_retriever_cost_latency(
            "bm25",
            BM25Retriever(),
            chunks,
            questions,
            config.k,
            config.n_warmup,
            config.n_repeats,
            device=None,
        )
    )

    # Imported lazily, mirroring cragb.eval.run_retrieval_eval.build_all_retrievers:
    # the dense stack is heavy and Windows-MAX_PATH-sensitive (PLAN.md §14.1), so this
    # module stays importable (and its BM25-only path runnable) without it installed.
    import torch

    from cragb.retrieval.dense import DenseRetriever

    resolved_device = config.dense.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows.append(
        measure_retriever_cost_latency(
            "dense",
            DenseRetriever(
                model_name=config.dense.model_name,
                batch_size=config.dense.batch_size,
                device=config.dense.device,
            ),
            chunks,
            questions,
            config.k,
            config.n_warmup,
            config.n_repeats,
            device=resolved_device,
        )
    )

    return pd.DataFrame([r.to_dict() for r in rows])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure BM25 + dense retrieval cost/latency over corpus_v1 under "
        "the locked chunking scheme (T5.3; PLAN.md §3 E6, §8 G4)."
    )
    parser.add_argument("--config", default="configs/cost_latency.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_cost_latency_config(args.config)

    from cragb.utils.seeds import set_global_seed

    set_global_seed(config.seed)

    corpus = pd.read_parquet(resolve_path(config.corpus_in), columns=["text"])
    logger.info("loaded corpus_in=%s: %d reviews", config.corpus_in, len(corpus))

    table = run_cost_latency_eval(corpus, config)

    out_path = resolve_path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("wrote cost/latency table (%d rows) to %s", len(table), out_path)


if __name__ == "__main__":
    main()
