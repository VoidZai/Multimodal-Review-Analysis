"""Retrieval eval harness: config, index build, RQ2 eval, per-type breakdown
(T3.5/T3.6/T3.7; PLAN.md §3 E3).

Three stages, all in this one module (T3.5's docstring committed to
adding T3.6 here rather than a new file, since both share this module's
config and indexes; T3.7 follows the same reasoning — it only consumes
T3.6's output, so it belongs alongside it):

- **Index build (T3.5):** `build_retriever`/`build_all_retrievers` chunk
  `corpus_v1` under T3.4's locked scheme, index BM25 and dense, and
  confirm both are queryable via a smoke query — logging build time for
  the eventual cost/latency table (E6/M5).
- **RQ2 eval run (T3.6):** `score_retriever`/`summarize_metrics`/
  `compute_significance`/`run_rq2_eval` run every CRAGB v1 question
  through both indexed retrievers at every `k`, computing Recall/Hit/
  nDCG/MRR (`cragb.eval.metrics_retrieval`) with bootstrap 95% CIs
  (`cragb.eval.bootstrap`) and a paired Wilcoxon significance test
  between BM25 and dense at each `k` — the headline RQ2 table and
  figure (PLAN.md §7). `main(mode="eval")` also persists the raw
  per-question scores (`results/tables/retrieval_eval_per_question_v1.csv`)
  so T3.7 can slice them without ever rebuilding an index.
- **Per-type breakdown (T3.7):** `summarize_recall_by_type`/
  `h2_interaction_summary`/`plot_recall_per_type` slice T3.6's per-question
  scores by CRAGB's question `type` to test H2 (PLAN.md §2: dense should
  win paraphrased/semantic questions, BM25 competitive-or-better on
  lexical/attribute questions — no global winner, an *interaction*).
  `main(mode="by-type")` needs neither the corpus nor either index — it
  reads T3.6's already-computed per-question CSV directly, so it never
  touches the GPU or the dense stack at all.

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
from tqdm import tqdm

from cragb.eval.bootstrap import bootstrap_ci, paired_significance
from cragb.eval.chunking_study import collapse_chunk_ranking_to_parents
from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.metrics_retrieval import score_all
from cragb.retrieval.base import Retriever
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import chunk_corpus, load_chunking_config
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

_METRIC_NAMES = ("recall", "hit", "ndcg", "mrr")


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
# RQ2 eval run (T3.6)
# --------------------------------------------------------------------------


def score_retriever(
    retriever: Retriever,
    chunk_to_parent: dict[str, str],
    questions: list[RetrievalQuestion],
    k_values: tuple[int, ...],
    chunk_search_multiplier: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Score one already-indexed retriever against every question, at every k.

    Args:
        retriever: an already-`.index()`-ed `Retriever`.
        chunk_to_parent: maps every indexed chunk id to its parent review
            id, exactly as in `cragb.eval.chunking_study.run_scheme_recall`
            — chunk-level hits are collapsed to review-level hits before
            scoring, so results mean the same thing regardless of
            `configs/chunking.yaml`'s scheme.
        questions: pre-filtered scorable questions
            (`cragb.eval.cragb_questions.filter_scorable`) — a question
            with empty `relevant_ids` makes `score_all` raise, by design.
        k_values: the `k`s to score at.
        chunk_search_multiplier: raw chunks requested per query is
            `max(k_values) * chunk_search_multiplier` before collapsing
            to unique parents (see `run_scheme_recall`'s docstring for
            why). Defaults to `1` because T3.4 locked `whole_review`
            (one chunk per review, so `chunk_id == parent_doc_id`
            already) — a future re-locked multi-chunk scheme would need
            a larger multiplier here, the same way T3.4's chunking study
            needed one.
        show_progress: show a `tqdm` progress bar over questions.

    Returns:
        Long-format `[question_id, type, k, recall, hit, ndcg, mrr]`,
        one row per (question, k) pair.
    """
    chunk_search_k = max(k_values) * chunk_search_multiplier

    rows: list[dict[str, object]] = []
    iterator = tqdm(questions, desc="scoring", disable=not show_progress)
    for question in iterator:
        hits = retriever.search(question.question, k=chunk_search_k)
        ranked_parents = collapse_chunk_ranking_to_parents(
            [hit.doc_id for hit in hits], chunk_to_parent
        )
        scores_by_k = score_all(ranked_parents, question.relevant_ids, list(k_values))
        for k, metric_scores in scores_by_k.items():
            rows.append(
                {"question_id": question.id, "type": question.type, "k": k, **metric_scores}
            )

    return pd.DataFrame(rows)


def summarize_metrics(
    per_question: pd.DataFrame,
    retriever: str,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng=None,
) -> pd.DataFrame:
    """Per-`k` mean + bootstrap CI for every metric, from `score_retriever`'s output.

    Args:
        per_question: output of `score_retriever` for one retriever.
        retriever: label to attach as the `retriever` column.
        n_boot, alpha, rng: forwarded to `cragb.eval.bootstrap.bootstrap_ci`
            for every (metric, k) pair.

    Returns:
        One row per distinct `k`, columns
        `[retriever, k, n_questions, recall_mean, recall_ci_lo,
        recall_ci_hi, hit_mean, ..., mrr_ci_hi]`.
    """
    rows: list[dict[str, object]] = []
    for k, group in per_question.groupby("k"):
        row: dict[str, object] = {"retriever": retriever, "k": k, "n_questions": len(group)}
        for metric in _METRIC_NAMES:
            scores = group[metric].tolist()
            lo, hi = bootstrap_ci(scores, n_boot=n_boot, alpha=alpha, rng=rng)
            row[f"{metric}_mean"] = sum(scores) / len(scores)
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def compute_significance(
    per_question_bm25: pd.DataFrame,
    per_question_dense: pd.DataFrame,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    """Paired Wilcoxon p-value per (metric, k) between BM25 and dense.

    Questions are the paired unit (PLAN.md §8): each question's BM25
    score at a given `k` is paired with that *same* question's dense
    score at that `k`, not compared as independent samples.

    Args:
        per_question_bm25: `score_retriever`'s output for BM25.
        per_question_dense: `score_retriever`'s output for dense.
        k_values: the `k`s to test at.

    Returns:
        One row per `k`, columns
        `[k, recall_wilcoxon_p, hit_wilcoxon_p, ndcg_wilcoxon_p, mrr_wilcoxon_p]`.

    Raises:
        ValueError: if the two retrievers were scored on different
            question sets at some `k` — they must be pairable 1:1 by
            `question_id`.
    """
    rows: list[dict[str, object]] = []
    for k in k_values:
        bm25_k = per_question_bm25.loc[per_question_bm25["k"] == k].set_index("question_id")
        dense_k = per_question_dense.loc[per_question_dense["k"] == k].set_index("question_id")
        if set(bm25_k.index) != set(dense_k.index):
            raise ValueError(
                f"BM25 and dense were scored on different questions at k={k}; "
                "cannot pair for a significance test."
            )
        row: dict[str, object] = {"k": k}
        for metric in _METRIC_NAMES:
            row[f"{metric}_wilcoxon_p"] = paired_significance(
                bm25_k[metric].tolist(), dense_k.loc[bm25_k.index, metric].tolist()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_rq2_eval(
    corpus: pd.DataFrame,
    config: RetrievalEvalConfig,
    questions: list[RetrievalQuestion],
    smoke_query: str = "does this run true to size",
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng=None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, IndexBuildReport]]:
    """Run the full RQ2 comparison end-to-end: build both indexes, score, summarize.

    The single entry point for T3.6: builds BM25 and dense indexes
    (T3.5's `build_all_retrievers`, unchanged — reused rather than
    duplicated), scores every question in `questions` against both at
    every `k` in `config.k_values`, and returns the headline table plus
    the raw per-question scores (needed by T3.7's per-type breakdown).

    Args:
        corpus: `corpus_v1`-shaped DataFrame.
        config: a `RetrievalEvalConfig`.
        questions: pre-filtered scorable questions (see `score_retriever`).
        smoke_query: forwarded to `build_all_retrievers`.
        n_boot, alpha, rng: forwarded to `summarize_metrics`. Pass a
            seeded `rng` (e.g. via `cragb.utils.seeds.set_global_seed`)
            for a fully reproducible run.
        show_progress: show a `tqdm` progress bar per retriever.

    Returns:
        `(rq2_table, per_question_by_retriever, build_reports)`:
        - `rq2_table`: one row per (retriever, k) — Recall/Hit/nDCG/MRR
          mean + 95% CI, `n_questions`, and a paired Wilcoxon p-value per
          metric (duplicated across both retrievers' rows at a given
          `k`, since significance is a property of the *pair*, not of
          either retriever alone — this keeps the CSV self-contained
          without requiring a join to read the significance column).
        - `per_question_by_retriever`: `{"bm25": df, "dense": df}`, each
          `score_retriever`'s raw long-format output.
        - `build_reports`: `{"bm25": IndexBuildReport, "dense": IndexBuildReport}`.
    """
    built = build_all_retrievers(corpus, config, smoke_query=smoke_query)

    chunking_config = load_chunking_config(config.chunking_config_path)
    chunks = chunk_corpus(corpus, chunking_config)
    chunk_to_parent = dict(zip(chunks["chunk_id"], chunks["parent_doc_id"]))

    per_question: dict[str, pd.DataFrame] = {}
    for name, (retriever, _report) in built.items():
        logger.info("scoring %s against %d questions", name, len(questions))
        per_question[name] = score_retriever(
            retriever, chunk_to_parent, questions, config.k_values, show_progress=show_progress
        )

    summaries = {
        name: summarize_metrics(df, retriever=name, n_boot=n_boot, alpha=alpha, rng=rng)
        for name, df in per_question.items()
    }
    significance = compute_significance(
        per_question["bm25"], per_question["dense"], config.k_values
    )

    rq2_table = (
        pd.concat(summaries.values(), ignore_index=True)
        .merge(significance, on="k", how="left")
        .sort_values(["k", "retriever"])
        .reset_index(drop=True)
    )

    build_reports = {name: report for name, (_retriever, report) in built.items()}
    return rq2_table, per_question, build_reports


def plot_recall_at_k(rq2_table: pd.DataFrame, out_path: str | Path) -> None:
    """Recall@k-vs-k line chart, both retrievers overlaid with 95% CI error bars.

    PLAN.md §7 figure #2 — the RQ2 headline figure.

    Args:
        rq2_table: `run_rq2_eval`'s summary table (must have
            `retriever`, `k`, `recall_mean`, `recall_ci_lo`,
            `recall_ci_hi` columns).
        out_path: where to save the figure (parent directories created
            as needed), absolute or relative to the repo root.
    """
    import matplotlib.pyplot as plt  # lazy: plotting is optional for callers that only need the table

    fig, ax = plt.subplots(figsize=(7, 5))
    for retriever, group in rq2_table.groupby("retriever"):
        group = group.sort_values("k")
        ax.errorbar(
            group["k"],
            group["recall_mean"],
            yerr=[
                group["recall_mean"] - group["recall_ci_lo"],
                group["recall_ci_hi"] - group["recall_mean"],
            ],
            marker="o",
            capsize=3,
            label=retriever,
        )
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("RQ2: Recall@k, BM25 vs dense (CRAGB v1, 95% bootstrap CI)")
    ax.set_xticks(sorted(rq2_table["k"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    resolved_out_path = resolve_path(out_path)
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-question-type breakdown (T3.7)
# --------------------------------------------------------------------------


def summarize_recall_by_type(
    per_question_by_retriever: dict[str, pd.DataFrame],
    k: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng=None,
) -> pd.DataFrame:
    """Recall@`k` mean + bootstrap CI, grouped by CRAGB question `type`, per retriever.

    Slices at a single `k` (PLAN.md §7's headline is Recall@5) rather
    than every `k` in `config.k_values`: the per-type breakdown is meant
    to be read as one bar chart, and a `type` x `retriever` x `k` cube
    would be harder to present than it is to compute — pick the k that
    matters for the report, not every k that's cheap to include.

    Args:
        per_question_by_retriever: `{"bm25": df, "dense": df}`, each in
            the long format `score_retriever`/`run_rq2_eval` produce
            (`[question_id, type, k, recall, hit, ndcg, mrr]`).
        k: the single `k` to slice at.
        n_boot, alpha, rng: forwarded to `cragb.eval.bootstrap.bootstrap_ci`.

    Returns:
        `[retriever, type, k, recall_mean, recall_ci_lo, recall_ci_hi, n_questions]`,
        one row per (retriever, type).

    Raises:
        ValueError: if `k` is not present in a retriever's per-question data.
    """
    rows: list[dict[str, object]] = []
    for retriever, df in per_question_by_retriever.items():
        df_k = df.loc[df["k"] == k]
        if df_k.empty:
            raise ValueError(
                f"k={k} not found in per-question data for retriever={retriever!r}; "
                f"available k values: {sorted(df['k'].unique())}"
            )
        for qtype, group in df_k.groupby("type"):
            scores = group["recall"].tolist()
            lo, hi = bootstrap_ci(scores, n_boot=n_boot, alpha=alpha, rng=rng)
            rows.append(
                {
                    "retriever": retriever,
                    "type": qtype,
                    "k": k,
                    "recall_mean": sum(scores) / len(scores),
                    "recall_ci_lo": lo,
                    "recall_ci_hi": hi,
                    "n_questions": len(scores),
                }
            )
    return pd.DataFrame(rows).sort_values(["type", "retriever"]).reset_index(drop=True)


def h2_interaction_summary(by_type_table: pd.DataFrame) -> pd.DataFrame:
    """Widen the per-type table to one row per type, naming which retriever leads.

    A quick, printable read on H2 (PLAN.md §2: "dense ≥ BM25 on
    paraphrased/semantic questions; BM25 competitive or better on
    exact-attribute/lexical questions... expect a query-type interaction,
    not a global winner") — whether at least one type actually favors
    BM25 is the specific thing T3.7's validation checklist asks to
    confirm (or honestly flag if it doesn't hold).

    Args:
        by_type_table: output of `summarize_recall_by_type`.

    Returns:
        `[type, bm25_recall_mean, dense_recall_mean, leader]`, `leader`
        one of `"bm25"`, `"dense"`, or `"tie"` (means equal within
        floating-point tolerance).
    """
    wide = by_type_table.pivot(index="type", columns="retriever", values="recall_mean")

    def _leader(row: pd.Series) -> str:
        if abs(row["bm25"] - row["dense"]) < 1e-9:
            return "tie"
        return "bm25" if row["bm25"] > row["dense"] else "dense"

    wide["leader"] = wide.apply(_leader, axis=1)
    wide = wide.rename(columns={"bm25": "bm25_recall_mean", "dense": "dense_recall_mean"})
    return wide.reset_index()[["type", "bm25_recall_mean", "dense_recall_mean", "leader"]]


def plot_recall_per_type(by_type_table: pd.DataFrame, out_path: str | Path, k: int) -> None:
    """Grouped bar chart: Recall@`k` by question type, BM25/dense side by side.

    PLAN.md §7 figure #3 — the per-question-type retrieval bar chart.

    Args:
        by_type_table: output of `summarize_recall_by_type`.
        out_path: where to save the figure, absolute or relative to the
            repo root.
        k: used only for the title/axis label (the table is already
            sliced to one `k`).
    """
    import matplotlib.pyplot as plt  # lazy: see plot_recall_at_k
    import numpy as np

    types = sorted(by_type_table["type"].unique())
    retrievers = ["bm25", "dense"]
    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, retriever in enumerate(retrievers):
        sub = by_type_table.loc[by_type_table["retriever"] == retriever].set_index("type").loc[types]
        offset = (i - 0.5) * width
        yerr = [
            sub["recall_mean"] - sub["recall_ci_lo"],
            sub["recall_ci_hi"] - sub["recall_mean"],
        ]
        ax.bar(x + offset, sub["recall_mean"], width, yerr=yerr, capsize=3, label=retriever)

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"RQ2: Recall@{k} by question type, BM25 vs dense (CRAGB v1, 95% bootstrap CI)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    resolved_out_path = resolve_path(out_path)
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


_PER_QUESTION_PATH = "results/tables/retrieval_eval_per_question_v1.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 + dense retrieval indexes over corpus_v1 (T3.5), "
        "run the full RQ2 eval against CRAGB v1 (T3.6), or slice it by question "
        "type (T3.7)."
    )
    parser.add_argument("--config", default="configs/retrieval_eval.yaml")
    parser.add_argument("--smoke-query", default="does this run true to size")
    parser.add_argument(
        "--mode",
        choices=["build", "eval", "by-type"],
        default="build",
        help="'build' (default, T3.5): index both retrievers, smoke-test them, "
        "write the build report. 'eval' (T3.6): run the full RQ2 comparison "
        "against CRAGB v1 and write results/tables/retrieval_eval_v1.csv + "
        f"{_PER_QUESTION_PATH} + reports/figures/recall_at_k_bm25_vs_dense.png. "
        "'by-type' (T3.7): slice T3.6's per-question output by CRAGB question "
        "type and write results/tables/retrieval_eval_by_type_v1.csv + "
        "reports/figures/recall_per_type.png — reads "
        f"{_PER_QUESTION_PATH} directly, so it needs neither the corpus, an "
        "index, nor the GPU (run 'eval' first).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="override config.seed for bootstrap CIs"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k to slice the per-type breakdown at (mode=by-type only; PLAN.md "
        "§7's headline k)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_retrieval_eval_config(args.config)

    from cragb.utils.seeds import set_global_seed

    seed = args.seed if args.seed is not None else config.seed
    seed_state = set_global_seed(seed)

    if args.mode == "by-type":
        per_question_path = resolve_path(_PER_QUESTION_PATH)
        if not per_question_path.is_file():
            raise FileNotFoundError(
                f"{per_question_path} not found; run `--mode eval` first (T3.6) "
                "to produce per-question scores before the by-type breakdown (T3.7)."
            )
        per_question_all = pd.read_csv(per_question_path)
        per_question_by_retriever = {
            name: group.drop(columns="retriever").reset_index(drop=True)
            for name, group in per_question_all.groupby("retriever")
        }
        logger.info(
            "loaded %s: %d rows across %d retriever(s)",
            per_question_path,
            len(per_question_all),
            len(per_question_by_retriever),
        )

        by_type_table = summarize_recall_by_type(
            per_question_by_retriever, k=args.k, rng=seed_state.numpy_rng
        )

        by_type_path = resolve_path("results/tables/retrieval_eval_by_type_v1.csv")
        by_type_path.parent.mkdir(parents=True, exist_ok=True)
        by_type_table.to_csv(by_type_path, index=False)
        logger.info("wrote per-type table (%d rows) to %s", len(by_type_table), by_type_path)

        fig_path = "reports/figures/recall_per_type.png"
        plot_recall_per_type(by_type_table, fig_path, k=args.k)
        logger.info("wrote per-type figure to %s", resolve_path(fig_path))

        h2_summary = h2_interaction_summary(by_type_table)
        logger.info("H2 interaction summary (leader per type):\n%s", h2_summary.to_string(index=False))
        return

    corpus = pd.read_parquet(resolve_path(config.corpus_in), columns=["text"])
    logger.info("loaded corpus_in=%s: %d reviews", config.corpus_in, len(corpus))

    if args.mode == "build":
        built = build_all_retrievers(corpus, config, smoke_query=args.smoke_query)

        out_path = resolve_path(config.build_report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reports = [report.to_dict() for _retriever, report in built.values()]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        logger.info("wrote build report (%d retriever(s)) to %s", len(reports), out_path)
        return

    # args.mode == "eval"
    from cragb.eval.cragb_questions import filter_scorable, load_retrieval_questions

    questions = filter_scorable(load_retrieval_questions(config.questions_in))
    logger.info("loaded %s: %d scorable questions", config.questions_in, len(questions))

    rq2_table, per_question, _build_reports = run_rq2_eval(
        corpus, config, questions, smoke_query=args.smoke_query, rng=seed_state.numpy_rng
    )

    table_path = resolve_path("results/tables/retrieval_eval_v1.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rq2_table.to_csv(table_path, index=False)
    logger.info("wrote RQ2 table (%d rows) to %s", len(rq2_table), table_path)

    per_question_combined = pd.concat(
        [df.assign(retriever=name) for name, df in per_question.items()], ignore_index=True
    )
    per_question_path = resolve_path(_PER_QUESTION_PATH)
    per_question_path.parent.mkdir(parents=True, exist_ok=True)
    per_question_combined.to_csv(per_question_path, index=False)
    logger.info(
        "wrote per-question scores (%d rows) to %s", len(per_question_combined), per_question_path
    )

    fig_path = "reports/figures/recall_at_k_bm25_vs_dense.png"
    plot_recall_at_k(rq2_table, fig_path)
    logger.info("wrote Recall@k figure to %s", resolve_path(fig_path))


if __name__ == "__main__":
    main()
