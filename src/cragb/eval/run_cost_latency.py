"""End-to-end latency harness: live, cache-bypassed timed run per arm
(T5.5; PLAN.md §3 E6, §8 G4, M5.md T5.5).

The one G4 number that cannot come from anything already on disk: how long
a real question actually takes, arm by arm — retrieval + context assembly
+ generation, wall-clock, as a user would feel it. T5.3 already timed
retrieval search in isolation and T5.4 already recovered token counts from
the disk cache; this module is deliberately the odd one out, because a
cache hit is served in microseconds regardless of how slow the original
call was, and timing hits would make every number here fiction. Every
generation call in this module goes through
`GroqClient.complete_with_usage(..., bypass_cache=True)` (T5.5's addition
to `cragb.generate.api_clients`), which skips the disk cache on both the
read and the write side — see that method's docstring for why the write
side matters too (a live sample at temperature > 0 can legitimately return
a different completion than the one already cached from T4b.2's canonical
run, and overwriting that would be worse than not caching this call).

Three arms, one shared BM25 index for the two RAG arms (same
retriever/chunking/k the RQ1 control already relies on, PLAN.md §2) built
*once*, outside the timed region — index build cost is T5.3's job, not
this one's; what's timed per question is only `build_context`'s retrieval
+ assembly step and the LLM call itself. `closed_book` has no retrieval
step at all, so its `retrieval_ms` is reported as `0.0`, not omitted —
keeping one row shape across all three arms.

Testability follows this project's established `chat_fn`/`usage_fn`
injection pattern (`cragb.generate.grounded_qa`, `cragb.eval.cost_model`):
`run_one_closed_book_question`/`run_one_grounded_question` take a
`usage_fn: UsageFn` rather than a concrete `GroqClient`, so their timing
and row-shape logic is unit-testable with a fake that returns instantly —
no network, no API key, no real latency needed to test that the *pipeline*
is wired correctly. Only `main()` constructs a real `GroqClient` bound to
`bypass_cache=True`.

Question selection: a fixed, seeded subset stratified across CRAGB's
taxonomy `type` (default 15 of 60) rather than the first N — question
type changes context length and therefore latency, and the first N rows
of `cragb_v1.jsonl` are not type-balanced (see
`cragb.eval.cragb_questions`). `stratified_question_sample`'s per-type
quotas are a pure function of `n` and the type distribution (the
largest-remainder method, tie-broken alphabetically) — only *which*
questions within a type are drawn depends on the seed, so the sample size
per type is reproducible even before you check the seed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cragb.eval.cragb_questions import RetrievalQuestion, load_retrieval_questions
from cragb.generate.api_clients import CompletionResult, GroqClient
from cragb.generate.closed_book_qa import render_prompt as render_closed_book_prompt
from cragb.generate.context_builder import (
    CorpusLookup,
    build_context,
    build_corpus_lookup,
    index_bm25_retriever,
)
from cragb.generate.grounded_qa import load_prompt_template
from cragb.generate.grounded_qa import render_prompt as render_grounded_prompt
from cragb.retrieval.base import Retriever
from cragb.retrieval.chunking import load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.timing import Timer, latency_stats

logger = logging.getLogger(__name__)

DEFAULT_N_QUESTIONS = 15
DEFAULT_SEED = 42

UsageFn = Callable[[list[dict[str, str]]], CompletionResult]

# (arm label, generation config) for the three RQ0/RQ1 arms. Order matters only in that
# both RAG arms share one BM25 index built once, before either arm's questions are timed.
_CLOSED_BOOK_CONFIG = "configs/closed_book_qa.yaml"
_RAG_ARM_CONFIGS: tuple[tuple[str, str], ...] = (
    ("rag_small", "configs/grounded_qa.yaml"),
    ("rag_large", "configs/grounded_qa_large.yaml"),
)


# --------------------------------------------------------------------------
# Stratified question sample
# --------------------------------------------------------------------------


def stratified_question_sample(
    questions: list[RetrievalQuestion], n: int, rng: np.random.Generator
) -> list[RetrievalQuestion]:
    """Pick `n` questions, proportionally stratified by `.type`.

    Per-type quotas come from the largest-remainder (Hamilton) apportionment
    method — `floor(n * type_count / total)`, with the `n - sum(quotas)`
    leftover seats given to the types with the largest fractional remainder,
    ties broken alphabetically by type name. This makes the quotas a pure
    function of `n` and the type distribution, not of `rng`: two calls with
    different seeds draw a different *subset* but always the same *count*
    per type, so `n` alone determines how many types get represented and by
    how much.

    Args:
        questions: the full question pool (CRAGB v1: 60 questions, 7 types).
        n: how many to sample in total.
        rng: seeded generator controlling which questions are drawn within
            each type's quota.

    Returns:
        `n` questions (fewer only if a type's quota exceeds its own pool
        size, which cannot happen for `n <= len(questions)` under this
        apportionment), grouped by type (alphabetical), quota order within
        a type following `rng`'s draw.

    Raises:
        ValueError: if `n` is not positive, or exceeds `len(questions)`.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(questions):
        raise ValueError(f"n={n} exceeds the available pool of {len(questions)} questions")

    by_type: dict[str, list[RetrievalQuestion]] = {}
    for q in questions:
        by_type.setdefault(q.type, []).append(q)

    total = len(questions)
    exact_quotas = {t: n * len(qs) / total for t, qs in by_type.items()}
    quotas = {t: int(q) for t, q in exact_quotas.items()}
    remainder = n - sum(quotas.values())
    # Largest fractional remainder first; alphabetical tie-break for full determinism.
    remainder_order = sorted(by_type, key=lambda t: (-(exact_quotas[t] - quotas[t]), t))
    for t in remainder_order[:remainder]:
        quotas[t] += 1

    sample: list[RetrievalQuestion] = []
    for t in sorted(by_type):
        pool = by_type[t]
        k = min(quotas[t], len(pool))
        chosen_idx = rng.choice(len(pool), size=k, replace=False)
        sample.extend(pool[i] for i in sorted(chosen_idx))
    return sample


# --------------------------------------------------------------------------
# Per-question timed runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class E2ELatencyRow:
    """One question's timed pipeline run for one arm."""

    arm: str
    question_id: str
    model: str
    retrieval_ms: float
    generate_ms: float
    e2e_ms: float
    cached: bool
    run_timestamp: str

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "question_id": self.question_id,
            "model": self.model,
            "retrieval_ms": self.retrieval_ms,
            "generate_ms": self.generate_ms,
            "e2e_ms": self.e2e_ms,
            "cached": self.cached,
            "run_timestamp": self.run_timestamp,
        }


def run_one_closed_book_question(
    question: RetrievalQuestion, template: Template, usage_fn: UsageFn
) -> E2ELatencyRow:
    """Time one closed-book question: prompt render + generation, no retrieval.

    Args:
        question: the CRAGB question to answer.
        template: the closed-book prompt template.
        usage_fn: `GroqClient.complete_with_usage` (bound to
            `bypass_cache=True` in production), or a fake for tests.

    Returns:
        An `E2ELatencyRow` with `retrieval_ms=0.0`.
    """
    with Timer() as e2e_timer:
        prompt = render_closed_book_prompt(template, question.question)
        messages = [{"role": "user", "content": prompt}]
        with Timer() as generate_timer:
            result = usage_fn(messages)
    assert e2e_timer.elapsed_s is not None and generate_timer.elapsed_s is not None

    return E2ELatencyRow(
        arm="closed_book",
        question_id=question.id,
        model=result.model,
        retrieval_ms=0.0,
        generate_ms=generate_timer.elapsed_s * 1000,
        e2e_ms=e2e_timer.elapsed_s * 1000,
        cached=result.cached,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_one_grounded_question(
    arm: str,
    question: RetrievalQuestion,
    template: Template,
    retriever: Retriever,
    chunk_to_parent: dict[str, str],
    lookup: CorpusLookup,
    k: int,
    usage_fn: UsageFn,
) -> E2ELatencyRow:
    """Time one grounded-QA question: retrieval + context assembly + generation.

    `retriever` is assumed already indexed — index build time is T5.3's
    concern, not this per-question measurement's.

    Args:
        arm: `"rag_small"` or `"rag_large"` — carried through to the row.
        question: the CRAGB question to answer.
        template: that arm's grounded-QA prompt template.
        retriever, chunk_to_parent, lookup, k: forwarded to
            `cragb.generate.context_builder.build_context`.
        usage_fn: `GroqClient.complete_with_usage` (bound to
            `bypass_cache=True` in production), or a fake for tests.

    Returns:
        An `E2ELatencyRow`. `e2e_ms >= retrieval_ms + generate_ms` by
        construction: `e2e_timer` wraps both sub-timed regions plus the
        (fast, untimed-separately) prompt rendering between them.
    """
    with Timer() as e2e_timer:
        with Timer() as retrieval_timer:
            context = build_context(question.question, retriever, chunk_to_parent, lookup, k)
        prompt = render_grounded_prompt(template, question.question, context)
        messages = [{"role": "user", "content": prompt}]
        with Timer() as generate_timer:
            result = usage_fn(messages)
    assert (
        e2e_timer.elapsed_s is not None
        and retrieval_timer.elapsed_s is not None
        and generate_timer.elapsed_s is not None
    )

    return E2ELatencyRow(
        arm=arm,
        question_id=question.id,
        model=result.model,
        retrieval_ms=retrieval_timer.elapsed_s * 1000,
        generate_ms=generate_timer.elapsed_s * 1000,
        e2e_ms=e2e_timer.elapsed_s * 1000,
        cached=result.cached,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize_e2e_latency(per_question: pd.DataFrame) -> pd.DataFrame:
    """Per-arm latency summary (`results/tables/e2e_latency_v1.csv`'s exact shape).

    Percentiles reuse `cragb.utils.timing.latency_stats`'s convention
    (numpy's default linear interpolation), so this table's percentiles
    are computed the same way T5.3's retrieval-latency table's are.

    Args:
        per_question: `[arm, question_id, model, retrieval_ms, generate_ms,
            e2e_ms, cached, run_timestamp]`, as produced by concatenating
            `E2ELatencyRow.to_dict()` rows across all arms/questions.

    Returns:
        One row per arm: `arm, model, n, retrieval_ms_p50, generate_ms_p50,
        e2e_ms_p50, e2e_ms_p95, e2e_ms_mean, run_timestamp,
        cache_bypassed`. `cache_bypassed` is `True` iff every row in that
        arm has `cached=False` — the direct, data-derived answer to "did
        the bypass actually work", independent of what any CLI flag claims.

    Raises:
        ValueError: if `per_question` is empty.
    """
    if per_question.empty:
        raise ValueError("per_question must be non-empty")

    rows: list[dict] = []
    for arm, group in per_question.groupby("arm", sort=False):
        e2e_stats = latency_stats((group["e2e_ms"] / 1000).tolist())
        generate_stats = latency_stats((group["generate_ms"] / 1000).tolist())
        retrieval_stats = latency_stats((group["retrieval_ms"] / 1000).tolist())
        rows.append(
            {
                "arm": arm,
                "model": group["model"].iloc[0],
                "n": len(group),
                "retrieval_ms_p50": retrieval_stats["p50"] * 1000,
                "generate_ms_p50": generate_stats["p50"] * 1000,
                "e2e_ms_p50": e2e_stats["p50"] * 1000,
                "e2e_ms_p95": e2e_stats["p95"] * 1000,
                "e2e_ms_mean": e2e_stats["mean"] * 1000,
                "run_timestamp": group["run_timestamp"].iloc[0],
                "cache_bypassed": bool((~group["cached"]).all()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Orchestration + CLI
# --------------------------------------------------------------------------


def _client_and_template(config_path: str, bypass_cache: bool) -> tuple[UsageFn, Template]:
    cfg = load_config(config_path)
    provider = cfg["provider"]
    client = GroqClient(
        model=provider["model"],
        api_base=provider["api_base"],
        api_key_env=provider["api_key_env"],
        temperature=provider["temperature"],
        max_tokens=provider["max_tokens"],
        reasoning_effort=provider.get("reasoning_effort"),
        timeout_s=provider["timeout_s"],
        max_retries=provider["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )
    template = load_prompt_template(cfg["paths"]["prompt_template"])

    def usage_fn(messages: list[dict[str, str]]) -> CompletionResult:
        return client.complete_with_usage(messages, bypass_cache=bypass_cache)

    return usage_fn, template


def run_e2e_latency(
    questions: list[RetrievalQuestion], corpus: pd.DataFrame, bypass_cache: bool
) -> pd.DataFrame:
    """Run all three arms against `questions` and return the raw per-question rows.

    Args:
        questions: questions to time (e.g. `stratified_question_sample`'s output).
        corpus: `corpus_v1`-shaped DataFrame (full columns — `build_corpus_lookup`
            needs `has_image`, not just `text`).
        bypass_cache: forwarded to every `GroqClient.complete_with_usage` call.
            `False` is a fast, free, wiring-only dry run through the disk
            cache (every question already answered in T4b.2 is a cache hit,
            so the "latencies" are meaningless — useful only for confirming
            the pipeline runs end-to-end without spending quota). `True` is
            the real measurement this harness exists for.

    Returns:
        `[arm, question_id, model, retrieval_ms, generate_ms, e2e_ms,
        cached, run_timestamp]`, `closed_book` rows first, then
        `rag_small`, then `rag_large`.
    """
    rows: list[dict] = []

    cb_usage_fn, cb_template = _client_and_template(_CLOSED_BOOK_CONFIG, bypass_cache)
    for q in questions:
        rows.append(run_one_closed_book_question(q, cb_template, cb_usage_fn).to_dict())

    # One BM25 index shared by both RAG arms -- same chunking/retriever/k the RQ1
    # control relies on (PLAN.md §2); index build time is T5.3's concern, not timed here.
    chunking_config = load_chunking_config(load_config(_RAG_ARM_CONFIGS[0][1])["retrieval"]["chunking_config"])
    retriever, chunk_to_parent = index_bm25_retriever(corpus, chunking_config)
    lookup = build_corpus_lookup(corpus)

    for arm, config_path in _RAG_ARM_CONFIGS:
        usage_fn, template = _client_and_template(config_path, bypass_cache)
        k = load_config(config_path)["retrieval"]["k"]
        for q in questions:
            rows.append(
                run_one_grounded_question(
                    arm, q, template, retriever, chunk_to_parent, lookup, k, usage_fn
                ).to_dict()
            )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end (retrieval + generation) latency per arm "
        "(closed-book, RAG-small, RAG-large) over a stratified sample of CRAGB v1 "
        "questions (T5.5; PLAN.md §3 E6, §8 G4)."
    )
    parser.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--questions-in", default="benchmark/cragb_v1.jsonl")
    parser.add_argument("--corpus-in", default="data/processed/corpus_v1.parquet")
    parser.add_argument("--out", default="results/tables/e2e_latency_v1.csv")
    parser.add_argument(
        "--per-question-out", default="results/tables/e2e_latency_per_question_v1.csv"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the disk cache and issue live API calls for every question -- the "
        "real measurement this harness exists for. Without this flag, calls go through "
        "the normal cache (every T4b.2 question is already cached, so this is a fast, "
        "free, zero-quota dry run of the pipeline wiring, not a latency measurement).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    if not args.no_cache:
        logger.warning(
            "running WITHOUT --no-cache: every already-answered question is a cache hit, "
            "so these are NOT real end-to-end latencies -- this is a wiring dry run only."
        )

    all_questions = load_retrieval_questions(args.questions_in)
    rng = np.random.default_rng(args.seed)
    questions = stratified_question_sample(all_questions, args.n_questions, rng)
    logger.info(
        "sampled %d questions across %d taxonomy types (seed=%d)",
        len(questions),
        len({q.type for q in questions}),
        args.seed,
    )

    corpus = pd.read_parquet(resolve_path(args.corpus_in))
    logger.info("loaded corpus_in=%s: %d reviews", args.corpus_in, len(corpus))

    per_question = run_e2e_latency(questions, corpus, bypass_cache=args.no_cache)

    per_question_path = resolve_path(args.per_question_out)
    per_question_path.parent.mkdir(parents=True, exist_ok=True)
    per_question.to_csv(per_question_path, index=False)
    logger.info("wrote %d per-question rows to %s", len(per_question), per_question_path)

    summary = summarize_e2e_latency(per_question)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("wrote %d summary rows to %s", len(summary), out_path)
    logger.info(
        "NOTE: single-client, sequential, free-tier measurement -- Groq's shared free "
        "tier is not a latency SLA; do not present these as production numbers."
    )


if __name__ == "__main__":
    main()
