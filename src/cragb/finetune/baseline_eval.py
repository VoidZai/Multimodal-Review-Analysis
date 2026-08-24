"""Untuned local base-model baseline on CRAGB + the behaviour probe (T7.8; PLAN.md §3 E8,
§10, M7.md T7.8).

Answers the one question PLAN.md §10's go/no-go rule ("proceed only if a pilot LoRA
shows >= X improvement on citation/abstention") cannot be written without: what does the
*untuned* local model actually do first? `results/tables/grounded_qa_validity_v1.csv`'s
1.0 citation-validity and `judge_scores_v1.csv`'s 4.93 mean faithfulness are real numbers,
but they describe `openai/gpt-oss-20b` running on Groq (T4a/T4b) -- quoting them as this
project's fine-tuning baseline would be a category error, the same way comparing a 3B
local model's cost to a 120B API model's cost would be.

**This module is thin plumbing, deliberately.** Every metric here is computed by the
exact function that already produced the M4a/M4b numbers -- `cragb.eval.citation_validity
.score_transcripts`/`.summarize`, `cragb.eval.judge.score_answer`,
`cragb.eval.run_cost_latency.run_one_grounded_question`,
`cragb.eval.run_answer_generation.validate_full_run` -- pointed at T7.7's `LocalHFClient`
instead of `GroqClient`. If a metric needed reimplementing here, that would mean T7.7's
interface parity claim was wrong; nothing in this module reimplements one.

**Two evaluation sets, scored the same way, reported as two rows of one table:**

- **CRAGB's 60 questions**, retrieved fresh via the *identical* BM25/k=5 setup RAG-small
  uses (`assert_retrieval_matches_rag_small` proves this at runtime, not just by config
  inspection -- a silent retrieval drift here would confound the entire comparison and
  make the CI meaningless). Judged against CRAGB's own human reference answers, exactly
  as T4b.5 does.
- **The behaviour probe** (`data/finetune/probe.jsonl`, T7.6) -- no retrieval needed
  (`TrainingExample.context_text`/`.source_doc_ids` already carry a full rendered
  context), and abstention accuracy here is the number that actually means something:
  CRAGB itself has only 2 abstention questions (PLAN.md §14.2), too few to support a
  go/no-go threshold; the probe was built stratified ~40/~40 answerable/abstention for
  exactly this reason. Judged against **each probe example's own stored `answer`** as the
  reference -- unlike T7.5's *candidate == reference* trick (no independent ground truth
  existed there at all), a probe example's `answer` genuinely is an independent reference
  here: it is T7.3's teacher's (or T7.4's constructed) answer, and the transcript being
  scored is the *local* model's own, freshly-generated, independent attempt at the same
  question. Every probe transcript is judged, abstentions included -- exactly how CRAGB's
  own 2 abstention questions are already judged in T4b.5 -- because `answer_judge_v1.md`'s
  own "Special rule: abstention" is precisely the instrument that turns "the model
  hedged instead of abstaining" into a low, visible faithfulness score, which is one of
  the exact failure modes this baseline exists to surface.

**Generation latency is measured on a stratified subset, not every question** (T5.5's own
`run_one_grounded_question`, reused directly, bound to `bypass_cache=True` per PLAN.md
§14.5's lesson -- a cache hit's near-zero replay time would make every number here
fiction) -- a local 4-bit forward pass costs tens of seconds per call on this hardware
(T7.7's live check), so timing all 60+probe questions the way T5.5 could for a fast Groq
API call is not a reasonable use of wall-clock time; `configs/finetune_baseline.yaml`'s
`latency.n_*_questions` caps it, mirroring `run_cost_latency.DEFAULT_N_QUESTIONS`.

Usage:
    python -m cragb.finetune.baseline_eval --config configs/finetune_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cragb.bench.reference_answers import load_reference_answers
from cragb.eval.bootstrap import bootstrap_ci
from cragb.eval.citation_validity import (
    TranscriptScore,
    gold_relevant_ids_by_question,
    load_expected_abstentions,
    score_transcripts,
    summarize,
)
from cragb.eval.cragb_questions import load_retrieval_questions
from cragb.eval.judge import JudgeScore, score_answer
from cragb.eval.run_answer_generation import validate_full_run
from cragb.eval.run_cost_latency import stratified_question_sample
from cragb.eval.run_cost_latency import run_one_grounded_question as _run_one_grounded_question_timed
from cragb.eval.run_judge_eval import score_transcripts as score_transcripts_with_judge
from cragb.finetune.local_client import LocalHFClient
from cragb.finetune.schema import TrainingExample, load_training_examples_jsonl
from cragb.generate.context_builder import (
    ContextBlock,
    build_corpus_lookup,
    index_bm25_retriever,
)
from cragb.generate.grounded_qa import (
    GroundedQATranscript,
    generate_answer,
    load_prompt_template,
    load_transcripts_jsonl,
    run_grounded_qa,
)
from cragb.retrieval.chunking import load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed
from cragb.utils.timing import latency_stats

logger = logging.getLogger(__name__)

CRAGB_SOURCE = "cragb_60"
PROBE_SOURCE = "probe"


# --------------------------------------------------------------------------
# Retrieval-parity guard
# --------------------------------------------------------------------------


def assert_retrieval_matches_rag_small(
    transcripts: list[GroundedQATranscript],
    rag_small_transcripts_path: str | Path,
) -> None:
    """Prove this run's retrieval is byte-identical to the RAG-small arm's, question for
    question -- if it isn't, the generator comparison this whole module exists to make is
    confounded (a different context shown to the model explains any score difference at
    least as well as the generator does) and the resulting CI would be meaningless.

    Args:
        transcripts: this baseline's generated transcripts (any subset of CRAGB
            question ids is fine -- only ids present in both this list and
            `rag_small_transcripts_path` are compared).
        rag_small_transcripts_path: `results/tables/answer_gen_rag_small_v1.jsonl`
            (T4b.2's canonical RAG-small run).

    Raises:
        ValueError: if any shared question id's retrieved `context.doc_ids` differ, or if
            no question ids are shared at all (a config/path mistake, not a real
            comparison).
    """
    rag_small = {t.question_id: t for t in load_transcripts_jsonl(rag_small_transcripts_path)}
    shared_ids = [t.question_id for t in transcripts if t.question_id in rag_small]
    if not shared_ids:
        raise ValueError(
            f"No question ids in common between this run's transcripts and "
            f"{rag_small_transcripts_path!r} -- nothing to compare against."
        )

    mismatches = []
    for t in transcripts:
        if t.question_id not in rag_small:
            continue
        expected_doc_ids = rag_small[t.question_id].context.doc_ids
        if t.context.doc_ids != expected_doc_ids:
            mismatches.append((t.question_id, t.context.doc_ids, expected_doc_ids))

    if mismatches:
        lines = "\n".join(f"  {qid}: got {got}, expected {expected}" for qid, got, expected in mismatches)
        raise ValueError(
            f"Retrieval mismatch against RAG-small on {len(mismatches)}/{len(shared_ids)} "
            f"shared question(s) -- this baseline is not comparable to RAG-small:\n{lines}"
        )
    logger.info("Retrieval parity confirmed against RAG-small on all %d shared question(s).", len(shared_ids))


# --------------------------------------------------------------------------
# Probe-set judge scoring (no CRAGB reference exists -- see module docstring)
# --------------------------------------------------------------------------


def score_probe_transcript(
    transcript: GroundedQATranscript,
    example: TrainingExample,
    template: Template,
    chat_fn,
) -> JudgeScore:
    """Judge one probe transcript, using `example.answer` (T7.3's teacher's or T7.4's
    constructed answer) as the reference -- an independent, genuine reference for the
    *local* model's own freshly-generated `transcript.answer_text`, not a self-comparison
    (see module docstring for why this differs from T7.5's candidate==reference design).
    """
    return score_answer(
        transcript.question, transcript.context.text, transcript.answer_text, example.answer, template, chat_fn
    )


def score_probe_transcripts(
    transcripts: list[GroundedQATranscript],
    examples_by_id: dict[str, TrainingExample],
    template: Template,
    chat_fn,
) -> list[JudgeScore]:
    """`score_probe_transcript` over every transcript, in order.

    Raises:
        KeyError: if a transcript's `question_id` has no matching probe example -- mirrors
            `cragb.eval.run_judge_eval.score_transcripts`'s convention for the identical
            "scoring against ground truth that doesn't exist" failure.
    """
    missing = [t.question_id for t in transcripts if t.question_id not in examples_by_id]
    if missing:
        raise KeyError(f"No probe example for transcript question_id(s): {missing}")
    return [score_probe_transcript(t, examples_by_id[t.question_id], template, chat_fn) for t in transcripts]


# --------------------------------------------------------------------------
# CRAGB evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEvalResult:
    """One evaluation source's (CRAGB or probe) full result bundle, ready to summarize."""

    source: str
    model: str
    transcripts: list[GroundedQATranscript]
    citation_scores: list[TranscriptScore]
    faithfulness_scores: list[int]
    latency_seconds: list[float]


def run_cragb_evaluation(cfg: dict, template: Template, judge_template: Template, local_client, judge_chat_fn) -> SourceEvalResult:
    """Generate, validate, and score the untuned local model over all 60 CRAGB questions."""
    questions = load_retrieval_questions(cfg["paths"]["cragb_questions_in"])
    expected_ids = tuple(q.id for q in questions)

    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    chunking_config = load_chunking_config(cfg["paths"]["chunking_config"])
    retriever, chunk_to_parent = index_bm25_retriever(corpus, chunking_config)
    lookup = build_corpus_lookup(corpus)
    k = cfg["retrieval"]["k"]

    logger.info("Generating %d CRAGB transcript(s) with %s...", len(questions), local_client.model)
    transcripts = run_grounded_qa(questions, retriever, chunk_to_parent, lookup, template, local_client.complete, k=k)
    validate_full_run(transcripts, expected_ids)
    assert_retrieval_matches_rag_small(transcripts, cfg["paths"]["rag_small_transcripts_in"])

    expected_abstentions = load_expected_abstentions(cfg["paths"]["cragb_questions_in"])
    gold_relevant = gold_relevant_ids_by_question(questions)
    citation_scores = score_transcripts(transcripts, expected_abstentions, gold_relevant)

    references = load_reference_answers(cfg["paths"]["cragb_references_in"])
    judge_df = score_transcripts_with_judge("ft_baseline", transcripts, references, judge_template, judge_chat_fn)
    faithfulness_scores = judge_df["faithfulness"].tolist()

    latency_cfg = cfg.get("latency", {})
    n_latency = min(int(latency_cfg.get("n_cragb_questions", 15)), len(questions))
    rng = np.random.default_rng(cfg["seed"])
    latency_sample = stratified_question_sample(questions, n_latency, rng)
    usage_fn = partial(local_client.complete_with_usage, bypass_cache=True)
    logger.info("Timing generation on %d stratified CRAGB question(s) (bypass_cache=True)...", len(latency_sample))
    latency_rows = [
        _run_one_grounded_question_timed("ft_baseline", q, template, retriever, chunk_to_parent, lookup, k, usage_fn)
        for q in latency_sample
    ]
    latency_seconds = [row.generate_ms / 1000 for row in latency_rows]

    return SourceEvalResult(
        source=CRAGB_SOURCE,
        model=local_client.model,
        transcripts=transcripts,
        citation_scores=citation_scores,
        faithfulness_scores=faithfulness_scores,
        latency_seconds=latency_seconds,
    )


# --------------------------------------------------------------------------
# Probe evaluation
# --------------------------------------------------------------------------


def run_probe_evaluation(cfg: dict, template: Template, judge_template: Template, local_client, judge_chat_fn) -> SourceEvalResult:
    """Generate, score, and time the untuned local model over the behaviour-probe set.

    No retrieval: each `TrainingExample.context_text`/`.source_doc_ids` is already a
    fully-rendered context (T7.2's sampler), reconstructed here as a `ContextBlock` the
    same way `cragb.finetune.schema.render_training_prompt` and
    `cragb.eval.cost_model.build_messages_for_row` both already do, for the same reason:
    `render_prompt` only ever needs `context.text`.
    """
    examples = load_training_examples_jsonl(cfg["paths"]["probe_in"])
    examples_by_id = {e.example_id: e for e in examples}

    logger.info("Generating %d probe transcript(s) with %s...", len(examples), local_client.model)
    transcripts = []
    for example in examples:
        context = ContextBlock(text=example.context_text, doc_ids=example.source_doc_ids, photo_flags={})
        transcripts.append(generate_answer(example.example_id, example.question, context, template, local_client.complete))
    validate_full_run(transcripts, tuple(e.example_id for e in examples))

    expected_abstentions = {e.example_id: e.is_abstention for e in examples}
    citation_scores = score_transcripts(transcripts, expected_abstentions, gold_relevant_ids=None)

    judge_scores = score_probe_transcripts(transcripts, examples_by_id, judge_template, judge_chat_fn)
    faithfulness_scores = [s.faithfulness for s in judge_scores]

    latency_cfg = cfg.get("latency", {})
    n_latency = min(int(latency_cfg.get("n_probe_questions", 15)), len(examples))
    rng = np.random.default_rng(cfg["seed"])
    # Index-based sampling (mirrors cragb.eval.run_cost_latency.stratified_question_sample's
    # own approach) rather than rng.choice(examples, ...) directly -- avoids numpy's
    # object-array coercion of a list of dataclass instances.
    chosen_idx = sorted(rng.choice(len(examples), size=n_latency, replace=False)) if n_latency else []
    latency_sample = [examples[i] for i in chosen_idx]
    logger.info("Timing generation on %d probe question(s) (bypass_cache=True)...", len(latency_sample))
    latency_seconds = []
    for example in latency_sample:
        context = ContextBlock(text=example.context_text, doc_ids=example.source_doc_ids, photo_flags={})
        prompt = template.substitute(question=example.question, context_block=context.text)
        result = local_client.complete_with_usage([{"role": "user", "content": prompt}], bypass_cache=True)
        assert result.latency_s is not None
        latency_seconds.append(result.latency_s)

    return SourceEvalResult(
        source=PROBE_SOURCE,
        model=local_client.model,
        transcripts=transcripts,
        citation_scores=citation_scores,
        faithfulness_scores=faithfulness_scores,
        latency_seconds=latency_seconds,
    )


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------


def build_baseline_row(result: SourceEvalResult, n_boot: int, alpha: float, rng: np.random.Generator | None) -> dict:
    """One `ft_base_baseline_v1.csv` row from an already-scored `SourceEvalResult`.

    Every field but `source`/`model`/the faithfulness+latency stats comes straight from
    `cragb.eval.citation_validity.summarize` -- the same headline-metrics function T4a.4's
    table uses, so these columns are literally comparable to that one.
    """
    citation_summary = summarize(result.citation_scores).iloc[0].to_dict()
    n_fabricated = citation_summary["n_fabricated_citations"]
    n_total = citation_summary["n_total_citations"]
    fabricated_citation_rate = (n_fabricated / n_total) if n_total else None

    faithfulness_mean = float(np.mean(result.faithfulness_scores))
    faithfulness_ci_lo, faithfulness_ci_hi = bootstrap_ci(
        result.faithfulness_scores, n_boot=n_boot, alpha=alpha, rng=rng
    )
    # latency_stats (not raw np.median) so the percentile convention matches every other
    # latency table in this project (cragb.utils.timing's fixed numpy-interpolation rule).
    median_latency_s = latency_stats(result.latency_seconds)["p50"] if result.latency_seconds else None

    return {
        "source": result.source,
        "model": result.model,
        "n_questions": citation_summary["n_questions"],
        "format_compliance_rate": citation_summary["format_compliance_rate"],
        "citation_validity_rate": citation_summary["citation_validity_rate"],
        "n_fabricated_citations": n_fabricated,
        "fabricated_citation_rate": fabricated_citation_rate,
        "gold_grounding_rate": citation_summary["gold_grounding_rate"],
        "abstention_accuracy": citation_summary["abstention_accuracy"],
        "self_contradiction_rate": citation_summary["self_contradiction_rate"],
        "ungrounded_answer_rate": citation_summary["ungrounded_answer_rate"],
        "faithfulness_mean": faithfulness_mean,
        "faithfulness_ci_lo": faithfulness_ci_lo,
        "faithfulness_ci_hi": faithfulness_ci_hi,
        "n_latency_questions": len(result.latency_seconds),
        "median_latency_s": median_latency_s,
    }


def build_baseline_table(results: list[SourceEvalResult], n_boot: int, alpha: float, rng: np.random.Generator | None) -> pd.DataFrame:
    return pd.DataFrame([build_baseline_row(r, n_boot, alpha, rng) for r in results])


# --------------------------------------------------------------------------
# Appendix transcripts
# --------------------------------------------------------------------------


def write_baseline_transcripts_jsonl(results: list[SourceEvalResult], out_path: str | Path) -> Path:
    """Write every result's transcripts to one combined appendix-quoting file, tagged by
    `source` -- deliberately not `cragb.generate.grounded_qa.write_transcripts_jsonl`
    (which has no room for that tag): this file is a report artifact for a human to read
    and pull worked examples from, not a pipeline input another module re-parses, so a
    bespoke shape here is the right call rather than reusing a stricter contract that
    doesn't fit.
    """
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for result in results:
            for t in result.transcripts:
                row = t.to_dict()
                row["source"] = result.source
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
    return resolved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_baseline.yaml", help="Path to the baseline config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    template = load_prompt_template(cfg["paths"]["grounded_qa_prompt_template"])
    judge_template = load_prompt_template(cfg["paths"]["judge_prompt_template"])

    provider_cfg = cfg["provider"]
    local_client = LocalHFClient(
        model=provider_cfg["model"],
        device=provider_cfg.get("device"),
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        load_in_4bit=provider_cfg["load_in_4bit"],
        cache_dir=cfg["paths"]["cache_dir"],
    )

    from cragb.generate.api_clients import GroqClient

    judge_provider_cfg = cfg["judge"]["provider"]
    judge_client = GroqClient(
        model=judge_provider_cfg["model"],
        api_base=judge_provider_cfg["api_base"],
        api_key_env=judge_provider_cfg["api_key_env"],
        temperature=judge_provider_cfg["temperature"],
        max_tokens=judge_provider_cfg["max_tokens"],
        reasoning_effort=judge_provider_cfg.get("reasoning_effort"),
        timeout_s=judge_provider_cfg["timeout_s"],
        max_retries=judge_provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )

    cragb_result = run_cragb_evaluation(cfg, template, judge_template, local_client, judge_client.complete)
    probe_result = run_probe_evaluation(cfg, template, judge_template, local_client, judge_client.complete)

    bootstrap_cfg = cfg.get("bootstrap", {})
    rng = np.random.default_rng(cfg["seed"])
    table = build_baseline_table(
        [cragb_result, probe_result],
        n_boot=int(bootstrap_cfg.get("n_boot", 10000)),
        alpha=float(bootstrap_cfg.get("alpha", 0.05)),
        rng=rng,
    )
    out_path = resolve_path(cfg["paths"]["baseline_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    transcripts_path = write_baseline_transcripts_jsonl([cragb_result, probe_result], cfg["paths"]["transcripts_out"])

    logger.info("Wrote baseline table to %s", out_path)
    logger.info("Wrote %d transcript(s) to %s", len(cragb_result.transcripts) + len(probe_result.transcripts), transcripts_path)
    for row in table.to_dict("records"):
        logger.info(
            "  %s: citation_validity=%.2f abstention_accuracy=%.2f faithfulness=%.2f [%.2f, %.2f] median_latency=%.1fs",
            row["source"],
            row["citation_validity_rate"] or 0.0,
            row["abstention_accuracy"],
            row["faithfulness_mean"],
            row["faithfulness_ci_lo"],
            row["faithfulness_ci_hi"],
            row["median_latency_s"] or 0.0,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
