"""Grounded-QA pilot run: transcripts + citation-validity table (T4a.5; PLAN.md §3 E4,
M4a.md T4a.5).

Runs T4a.2 (context building) -> T4a.3 (generation) -> T4a.4 (scoring) end-to-end over a
deliberately curated slice of CRAGB v1, producing the two artifacts PLAN.md §7 lists as
mid-progress report material:

- `results/tables/grounded_qa_transcripts_v1.jsonl` — >=10 full worked transcripts.
- `results/tables/grounded_qa_validity_v1.csv` — the one-row citation-validity /
  abstention-accuracy headline table, plus a sibling per-question breakdown at
  `results/tables/grounded_qa_validity_per_question_v1.csv`.

Usage:
    python -m cragb.eval.run_grounded_qa_pilot --config configs/grounded_qa.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cragb.eval.cragb_questions import RetrievalQuestion, load_retrieval_questions
from cragb.eval.citation_validity import (
    TranscriptScore,
    gold_relevant_ids_by_question,
    load_expected_abstentions,
    per_question_dataframe,
    score_transcripts,
    summarize,
)
from cragb.generate.api_clients import GroqClient
from cragb.generate.context_builder import build_corpus_lookup, index_bm25_retriever
from cragb.generate.grounded_qa import (
    GroundedQATranscript,
    load_prompt_template,
    run_grounded_qa,
    write_transcripts_jsonl,
)
from cragb.retrieval.chunking import load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

# The pilot slice (M4a.md T4a.5: "~10-12 questions... spread across taxonomy types, both
# real is_abstention=True cases, and 1-2 of the surprisingly-answerable-negative questions
# from §14.2 as failure-mode material"). Curated by hand, not sampled, so the mid-progress
# report's worked examples are chosen for what they demonstrate, not left to chance.
CURATED_QUESTION_IDS: tuple[str, ...] = (
    # Both of CRAGB v1's *genuine* ground-truth abstentions (PLAN.md §14.2): the only two
    # negatives whose pools actually came back empty when T2.7 labeled them.
    "fabric_quality_neg_000",  # "exact thread count" -- no review reports lab measurements
    "defects_neg_000",  # "% units with a manufacturing defect per internal QA data"
    # Surprisingly-answerable negatives (PLAN.md §14.2): authored as negatives but T2.7
    # found real relevant evidence once the pools were actually labeled -- deliberately
    # included as failure-mode material (a "should this abstain?" edge case, not a clean
    # abstain-or-answer split).
    "fit_sizing_neg_001",  # the most striking miss: 17/19 pooled reviews were on-topic
    "durability_neg_000",
    # One representative positive question per taxonomy type, so the slice spans all 7
    # CRAGB v1 categories, not just fit/negatives.
    "fit_sizing_000",
    "colour_appearance_009",
    "fabric_quality_000",
    "durability_000",
    "defects_000",
    "occasion_000",
    "value_000",
)


def select_pilot_questions(
    all_questions: list[RetrievalQuestion],
    question_ids: tuple[str, ...] = CURATED_QUESTION_IDS,
) -> list[RetrievalQuestion]:
    """Pick `question_ids` out of `all_questions`, in `question_ids`' curated order.

    Args:
        all_questions: every loaded CRAGB question (e.g. from
            `load_retrieval_questions`).
        question_ids: which ids to select, in the order they should
            appear in the pilot's outputs.

    Returns:
        The matching `RetrievalQuestion`s, in `question_ids` order (not
        `all_questions`' order) — so the transcripts file reads as the
        curated narrative T4a.5/T4a.6 intend, not benchmark file order.

    Raises:
        ValueError: if any id in `question_ids` is not found in
            `all_questions`.
    """
    by_id = {q.id: q for q in all_questions}
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"Curated question id(s) not found in CRAGB: {missing}")
    return [by_id[qid] for qid in question_ids]


def validate_pilot_run(
    transcripts: list[GroundedQATranscript], expected_question_ids: tuple[str, ...]
) -> None:
    """Fail loudly on the two concrete ways this pipeline has already broken silently.

    T4a.3's build against the real API surfaced an empty-`answer_text`
    failure mode (a `max_tokens` cap too low for a reasoning model to
    finish its visible answer, PLAN.md-style bottleneck #3 territory);
    this is the regression guard for it, run before any output is
    written so a bad batch is never mistaken for a good one.

    Args:
        transcripts: the pilot's generated transcripts.
        expected_question_ids: the ids that were requested, in order.

    Raises:
        ValueError: if the transcript count/ids don't match what was
            requested, or if any transcript's `answer_text` is empty.
    """
    got_ids = tuple(t.question_id for t in transcripts)
    if got_ids != tuple(expected_question_ids):
        raise ValueError(
            f"Transcript question ids {list(got_ids)} do not match the requested "
            f"pilot slice {list(expected_question_ids)}."
        )
    empty = [t.question_id for t in transcripts if not t.answer_text.strip()]
    if empty:
        raise ValueError(
            f"{len(empty)} transcript(s) have an empty answer_text: {empty}. "
            "Likely a max_tokens cap too low for the model to finish its visible "
            "answer (see configs/grounded_qa.yaml's max_tokens comment)."
        )


def write_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write `df` to `out_path`, creating parent directories as needed."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved, index=False)
    return resolved


def _log_summary(scores: list[TranscriptScore], aggregate: pd.DataFrame) -> None:
    row = aggregate.iloc[0]
    logger.info(
        "Pilot summary over %d question(s): format_compliance_rate=%.2f "
        "citation_validity_rate=%s gold_grounding_rate=%s abstention_accuracy=%.2f "
        "self_contradiction_rate=%.2f ungrounded_answer_rate=%.2f",
        int(row["n_questions"]),
        row["format_compliance_rate"],
        row["citation_validity_rate"],
        row["gold_grounding_rate"],
        row["abstention_accuracy"],
        row["self_contradiction_rate"],
        row["ungrounded_answer_rate"],
    )
    for s in scores:
        logger.info(
            "  %s: abstained=%s (expected=%s, correct=%s) cited=%d fabricated=%s",
            s.question_id,
            s.predicted_abstained,
            s.expected_abstained,
            s.abstention_correct,
            s.n_citations,
            list(s.fabricated_citations),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grounded_qa.yaml", help="Path to grounded-QA config YAML.")
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

    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    all_questions = load_retrieval_questions(cfg["paths"]["questions_in"])
    questions = select_pilot_questions(all_questions, CURATED_QUESTION_IDS)

    template = load_prompt_template(cfg["paths"]["prompt_template"])
    chunking_config = load_chunking_config(cfg["retrieval"]["chunking_config"])
    retriever, chunk_to_parent = index_bm25_retriever(corpus, chunking_config)
    lookup = build_corpus_lookup(corpus)

    provider_cfg = cfg["provider"]
    client = GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )

    transcripts = run_grounded_qa(
        questions, retriever, chunk_to_parent, lookup, template, client.complete, k=cfg["retrieval"]["k"]
    )
    validate_pilot_run(transcripts, CURATED_QUESTION_IDS)

    expected_abstentions = load_expected_abstentions(cfg["paths"]["questions_in"])
    gold_relevant_ids = gold_relevant_ids_by_question(questions)
    scores = score_transcripts(transcripts, expected_abstentions, gold_relevant_ids=gold_relevant_ids)

    transcripts_path = write_transcripts_jsonl(transcripts, cfg["paths"]["transcripts_out"])
    aggregate = summarize(scores)
    per_question = per_question_dataframe(scores)
    validity_path = write_csv(aggregate, cfg["paths"]["validity_out"])
    per_question_path = write_csv(per_question, cfg["paths"]["validity_per_question_out"])

    logger.info("Wrote %d transcript(s) to %s", len(transcripts), transcripts_path)
    logger.info("Wrote aggregate validity table to %s", validity_path)
    logger.info("Wrote per-question validity table to %s", per_question_path)
    _log_summary(scores, aggregate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
