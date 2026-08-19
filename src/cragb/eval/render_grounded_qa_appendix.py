"""Grounded-QA transcripts appendix (T4a.6; PLAN.md §7, M4a.md T4a.6).

Renders a hand-picked slice of T4a.5's pilot transcripts into report-ready markdown:
`reports/grounded_qa_transcripts_v1.md`, the "5 grounded-QA transcripts with citations"
PLAN.md §7 lists as mid-progress report appendix material.

`APPENDIX_ENTRIES`' `note` field is a human-authored editorial judgment, not generated —
the same principle `cragb.eval.run_retrieval_eval.render_winloss_markdown`'s docstring
already states for T3.8's win/loss appendix: "why this retriever won" (there) or "why
this transcript is here" (here) is an interpretive call a human makes after reading the
real output, not text a template should invent. The facts each section renders (question,
retrieved reviews, the model's actual answer, its scores) are computed, never authored.

Usage:
    python -m cragb.eval.render_grounded_qa_appendix --config configs/grounded_qa.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from cragb.eval.citation_validity import (
    TranscriptScore,
    gold_relevant_ids_by_question,
    load_expected_abstentions,
    score_transcripts,
)
from cragb.eval.cragb_questions import load_retrieval_questions
from cragb.generate.grounded_qa import GroundedQATranscript, load_transcripts_jsonl
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppendixEntry:
    """One curated transcript, plus the editorial label/note explaining why it's here."""

    question_id: str
    label: str
    note: str


# The 5-transcript appendix slice (M4a.md T4a.6: "2-3 clean grounded answers, 1
# abstention, 1 failure mode"), hand-picked from T4a.5's 11-question pilot run
# (`results/tables/grounded_qa_transcripts_v1.jsonl`).
APPENDIX_ENTRIES: tuple[AppendixEntry, ...] = (
    AppendixEntry(
        question_id="fit_sizing_000",
        label="Clean grounded answer",
        note=(
            "A standard case: five reviews retrieved, the model reports the "
            "majority/minority split in what buyers actually say rather than "
            "picking one side, and cites every claim to a real review id."
        ),
    ),
    AppendixEntry(
        question_id="durability_000",
        label="Clean grounded answer (second taxonomy type)",
        note="Same pattern as fit_sizing_000, on a different taxonomy category (durability, not fit).",
    ),
    AppendixEntry(
        question_id="fit_sizing_neg_001",
        label="Clean grounded answer on a tricky edge case",
        note=(
            "Authored as a taxonomy negative (T2.3), but T2.7's pooling found real "
            "relevant evidence for it (PLAN.md §14.2: 17 of 19 pooled reviews were "
            "on-topic — the most striking miss of the 9 negatives that turned out "
            "answerable). CRAGB v1's ground truth is evidence-driven, not "
            "taxonomy-driven (T2.8), so this question's `is_abstention` is `False` "
            "— and the model correctly answers it instead of abstaining, matching "
            "that evidence-driven ground truth rather than the original taxonomy label."
        ),
    ),
    AppendixEntry(
        question_id="fabric_quality_neg_000",
        label="Correct abstention",
        note=(
            "One of CRAGB v1's only two genuine ground-truth abstentions (PLAN.md "
            "§14.2): no review reports an exact thread-count measurement, and the "
            "model correctly says so instead of guessing or fabricating a number."
        ),
    ),
    AppendixEntry(
        question_id="colour_appearance_009",
        label="Failure mode: unused photo-citation affordance",
        note=(
            "Two of the three retrieved reviews had a photo attached (`has_photo: "
            "yes` in the context shown to the model), and the prompt (T4a.1, rule 4) "
            "explicitly permits citing `[photo of doc_id]` when a photo is the best "
            "evidence for a colour/appearance claim — exactly this question's type. "
            "The model never reaches for it here: it cites review text only. This "
            "is not a scored failure (every text citation below is valid and "
            "gold-grounded) but a real, measured limitation worth flagging for E7's "
            "multimodal pilot — the model under-uses photo evidence even when the "
            "prompt explicitly offers it."
        ),
    ),
)


def render_transcript_markdown(
    transcript: GroundedQATranscript, score: TranscriptScore, entry: AppendixEntry
) -> str:
    """Render one transcript + its T4a.4 score as a markdown section."""
    context_lines = "\n".join(
        f"- `{doc_id}` (has_photo: {'yes' if transcript.context.photo_flags.get(doc_id) else 'no'})"
        for doc_id in transcript.context.doc_ids
    ) or "- (none retrieved)"

    if transcript.abstained:
        citation_note = "no citations (abstained)"
    elif score.n_citations == 0:
        citation_note = "no citations"
    elif score.fabricated_citations:
        citation_note = f"{score.n_citations} citation(s), {len(score.fabricated_citations)} fabricated"
    else:
        citation_note = f"{score.n_citations} citation(s), all valid"

    abstention_note = "expected" if score.expected_abstained else "not expected"

    return (
        f"## {entry.label}: `{transcript.question_id}`\n\n"
        f"*{entry.note}*\n\n"
        f"**Question:** {transcript.question}\n\n"
        f"**Reviews retrieved (k={len(transcript.context.doc_ids)}):**\n{context_lines}\n\n"
        f"**Model's answer:**\n\n> {transcript.answer_text}\n\n"
        f"**Scoring:** abstained={transcript.abstained} ({abstention_note}), "
        f"format_compliant={score.format_compliant}, {citation_note}.\n"
    )


def render_appendix_markdown(
    entries: list[AppendixEntry],
    transcripts_by_id: dict[str, GroundedQATranscript],
    scores_by_id: dict[str, TranscriptScore],
) -> str:
    """Assemble the full appendix document from `entries`, in order.

    Args:
        entries: which transcripts to include, and why (see `APPENDIX_ENTRIES`).
        transcripts_by_id: from `load_transcripts_jsonl`, keyed by `question_id`.
        scores_by_id: from `cragb.eval.citation_validity.score_transcripts`,
            keyed by `question_id`.

    Returns:
        Complete Markdown document text.

    Raises:
        KeyError: if an entry's `question_id` is missing from either
            `transcripts_by_id` or `scores_by_id` — the appendix must
            never silently drop a curated example.
    """
    missing_transcripts = [e.question_id for e in entries if e.question_id not in transcripts_by_id]
    if missing_transcripts:
        raise KeyError(f"question id(s) not found in transcripts: {missing_transcripts}")
    missing_scores = [e.question_id for e in entries if e.question_id not in scores_by_id]
    if missing_scores:
        raise KeyError(f"question id(s) not found in scores: {missing_scores}")

    header = (
        "# Grounded-QA worked transcripts (T4a.6)\n\n"
        "Five transcripts hand-picked from T4a.5's 11-question pilot run over CRAGB v1 "
        "(PLAN.md §3 E4, §7 appendix material): three clean grounded answers spanning "
        "different taxonomy types and edge cases, one correct abstention, and one "
        "documented failure mode. Every citation and every word of the model's answer "
        "below is exactly what it produced — nothing has been edited.\n\n"
    )
    sections = [
        render_transcript_markdown(transcripts_by_id[e.question_id], scores_by_id[e.question_id], e)
        for e in entries
    ]
    return header + "\n---\n\n".join(sections)


def build_appendix_markdown(cfg: dict) -> str:
    """Load T4a.5's pilot output + score it, and render the curated appendix.

    Args:
        cfg: a loaded `configs/grounded_qa.yaml`-shaped config dict.

    Returns:
        Complete Markdown document text (see `render_appendix_markdown`).
    """
    transcripts = load_transcripts_jsonl(cfg["paths"]["transcripts_out"])
    transcripts_by_id = {t.question_id: t for t in transcripts}

    all_questions = load_retrieval_questions(cfg["paths"]["questions_in"])
    expected_abstentions = load_expected_abstentions(cfg["paths"]["questions_in"])
    gold_relevant_ids = gold_relevant_ids_by_question(all_questions)
    scores = score_transcripts(transcripts, expected_abstentions, gold_relevant_ids=gold_relevant_ids)
    scores_by_id = {s.question_id: s for s in scores}

    return render_appendix_markdown(list(APPENDIX_ENTRIES), transcripts_by_id, scores_by_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grounded_qa.yaml", help="Path to grounded-QA config YAML.")
    parser.add_argument(
        "--out", default="reports/grounded_qa_transcripts_v1.md", help="Output markdown path."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    markdown = build_appendix_markdown(cfg)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %d-transcript appendix to %s", len(APPENDIX_ENTRIES), out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
