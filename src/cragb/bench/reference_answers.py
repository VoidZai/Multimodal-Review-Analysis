"""Human reference answers for CRAGB (T2.8; PLAN.md §3 E1, §14.2).

T2.7 labeled which pooled reviews are actually relevant to each question.
This module turns that evidence into one human-written, strictly
grounded reference answer per question — the ground truth RQ0/RQ1's
similarity scoring (E5) is measured against.

**Grounding is enforced from the answer text itself, not a side-channel
list.** Each answer cites its evidence inline as `[doc_id]` markers
(e.g. "...runs small [128775][161398]..."); `extract_citations` parses
those out, so the citation list can never silently drift from what the
prose actually says. `validate_reference_answers` then checks every
cited id was actually labeled relevant in T2.7 — a reference answer that
cites unlabeled "evidence" would defeat the entire point of building
trustworthy ground truth by pooling and labeling in the first place.

**Abstention is evidence-driven, not taxonomy-driven** — a deliberate
refinement over M2.md's original framing ("an explicit 'not enough
information' for deliberate negatives"). PLAN.md §14.2 records why: 9 of
T2.3's 11 authored negative questions turned out to have real relevant
evidence once T2.7 actually labeled the pools, so answering those from
the *taxonomy's* `is_negative` flag would have produced 9 dishonest
abstentions. Instead, a question abstains here if and only if it has
zero relevant documents — whatever the taxonomy originally predicted.

Usage:
    python -m cragb.bench.reference_answers --config configs/reference_answers.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from cragb.bench.pooling import QuestionRecord, load_questions
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

ABSTENTION_TEXT = "Not enough information in the available reviews to answer this question."

_CITATION_RE = re.compile(r"\[(\w+)\]")


def load_relevant_doc_ids(labels_path: str | Path) -> dict[str, set[str]]:
    """question_id -> set of doc_ids T2.7's `relevance_labels_v1.jsonl` marked relevant."""
    relevant: dict[str, set[str]] = {}
    with resolve_path(labels_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["relevant"]:
                relevant.setdefault(obj["question_id"], set()).add(obj["doc_id"])
    return relevant


def extract_citations(answer_text: str) -> tuple[str, ...]:
    """Pull `[doc_id]` citation markers out of `answer_text`, in first-seen order, de-duplicated."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(answer_text):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


@dataclass(frozen=True)
class ReferenceAnswer:
    question_id: str
    answer: str
    cited_doc_ids: tuple[str, ...]
    is_abstention: bool


def make_reference_answer(question_id: str, answer_text: str) -> ReferenceAnswer:
    """Build a `ReferenceAnswer`, inferring citations and abstention from `answer_text` itself.

    Raises:
        ValueError: if the text is exactly the canonical abstention
            string but also contains citation markers (a contradiction —
            an abstention has no grounding to cite).
    """
    answer_text = answer_text.strip()
    cited = extract_citations(answer_text)
    # Containment, not exact equality: catches an authoring mistake like
    # appending a stray citation after the canonical phrase (which exact
    # equality would silently miss, since the two could never match).
    is_abstention = ABSTENTION_TEXT in answer_text
    if is_abstention and cited:
        raise ValueError(f"{question_id}: abstention answer must not contain citations")
    return ReferenceAnswer(
        question_id=question_id, answer=answer_text, cited_doc_ids=cited, is_abstention=is_abstention
    )


def load_reference_answers(path: str | Path) -> dict[str, ReferenceAnswer]:
    """Load a reference-answers JSONL, keyed by question id.

    Raises:
        ValueError: on a duplicate `question_id`.
    """
    answers: dict[str, ReferenceAnswer] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["question_id"]
            if qid in answers:
                raise ValueError(f"Duplicate reference answer for {qid!r}")
            answers[qid] = ReferenceAnswer(
                question_id=qid,
                answer=obj["answer"],
                cited_doc_ids=tuple(obj["cited_doc_ids"]),
                is_abstention=bool(obj["is_abstention"]),
            )
    return answers


@dataclass(frozen=True)
class ReferenceAnswerReport:
    ok: bool
    total_questions: int
    answered: int
    missing_questions: tuple[str, ...]
    citation_violations: tuple[str, ...]  # cites a doc_id T2.7 didn't label relevant
    ungrounded_answers: tuple[str, ...]  # non-abstention answer with zero citations
    false_confidence: tuple[str, ...]  # non-abstention despite zero relevant evidence
    unjustified_abstentions: tuple[str, ...]  # abstains despite having relevant evidence
    messages: tuple[str, ...]


def validate_reference_answers(
    questions: list[QuestionRecord],
    relevant_by_question: dict[str, set[str]],
    answers: dict[str, ReferenceAnswer],
) -> ReferenceAnswerReport:
    """Check every question has one strictly-grounded reference answer.

    Hard failures (`ok=False`):
      - a question with no reference answer at all;
      - an answer citing a doc_id T2.7 did not label relevant for that
        question — the core "strictly grounded" guarantee RQ0 depends
        on;
      - a non-abstention answer with zero citations (a "grounded" answer
        that cites nothing isn't grounded);
      - a non-abstention answer for a question with zero relevant
        evidence (nothing to ground it in — this must be an abstention).

    Soft flag (`ok` unaffected):
      - an abstention for a question that *does* have relevant evidence
        in its pool. Not necessarily wrong — the pooled evidence might
        not actually answer the specific question even though it was
        retrieved for it — but worth a second look.
    """
    messages: list[str] = []
    missing = tuple(q.id for q in questions if q.id not in answers)
    if missing:
        messages.append(f"{len(missing)} question(s) have no reference answer: {list(missing)}")

    citation_violations: list[str] = []
    ungrounded: list[str] = []
    false_confidence: list[str] = []
    unjustified_abstentions: list[str] = []

    for q in questions:
        answer = answers.get(q.id)
        if answer is None:
            continue
        relevant_ids = relevant_by_question.get(q.id, set())
        has_evidence = bool(relevant_ids)

        not_relevant_cited = set(answer.cited_doc_ids) - relevant_ids
        if not_relevant_cited:
            citation_violations.append(q.id)
            messages.append(f"{q.id}: cites doc_id(s) not labeled relevant: {sorted(not_relevant_cited)}")

        if not answer.is_abstention and not answer.cited_doc_ids:
            ungrounded.append(q.id)
            messages.append(f"{q.id}: non-abstention answer has no citations")

        if not answer.is_abstention and not has_evidence:
            false_confidence.append(q.id)
            messages.append(f"{q.id}: answers despite zero relevant evidence in its pool — should abstain")

        if answer.is_abstention and has_evidence:
            unjustified_abstentions.append(q.id)
            messages.append(f"{q.id}: abstains despite {len(relevant_ids)} relevant doc(s) in its pool")

    ok = not (missing or citation_violations or ungrounded or false_confidence)
    if ok and not messages:
        messages.append("Every question has a strictly-grounded reference answer.")

    return ReferenceAnswerReport(
        ok=ok,
        total_questions=len(questions),
        answered=len(questions) - len(missing),
        missing_questions=missing,
        citation_violations=tuple(citation_violations),
        ungrounded_answers=tuple(ungrounded),
        false_confidence=tuple(false_confidence),
        unjustified_abstentions=tuple(unjustified_abstentions),
        messages=tuple(messages),
    )


def write_reference_answers_jsonl(
    questions: list[QuestionRecord], answers: dict[str, ReferenceAnswer], out_path: str | Path
) -> Path:
    """Write reference answers as newline-delimited JSON, in `questions` order.

    Only questions with an answer are written — an incomplete run
    produces a partial (not corrupted) file, consistent with
    `validate_reference_answers` surfacing the gap as `missing_questions`.
    """
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for q in questions:
            answer = answers.get(q.id)
            if answer is None:
                continue
            row = {
                "question_id": answer.question_id,
                "answer": answer.answer,
                "cited_doc_ids": list(answer.cited_doc_ids),
                "is_abstention": answer.is_abstention,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/reference_answers.yaml", help="Path to reference-answers config YAML."
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
    questions = load_questions(cfg["paths"]["questions_in"])
    relevant_by_question = load_relevant_doc_ids(cfg["paths"]["labels_in"])
    answers = load_reference_answers(cfg["paths"]["answers_in"])

    report = validate_reference_answers(questions, relevant_by_question, answers)
    out_path = write_reference_answers_jsonl(questions, answers, cfg["paths"]["answers_out"])

    logger.info(
        "Wrote %d/%d reference answers to %s", report.answered, report.total_questions, out_path
    )
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
