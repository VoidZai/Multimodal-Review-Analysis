"""Citation-validity and abstention-correctness scoring (T4a.4; PLAN.md §3 E4, M4a.md T4a.4).

Scores a `cragb.generate.grounded_qa.GroundedQATranscript` against two independent
questions E4 names as its metrics:

- **Is every citation trustworthy?** — format compliance (every bracketed token in the
  answer is a well-formed `[doc_id]` or `[photo of doc_id]` citation, not some other
  bracket shape the model invented), citation existence (every cited `doc_id` was
  actually shown to the model, i.e. present in the context it was given — a fabricated
  id is the sharpest possible grounding failure), and, where CRAGB's pooled ground truth
  is available, whether the cited id is also independently known-relevant (a stronger,
  optional "does it actually support the claim" signal).
- **Did the model abstain exactly when it should have?** — comparing the model's own
  abstention signal (`GroundedQATranscript.abstained`, from T4a.3's containment check
  against `ABSTENTION_TEXT`) against CRAGB's ground-truth `is_abstention` field.
  Deliberately **not** `is_negative` — PLAN.md §14.2 records why: 9 of CRAGB v1's 11
  taxonomy-authored "negative" questions turned out to have real relevant evidence once
  T2.7 actually labeled the pools, so `is_negative` is known to be an unreliable
  abstention signal and `is_abstention` (evidence-derived, per T2.8) is the one the
  reference answers themselves were held to.

Two self-contradictory shapes are scored, not raised on — mirroring the design decision
already made in `cragb.generate.grounded_qa.parse_completion` (record model mistakes as
data for this module, don't crash the pipeline over them):

- **self-contradiction**: abstained yet still cites a `doc_id` — a `ReferenceAnswer`
  built by a human is guaranteed not to have this shape (`make_reference_answer` raises
  on it); a model's is not.
- **ungrounded answer**: answered (didn't abstain) yet cited nothing — the model
  produced a "grounded" answer that grounds nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.generate.grounded_qa import GroundedQATranscript
from cragb.utils.io import resolve_path

# Any `[...]` substring in an answer that is *not* one of these two shapes
# is a citation the model invented in a non-conforming format (e.g.
# "[see review 128775]") — recognisable as an attempted citation, but not
# one `cragb.bench.reference_answers.extract_citations`/
# `cragb.generate.grounded_qa.extract_photo_citations` would ever parse
# out, so it would otherwise vanish from `cited_doc_ids` silently.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_DOC_CITATION_SHAPE_RE = re.compile(r"^\[\w+\]$")
_PHOTO_CITATION_SHAPE_RE = re.compile(r"^\[photo of \w+\]$")


def find_malformed_citations(answer_text: str) -> tuple[str, ...]:
    """Bracketed substrings in `answer_text` that aren't a valid citation shape.

    Returns:
        Malformed bracket tokens (e.g. `"[see review 128775]"`), in order
        of appearance, duplicates included (each occurrence is a distinct
        formatting failure worth counting).
    """
    return tuple(
        token
        for token in (m.group(0) for m in _BRACKET_RE.finditer(answer_text))
        if not (_DOC_CITATION_SHAPE_RE.match(token) or _PHOTO_CITATION_SHAPE_RE.match(token))
    )


@dataclass(frozen=True)
class TranscriptScore:
    """Per-question scoring of one `GroundedQATranscript`."""

    question_id: str

    # --- citation quality ---
    format_compliant: bool
    malformed_citations: tuple[str, ...]
    n_citations: int
    fabricated_citations: tuple[str, ...]  # cited but not in the context shown to the model
    citation_validity_rate: float | None  # valid / cited; None if n_citations == 0
    n_grounded_in_gold: int | None  # cited AND in CRAGB's pooled relevant_ids; None if not evaluated
    ungrounded_in_gold: tuple[str, ...]  # cited but not in CRAGB's pooled relevant_ids

    # --- abstention correctness ---
    predicted_abstained: bool
    expected_abstained: bool
    abstention_correct: bool

    # --- self-consistency failure modes ---
    self_contradiction: bool  # abstained yet still cites a doc_id
    ungrounded_answer: bool  # answered (no abstention) yet cited nothing


def score_transcript(
    transcript: GroundedQATranscript,
    expected_abstained: bool,
    gold_relevant_ids: frozenset[str] | None = None,
) -> TranscriptScore:
    """Score one transcript's citation validity and abstention correctness.

    Args:
        transcript: a generated `GroundedQATranscript` (T4a.3).
        expected_abstained: CRAGB's ground-truth `is_abstention` for this
            question (see `load_expected_abstentions`).
        gold_relevant_ids: CRAGB's pooled relevant doc_ids for this
            question, if available (see
            `cragb.eval.cragb_questions.load_retrieval_questions`). When
            given, enables the stronger (optional) "cited id is
            independently known-relevant" check; when `None`,
            `n_grounded_in_gold` is `None` and `ungrounded_in_gold` is
            empty — that check is simply not evaluated, not failed.

    Returns:
        A `TranscriptScore`.
    """
    malformed = find_malformed_citations(transcript.answer_text)

    context_ids = set(transcript.context.doc_ids)
    cited = transcript.cited_doc_ids
    n_citations = len(cited)
    fabricated = tuple(c for c in cited if c not in context_ids)
    n_valid = n_citations - len(fabricated)
    citation_validity_rate = (n_valid / n_citations) if n_citations else None

    if gold_relevant_ids is not None:
        ungrounded_in_gold = tuple(c for c in cited if c not in gold_relevant_ids)
        n_grounded_in_gold = n_citations - len(ungrounded_in_gold)
    else:
        ungrounded_in_gold = ()
        n_grounded_in_gold = None

    return TranscriptScore(
        question_id=transcript.question_id,
        format_compliant=not malformed,
        malformed_citations=malformed,
        n_citations=n_citations,
        fabricated_citations=fabricated,
        citation_validity_rate=citation_validity_rate,
        n_grounded_in_gold=n_grounded_in_gold,
        ungrounded_in_gold=ungrounded_in_gold,
        predicted_abstained=transcript.abstained,
        expected_abstained=expected_abstained,
        abstention_correct=transcript.abstained == expected_abstained,
        self_contradiction=transcript.abstained and n_citations > 0,
        ungrounded_answer=(not transcript.abstained) and n_citations == 0,
    )


def load_expected_abstentions(path: str | Path = "benchmark/cragb_v1.jsonl") -> dict[str, bool]:
    """question_id -> ground-truth `is_abstention`, read directly from the assembled benchmark.

    A small, self-contained reader rather than a reuse of
    `cragb.eval.cragb_questions.RetrievalQuestion` (which does not carry
    `is_abstention` — it only needs `is_negative`/`relevant_ids` for
    retrieval scoring) or `cragb.bench.assemble.CragbEntry` (the
    assembly pipeline's own internal, write-side shape). Reading the
    finished artifact back for this narrower need gets its own minimal
    view, following the precedent `cragb.eval.cragb_questions`'s own
    docstring already sets for exactly this situation.
    """
    expected: dict[str, bool] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expected[obj["id"]] = bool(obj["is_abstention"])
    return expected


def gold_relevant_ids_by_question(
    questions: list[RetrievalQuestion],
) -> dict[str, frozenset[str]]:
    """question_id -> pooled relevant doc_ids, from already-loaded `RetrievalQuestion`s."""
    return {q.id: q.relevant_ids for q in questions}


def score_transcripts(
    transcripts: list[GroundedQATranscript],
    expected_abstentions: dict[str, bool],
    gold_relevant_ids: dict[str, frozenset[str]] | None = None,
) -> list[TranscriptScore]:
    """Score every transcript in `transcripts`, in order.

    Args:
        transcripts: generated transcripts (T4a.3).
        expected_abstentions: from `load_expected_abstentions`.
        gold_relevant_ids: from `gold_relevant_ids_by_question`, or
            `None` to skip the gold-grounding check for every transcript.

    Returns:
        One `TranscriptScore` per transcript, same order.

    Raises:
        KeyError: if a transcript's `question_id` has no entry in
            `expected_abstentions` — scoring abstention correctness
            against a missing ground-truth label would silently produce
            a meaningless result, so this fails loudly instead.
    """
    scores = []
    for t in transcripts:
        if t.question_id not in expected_abstentions:
            raise KeyError(
                f"No ground-truth abstention label for question_id {t.question_id!r}; "
                "check it exists in benchmark/cragb_v1.jsonl."
            )
        gold = gold_relevant_ids.get(t.question_id) if gold_relevant_ids is not None else None
        scores.append(score_transcript(t, expected_abstentions[t.question_id], gold_relevant_ids=gold))
    return scores


def per_question_dataframe(scores: list[TranscriptScore]) -> pd.DataFrame:
    """One row per `TranscriptScore`, tuple fields converted to lists for a clean table/CSV."""
    rows = []
    for s in scores:
        row = asdict(s)
        row["malformed_citations"] = list(s.malformed_citations)
        row["fabricated_citations"] = list(s.fabricated_citations)
        row["ungrounded_in_gold"] = list(s.ungrounded_in_gold)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(scores: list[TranscriptScore]) -> pd.DataFrame:
    """Aggregate `scores` into the one-row headline metrics table E4 asks for.

    `citation_validity_rate` and `gold_grounding_rate` are **micro-averaged**
    (total valid citations / total citations across all questions), not a
    mean of per-question rates — a question with 5 citations and a
    question with 1 citation should not count equally toward the
    headline rate. Questions that cited nothing (`n_citations == 0`,
    typically abstentions) are excluded from both, exactly as a
    per-question rate of `None` already signals "not applicable" rather
    than "failed".

    Args:
        scores: from `score_transcripts`.

    Returns:
        A one-row `pd.DataFrame` with `n_questions`,
        `format_compliance_rate`, `citation_validity_rate`,
        `gold_grounding_rate` (`None` if no score carried a gold check),
        `abstention_accuracy`, `self_contradiction_rate`,
        `ungrounded_answer_rate`, `n_total_citations`,
        `n_fabricated_citations`.

    Raises:
        ValueError: if `scores` is empty.
    """
    if not scores:
        raise ValueError("Cannot summarize an empty list of scores.")

    n = len(scores)
    total_citations = sum(s.n_citations for s in scores)
    total_fabricated = sum(len(s.fabricated_citations) for s in scores)
    citation_validity_rate = (
        (total_citations - total_fabricated) / total_citations if total_citations else None
    )

    gold_scores = [s for s in scores if s.n_grounded_in_gold is not None]
    if gold_scores:
        gold_citations = sum(s.n_citations for s in gold_scores)
        gold_grounded = sum(s.n_grounded_in_gold for s in gold_scores)
        gold_grounding_rate = (gold_grounded / gold_citations) if gold_citations else None
    else:
        gold_grounding_rate = None

    return pd.DataFrame(
        [
            {
                "n_questions": n,
                "format_compliance_rate": sum(s.format_compliant for s in scores) / n,
                "citation_validity_rate": citation_validity_rate,
                "gold_grounding_rate": gold_grounding_rate,
                "abstention_accuracy": sum(s.abstention_correct for s in scores) / n,
                "self_contradiction_rate": sum(s.self_contradiction for s in scores) / n,
                "ungrounded_answer_rate": sum(s.ungrounded_answer for s in scores) / n,
                "n_total_citations": total_citations,
                "n_fabricated_citations": total_fabricated,
            }
        ]
    )
