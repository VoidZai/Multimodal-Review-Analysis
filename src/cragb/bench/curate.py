"""Curate CRAGB draft questions into `cragb_questions_v1` (T2.3; PLAN.md §3 E1, §9).

T2.2 over-generated ~90 LLM-drafted candidate questions from real corpus
reviews. This module turns that draft set into the taxonomy-balanced
final ~60 by combining three things:

1. **Automated quality filters** (`apply_quality_filters`) — cheap,
   objective checks (length, well-formed as a question) that reject
   obviously broken drafts before a human ever looks at them.
2. **An explicit human decision file**
   (`benchmark/curation_decisions_v1.yaml`) — every quality-passed draft
   id must appear in exactly one of that file's `accepted`/`rejected`
   lists, with an optional edited question text and (for rejections) a
   reason. `build_accepted_questions` enforces completeness: a
   quality-passed draft with no decision, or a decision referencing an
   id that doesn't exist, is a hard error — curation must be exhaustive
   and typo-checked, not "whatever happened to get picked."
3. **Authored negative questions** — T2.1's taxonomy reserves a
   deliberately-unanswerable quota per category (the fix for PLAN.md
   risk C), but T2.2's draft prompt only asked the LLM for
   answerable-style questions. Negatives are therefore authored directly
   in the decisions file's `negative_questions` section, not drafted.

A final pass (`flag_image_targets`) tags which accepted, non-negative
questions belong to a category with real image-bearing evidence in the
corpus — PLAN.md §3 E7 needs a meaningful image-bearing subset of CRAGB
for RQ4, and this flag is what T2.9 later uses to know which questions
are worth pairing with a best-evidence photo.

Usage:
    python -m cragb.bench.curate --config configs/curate.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import yaml

from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec, load_taxonomy
from cragb.data.vocab_check import matches_any_keyword
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Drafts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftRecord:
    id: str
    category: str
    question: str
    rationale: str


def load_drafts(path: str | Path) -> list[DraftRecord]:
    """Load T2.2's `cragb_draft.jsonl` into `DraftRecord`s."""
    records: list[DraftRecord] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                DraftRecord(
                    id=obj["id"],
                    category=obj["category"],
                    question=obj["question"],
                    rationale=obj.get("rationale", ""),
                )
            )
    return records


# --------------------------------------------------------------------------
# Automated quality filters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityIssue:
    id: str
    reason: str


def check_quality(
    draft: DraftRecord, min_words: int, max_words: int, require_question_mark: bool
) -> str | None:
    """Return a rejection reason, or `None` if `draft` passes automated checks."""
    n_words = len(draft.question.split())
    if n_words < min_words:
        return f"too short ({n_words} words < {min_words})"
    if n_words > max_words:
        return f"too long ({n_words} words > {max_words})"
    if require_question_mark and not draft.question.strip().endswith("?"):
        return "does not end with '?'"
    return None


def apply_quality_filters(
    drafts: list[DraftRecord],
    min_words: int,
    max_words: int,
    require_question_mark: bool,
) -> tuple[list[DraftRecord], list[QualityIssue]]:
    """Split `drafts` into those passing automated checks and those auto-rejected.

    Only quality-passed drafts require a human accept/reject decision
    (see `build_accepted_questions`) — a draft this stage already
    rejects doesn't need a human to weigh in on it too.
    """
    passed: list[DraftRecord] = []
    rejected: list[QualityIssue] = []
    for draft in drafts:
        reason = check_quality(draft, min_words, max_words, require_question_mark)
        if reason:
            rejected.append(QualityIssue(draft.id, reason))
        else:
            passed.append(draft)
    return passed, rejected


# --------------------------------------------------------------------------
# Human decisions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeQuestion:
    category: str
    question: str
    reason: str


@dataclass(frozen=True)
class CurationDecisions:
    accepted: dict[str, str | None]  # draft id -> edited question text, or None to keep as drafted
    rejected: dict[str, str]  # draft id -> reason
    negatives: tuple[NegativeQuestion, ...]


def load_decisions(path: str | Path) -> CurationDecisions:
    """Load the human curation decision file."""
    raw = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}

    accepted: dict[str, str | None] = {}
    for item in raw.get("accepted", []):
        accepted[item["id"]] = item.get("edited_question")

    rejected: dict[str, str] = {}
    for item in raw.get("rejected", []):
        rejected[item["id"]] = item["reason"]

    negatives = tuple(
        NegativeQuestion(category=n["category"], question=n["question"], reason=n["reason"])
        for n in raw.get("negative_questions", [])
    )

    overlap = set(accepted) & set(rejected)
    if overlap:
        raise ValueError(f"Draft id(s) both accepted and rejected: {sorted(overlap)}")

    return CurationDecisions(accepted=accepted, rejected=rejected, negatives=negatives)


# --------------------------------------------------------------------------
# Building the curated question set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratedQuestion:
    id: str
    category: str
    question: str
    is_negative: bool
    image_target: bool


def build_accepted_questions(
    passed: list[DraftRecord], decisions: CurationDecisions
) -> list[CuratedQuestion]:
    """Materialize accepted drafts as `CuratedQuestion`s, enforcing decision completeness.

    Raises:
        ValueError: if any quality-passed draft has no accept/reject
            decision (an unreviewed draft), or if a decision references
            an id that isn't a quality-passed draft (a typo, or a
            decision left over from a stale draft file).
    """
    accepted_ids = set(decisions.accepted)
    rejected_ids = set(decisions.rejected)
    passed_ids = {d.id for d in passed}

    missing = passed_ids - accepted_ids - rejected_ids
    if missing:
        raise ValueError(
            f"{len(missing)} quality-passed draft(s) have no accept/reject "
            f"decision in the curation file: {sorted(missing)}"
        )

    unknown = (accepted_ids | rejected_ids) - passed_ids
    if unknown:
        raise ValueError(
            f"Curation decisions reference unknown or filtered-out draft "
            f"id(s): {sorted(unknown)}"
        )

    by_id = {d.id: d for d in passed}
    questions: list[CuratedQuestion] = []
    for draft_id, edited_text in decisions.accepted.items():
        draft = by_id[draft_id]
        text = edited_text.strip() if edited_text else draft.question
        questions.append(
            CuratedQuestion(
                id=draft_id,
                category=draft.category,
                question=text,
                is_negative=False,
                image_target=False,
            )
        )
    return questions


def build_negative_questions(negatives: tuple[NegativeQuestion, ...]) -> list[CuratedQuestion]:
    """Materialize authored negative probes as `CuratedQuestion`s.

    Ids are assigned deterministically per category (`{category}_neg_NNN`)
    so re-running curation from the same decisions file reproduces the
    same ids.
    """
    counters: dict[str, int] = {}
    questions: list[CuratedQuestion] = []
    for neg in negatives:
        i = counters.get(neg.category, 0)
        counters[neg.category] = i + 1
        questions.append(
            CuratedQuestion(
                id=f"{neg.category}_neg_{i:03d}",
                category=neg.category,
                question=neg.question,
                is_negative=True,
                image_target=False,
            )
        )
    return questions


# --------------------------------------------------------------------------
# Image-target flagging
# --------------------------------------------------------------------------


def estimate_image_coverage(
    corpus: pd.DataFrame,
    category: TaxonomyCategory,
    text_col: str = "text",
    image_col: str = "has_image",
) -> float:
    """% of `category`-matching corpus reviews that carry an image, in [0, 100]."""
    text = corpus[text_col].fillna("").astype(str)
    matched = corpus.loc[matches_any_keyword(text, category.keywords)]
    if matched.empty:
        return 0.0
    # float(...): pandas/numpy would otherwise hand back a numpy.float64,
    # which silently turns every later ">= threshold" comparison into a
    # numpy.bool_ instead of a plain bool.
    return float(round(100 * matched[image_col].mean(), 2))


def flag_image_targets(
    questions: list[CuratedQuestion],
    corpus: pd.DataFrame,
    spec: TaxonomySpec,
    min_image_coverage_pct: float,
) -> tuple[list[CuratedQuestion], dict[str, float]]:
    """Tag non-negative questions in categories with real image-bearing evidence.

    RQ4/E7 needs a meaningful image-bearing subset of CRAGB (PLAN.md §3
    E7), so this flags which questions are *worth* checking for a
    best-evidence photo later (T2.9) — it does not assign a specific
    photo (that only becomes knowable once T2.6/T2.7 pool and label
    actual relevant reviews per question). The flag is corpus-driven
    (does this category actually have enough photographed reviews?)
    rather than an arbitrary per-question guess. Negative questions are
    never flagged: by construction they have no relevant reviews, so no
    photo either.

    Returns:
        `(flagged_questions, coverage_by_category)` — coverage is
        reported for every category regardless of the threshold, so a
        near-miss category is visible in the curation report rather than
        silently absent.
    """
    coverage_by_category = {c.name: estimate_image_coverage(corpus, c) for c in spec.categories}
    flagged = [
        replace(
            q,
            image_target=(not q.is_negative) and coverage_by_category[q.category] >= min_image_coverage_pct,
        )
        for q in questions
    ]
    return flagged, coverage_by_category


# --------------------------------------------------------------------------
# Validation against the taxonomy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurationReport:
    ok: bool
    total: int
    per_category_counts: dict[str, int]
    per_category_negative_counts: dict[str, int]
    messages: tuple[str, ...]


def validate_curation(questions: list[CuratedQuestion], spec: TaxonomySpec) -> CurationReport:
    """Check the curated set matches T2.1's taxonomy targets exactly.

    Unlike `TaxonomySpec`'s own tolerance-banded total check, per-category
    counts here must match `target_count`/`negative_count` exactly:
    curation is a deliberate, human-driven selection (not a sample), so
    drift from the agreed taxonomy should surface as a hard failure, not
    a warning.
    """
    messages: list[str] = []
    per_category = {c.name: 0 for c in spec.categories}
    per_category_negative = {c.name: 0 for c in spec.categories}

    for q in questions:
        if q.category not in per_category:
            raise ValueError(f"Question {q.id!r} has unknown category {q.category!r}")
        per_category[q.category] += 1
        if q.is_negative:
            per_category_negative[q.category] += 1

    ok = True
    for category in spec.categories:
        actual = per_category[category.name]
        if actual != category.target_count:
            ok = False
            messages.append(
                f"{category.name}: expected {category.target_count} questions, got {actual}"
            )
        actual_negative = per_category_negative[category.name]
        if actual_negative != category.negative_count:
            ok = False
            messages.append(
                f"{category.name}: expected {category.negative_count} negative "
                f"questions, got {actual_negative}"
            )

    total = len(questions)
    low, high = spec.expected_total - spec.total_tolerance, spec.expected_total + spec.total_tolerance
    if not (low <= total <= high):
        ok = False
        messages.append(f"total {total} outside [{low}, {high}] (expected {spec.expected_total})")

    if ok and not messages:
        messages.append("Curation matches taxonomy targets exactly.")

    return CurationReport(
        ok=ok,
        total=total,
        per_category_counts=per_category,
        per_category_negative_counts=per_category_negative,
        messages=tuple(messages),
    )


# --------------------------------------------------------------------------
# Orchestration + I/O
# --------------------------------------------------------------------------


def curate(
    drafts: list[DraftRecord],
    spec: TaxonomySpec,
    decisions: CurationDecisions,
    corpus: pd.DataFrame,
    min_words: int,
    max_words: int,
    require_question_mark: bool,
    min_image_coverage_pct: float,
) -> tuple[list[CuratedQuestion], CurationReport, dict[str, float]]:
    """Run the full curation pipeline: filter, decide, author negatives, flag, validate."""
    passed, auto_rejected = apply_quality_filters(drafts, min_words, max_words, require_question_mark)
    for issue in auto_rejected:
        logger.info("auto-rejected %s: %s", issue.id, issue.reason)

    accepted_questions = build_accepted_questions(passed, decisions)
    negative_questions = build_negative_questions(decisions.negatives)
    all_questions = accepted_questions + negative_questions

    flagged_questions, coverage_by_category = flag_image_targets(
        all_questions, corpus, spec, min_image_coverage_pct
    )
    report = validate_curation(flagged_questions, spec)
    return flagged_questions, report, coverage_by_category


def write_questions_jsonl(questions: list[CuratedQuestion], out_path: str | Path) -> Path:
    """Write `questions` as newline-delimited JSON, one object per line."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False))
            f.write("\n")
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/curate.yaml", help="Path to curate config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    spec = load_taxonomy(cfg["paths"]["taxonomy_config"])
    drafts = load_drafts(cfg["paths"]["drafts_in"])
    decisions = load_decisions(cfg["paths"]["decisions_in"])
    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))

    quality_cfg = cfg["quality"]
    questions, report, coverage = curate(
        drafts=drafts,
        spec=spec,
        decisions=decisions,
        corpus=corpus,
        min_words=quality_cfg["min_words"],
        max_words=quality_cfg["max_words"],
        require_question_mark=quality_cfg["require_question_mark"],
        min_image_coverage_pct=cfg["image_target"]["min_category_coverage_pct"],
    )

    out_path = write_questions_jsonl(questions, cfg["paths"]["questions_out"])
    logger.info("Wrote %d curated questions to %s", len(questions), out_path)
    for category in spec.categories:
        name = category.name
        logger.info(
            "  %s: %d (negative: %d, image coverage: %.2f%%, image_target: %s)",
            name,
            report.per_category_counts[name],
            report.per_category_negative_counts[name],
            coverage[name],
            coverage[name] >= cfg["image_target"]["min_category_coverage_pct"],
        )
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
