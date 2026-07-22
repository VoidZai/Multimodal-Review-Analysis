"""Best-evidence photo selection for CRAGB (T2.9; PLAN.md §3 E7, §1.4 risk D).

For each question T2.3 flagged `image_target: true` (a category with real
image-bearing coverage in `corpus_v1`), this module surfaces the
candidate photos worth choosing from and records the single
best-evidence pick — the photo a human reader would find most
probative for that specific question.

**Candidates are scoped to T2.7-labeled-*relevant* reviews with an
image, never the raw T2.6 pool.** This is the same "don't visually
over-claim" discipline PLAN.md risk D calls for at the generation stage,
applied one stage earlier: a photo attached to a review that was merely
*retrieved* but judged *not relevant* is not evidence for the question,
so it must never become the "best-evidence photo" shown alongside an
answer. `find_image_bearing_candidates` enforces the intersection
(relevant ∩ has_image) explicitly rather than trusting the pool's
image_target flag alone.

A question with `image_target: true` but zero image-bearing relevant
candidates is a real, reportable outcome, not an error — it means the
*category* has decent image coverage in general, but the specific
reviews that answer *this* question happen not to be photographed.
`validate_selections` requires every image-target question to have an
explicit decision either way (a doc_id, or an explicit "no candidate"),
so this can't be silently dropped from the coverage denominator.

Usage:
    python -m cragb.bench.best_photo export
    python -m cragb.bench.best_photo validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cragb.bench.reference_answers import load_relevant_doc_ids
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageQuestionRecord:
    id: str
    category: str
    question: str
    is_negative: bool
    image_target: bool


def load_image_target_questions(path: str | Path) -> list[ImageQuestionRecord]:
    """Load `cragb_questions_v1.jsonl`, keeping every field T2.9 needs.

    Unlike `cragb.bench.pooling.load_questions`, this keeps
    `image_target` (T2.3's flag) since it's what scopes this module's
    entire task — questions with `image_target: false` are out of scope
    here by design, not an oversight.
    """
    records: list[ImageQuestionRecord] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                ImageQuestionRecord(
                    id=obj["id"],
                    category=obj["category"],
                    question=obj["question"],
                    is_negative=bool(obj["is_negative"]),
                    image_target=bool(obj["image_target"]),
                )
            )
    return records


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


def find_image_bearing_candidates(
    relevant_doc_ids: set[str],
    corpus: pd.DataFrame,
    image_col: str = "has_image",
) -> list[str]:
    """doc_ids in `relevant_doc_ids` that carry at least one image, sorted for determinism."""
    wanted = {int(doc_id) for doc_id in relevant_doc_ids}
    subset = corpus.loc[corpus.index.isin(wanted) & corpus[image_col]]
    return sorted(subset.index.astype(str), key=int)


@dataclass(frozen=True)
class CandidateSnippet:
    text: str
    rating: float
    image_url: str | None


def load_candidate_snippets(
    corpus: pd.DataFrame,
    doc_ids: list[str],
    text_col: str = "text",
    image_urls_col: str = "image_urls",
    max_chars: int = 500,
) -> dict[str, CandidateSnippet]:
    """Look up review text/rating/first-image-URL for a list of candidate doc ids."""
    wanted = {int(doc_id) for doc_id in doc_ids}
    subset = corpus.loc[corpus.index.isin(wanted)]
    snippets: dict[str, CandidateSnippet] = {}
    for idx, row in subset.iterrows():
        urls = row[image_urls_col]
        first_url = str(urls[0]) if urls is not None and len(urls) > 0 else None
        snippets[str(idx)] = CandidateSnippet(
            text=str(row[text_col])[:max_chars], rating=float(row["rating"]), image_url=first_url
        )
    return snippets


# --------------------------------------------------------------------------
# Worksheet export
# --------------------------------------------------------------------------


def render_worksheet(
    questions: list[ImageQuestionRecord],
    relevant_by_question: dict[str, set[str]],
    corpus: pd.DataFrame,
    text_col: str,
    image_col: str,
    image_urls_col: str,
    max_chars: int,
) -> str:
    """Render a human-readable worksheet of image-bearing candidates per image-target question."""
    lines: list[str] = []
    lines.append("# CRAGB Best-Evidence Photo Worksheet (v1)")
    lines.append("")
    lines.append(
        "One block per `image_target: true` question, listing its "
        "T2.7-labeled-relevant, image-bearing candidate reviews. Pick the "
        "single most probative photo per question and record it in "
        "`benchmark/best_photo_v1.jsonl` — this file is a reading aid, not "
        "the selection store."
    )
    lines.append("")
    for q in questions:
        if not q.image_target:
            continue
        relevant_ids = relevant_by_question.get(q.id, set())
        candidates = find_image_bearing_candidates(relevant_ids, corpus, image_col=image_col)
        lines.append(f"## {q.id} [{q.category}]")
        lines.append(f"**Q:** {q.question}")
        lines.append("")
        if not candidates:
            lines.append("(no image-bearing relevant candidate found)")
            lines.append("")
            continue
        snippets = load_candidate_snippets(
            corpus, candidates, text_col=text_col, image_urls_col=image_urls_col, max_chars=max_chars
        )
        for doc_id in candidates:
            snippet = snippets[doc_id]
            lines.append(f"- `{doc_id}` ({snippet.rating:g}★): {snippet.text}")
            lines.append(f"  photo: {snippet.image_url}")
        lines.append("")
    return "\n".join(lines) + "\n"


def export_worksheet(
    questions: list[ImageQuestionRecord],
    relevant_by_question: dict[str, set[str]],
    corpus: pd.DataFrame,
    out_path: str | Path,
    text_col: str,
    image_col: str,
    image_urls_col: str,
    max_chars: int,
) -> Path:
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        render_worksheet(questions, relevant_by_question, corpus, text_col, image_col, image_urls_col, max_chars),
        encoding="utf-8",
    )
    return resolved


# --------------------------------------------------------------------------
# Selections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BestPhotoSelection:
    question_id: str
    doc_id: str | None  # None = explicitly no candidate available
    reason: str


def load_selections(path: str | Path) -> dict[str, BestPhotoSelection]:
    """Load the best-photo selections JSONL, keyed by question id.

    Raises:
        ValueError: on a duplicate `question_id`.
    """
    selections: dict[str, BestPhotoSelection] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["question_id"]
            if qid in selections:
                raise ValueError(f"Duplicate best-photo selection for {qid!r}")
            selections[qid] = BestPhotoSelection(
                question_id=qid, doc_id=obj["doc_id"], reason=obj.get("reason", "")
            )
    return selections


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BestPhotoReport:
    ok: bool
    n_image_target_questions: int
    n_selected: int
    n_no_candidate: tuple[str, ...]
    missing_decisions: tuple[str, ...]  # image-target question with no selection row at all
    invalid_selections: tuple[str, ...]  # picked doc_id not an image-bearing relevant candidate
    coverage_pct: float
    messages: tuple[str, ...]


def validate_selections(
    questions: list[ImageQuestionRecord],
    relevant_by_question: dict[str, set[str]],
    corpus: pd.DataFrame,
    selections: dict[str, BestPhotoSelection],
    image_col: str = "has_image",
) -> BestPhotoReport:
    """Check every `image_target` question has an explicit, valid selection.

    Hard failures (`ok=False`):
      - an `image_target: true` question with no selection row at all
        (a silent gap would understate coverage without anyone noticing);
      - a selection whose `doc_id` is not actually in that question's
        (relevant ∩ has_image) candidate set — the core guarantee that
        the surfaced photo is real evidence, not just any pooled image.

    Not a failure, and expected to happen sometimes: an `image_target`
    question whose candidate set is genuinely empty, recorded with
    `doc_id: null` — reported honestly in `n_no_candidate` /
    `coverage_pct`, not hidden.
    """
    target_questions = [q for q in questions if q.image_target]
    messages: list[str] = []

    missing = tuple(q.id for q in target_questions if q.id not in selections)
    if missing:
        messages.append(f"{len(missing)} image-target question(s) have no selection decision: {list(missing)}")

    no_candidate: list[str] = []
    invalid: list[str] = []
    n_selected = 0

    for q in target_questions:
        selection = selections.get(q.id)
        if selection is None:
            continue
        relevant_ids = relevant_by_question.get(q.id, set())
        valid_candidates = set(find_image_bearing_candidates(relevant_ids, corpus, image_col=image_col))

        if selection.doc_id is None:
            if valid_candidates:
                invalid.append(q.id)
                messages.append(
                    f"{q.id}: selection says no candidate, but {len(valid_candidates)} "
                    f"image-bearing relevant candidate(s) exist: {sorted(valid_candidates)}"
                )
            else:
                no_candidate.append(q.id)
        else:
            if selection.doc_id not in valid_candidates:
                invalid.append(q.id)
                messages.append(
                    f"{q.id}: selected doc_id {selection.doc_id!r} is not an image-bearing "
                    f"relevant candidate (candidates: {sorted(valid_candidates)})"
                )
            else:
                n_selected += 1

    total = len(target_questions)
    coverage_pct = round(100 * n_selected / total, 2) if total else 0.0

    ok = not (missing or invalid)
    if ok and not messages:
        messages.append("Every image-target question has a valid, evidence-grounded selection.")

    return BestPhotoReport(
        ok=ok,
        n_image_target_questions=total,
        n_selected=n_selected,
        n_no_candidate=tuple(no_candidate),
        missing_decisions=missing,
        invalid_selections=tuple(invalid),
        coverage_pct=coverage_pct,
        messages=tuple(messages),
    )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def write_best_photo_jsonl(
    questions: list[ImageQuestionRecord],
    selections: dict[str, BestPhotoSelection],
    out_path: str | Path,
) -> Path:
    """Write selections as newline-delimited JSON, in question order, image-target only."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for q in questions:
            if not q.image_target:
                continue
            selection = selections.get(q.id)
            if selection is None:
                continue
            row = {
                "question_id": selection.question_id,
                "doc_id": selection.doc_id,
                "reason": selection.reason,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return resolved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/best_photo.yaml", help="Path to best-photo config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("command", choices=["export", "validate"], help="export the worksheet, or validate selections")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    questions = load_image_target_questions(cfg["paths"]["questions_in"])
    relevant_by_question = load_relevant_doc_ids(cfg["paths"]["labels_in"])
    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    corpus_cfg = cfg["corpus"]

    if args.command == "export":
        out_path = export_worksheet(
            questions,
            relevant_by_question,
            corpus,
            cfg["paths"]["worksheet_out"],
            text_col=corpus_cfg["text_col"],
            image_col=corpus_cfg["image_col"],
            image_urls_col=corpus_cfg["image_urls_col"],
            max_chars=cfg["worksheet"]["max_chars"],
        )
        n_targets = sum(1 for q in questions if q.image_target)
        logger.info("Wrote worksheet (%d image-target questions) to %s", n_targets, out_path)
        return 0

    # args.command == "validate"
    selections = load_selections(cfg["paths"]["selections_in"])
    report = validate_selections(questions, relevant_by_question, corpus, selections, image_col=corpus_cfg["image_col"])

    write_best_photo_jsonl(questions, selections, cfg["paths"]["selections_out"])
    logger.info(
        "Selected %d/%d image-target questions (%.2f%% coverage); %d with no image-bearing "
        "relevant candidate",
        report.n_selected,
        report.n_image_target_questions,
        report.coverage_pct,
        len(report.n_no_candidate),
    )
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
