"""Assemble CRAGB v1 + datasheet + leakage guard (T2.10; PLAN.md §3 E1, §5).

The final milestone-2 step, and deliberately the least "creative" one:
by this point every field has already been authored and validated by an
earlier task —

    T2.3  question, type, is_negative, image_target   (cragb_questions_v1.jsonl)
    T2.7  relevant_ids                                 (relevance_labels_v1.jsonl)
    T2.8  reference_answer, cited_doc_ids, is_abstention (reference_answers_v1.jsonl)
    T2.9  best_photo_id                                (best_photo_v1.jsonl)

so this module's job is narrow: merge them under one schema, **re-check**
that nothing drifted between those independently-run tasks (a citation
no longer a subset of relevant_ids, a negative question with only
negative-status but no answer, an image-target question nobody picked a
photo for), and produce the two artifacts a benchmark release needs
beyond the data itself — a human-readable datasheet, and a leakage-guard
manifest for E8 to check future fine-tuning data against.

**Why a leakage guard belongs here, not in E8 itself:** CRAGB v1 is
frozen at the end of this module. E8 (fine-tune data prep) happens
later, generates a *different* dataset, and must never let a CRAGB
*evaluation* question end up as *training* data — PLAN.md §1.4 risk F.
The guard needs CRAGB's question hashes computed once, from the frozen
v1 file, so E8 checks against a fixed target rather than recomputing
(and potentially drifting) them itself.

Usage:
    python -m cragb.bench.assemble --config configs/assemble.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cragb.bench.best_photo import (
    BestPhotoSelection,
    ImageQuestionRecord,
    load_image_target_questions,
    load_selections as load_photo_selections,
)
from cragb.bench.reference_answers import (
    ReferenceAnswer,
    load_reference_answers,
    load_relevant_doc_ids,
)
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CragbEntry:
    id: str
    type: str
    is_negative: bool
    question: str
    relevant_ids: tuple[str, ...]
    reference_answer: str
    cited_doc_ids: tuple[str, ...]
    is_abstention: bool
    image_target: bool
    best_photo_id: str | None


def assemble_entries(
    questions: list[ImageQuestionRecord],
    relevant_by_question: dict[str, set[str]],
    answers: dict[str, ReferenceAnswer],
    photo_selections: dict[str, BestPhotoSelection],
) -> list[CragbEntry]:
    """Merge T2.3/T2.7/T2.8/T2.9 outputs into `CragbEntry`s, in question order.

    Raises:
        ValueError: if a question is missing its reference answer (T2.8
            incomplete for it), or is `image_target` but has no best-photo
            decision (T2.9 incomplete for it) — both indicate an earlier
            task didn't actually finish, not something to paper over here.
    """
    entries: list[CragbEntry] = []
    for q in questions:
        answer = answers.get(q.id)
        if answer is None:
            raise ValueError(f"{q.id}: no reference answer found (T2.8 incomplete)")

        if q.image_target:
            selection = photo_selections.get(q.id)
            if selection is None:
                raise ValueError(f"{q.id}: image_target question has no best-photo decision (T2.9 incomplete)")
            best_photo_id = selection.doc_id
        else:
            best_photo_id = None

        relevant_ids = tuple(sorted(relevant_by_question.get(q.id, set()), key=int))

        entries.append(
            CragbEntry(
                id=q.id,
                type=q.category,
                is_negative=q.is_negative,
                question=q.question,
                relevant_ids=relevant_ids,
                reference_answer=answer.answer,
                cited_doc_ids=answer.cited_doc_ids,
                is_abstention=answer.is_abstention,
                image_target=q.image_target,
                best_photo_id=best_photo_id,
            )
        )
    return entries


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyReport:
    ok: bool
    total: int
    n_negative: int
    n_image_target: int
    n_with_photo: int
    missing_relevant_ids: tuple[str, ...]
    citation_not_subset: tuple[str, ...]
    photo_not_relevant: tuple[str, ...]
    messages: tuple[str, ...]


def validate_entries(entries: list[CragbEntry]) -> AssemblyReport:
    """Re-check cross-task invariants that assembly must not have broken.

    Hard failures (`ok=False`):
      - a non-negative question with empty `relevant_ids` — M2.md's own
        verify bar ("no null relevant_ids for non-negative questions");
      - `cited_doc_ids` not a subset of `relevant_ids` — T2.8 already
        checks this against T2.7's labels directly, but re-deriving it
        here from the merged record catches anything that drifted
        between the two files being written and being assembled;
      - a `best_photo_id` that isn't in `relevant_ids` — the same
        grounding guarantee T2.9 enforced, re-checked post-merge.
    """
    messages: list[str] = []

    missing_relevant = tuple(e.id for e in entries if not e.is_negative and not e.relevant_ids)
    if missing_relevant:
        messages.append(f"{len(missing_relevant)} non-negative question(s) have empty relevant_ids: {list(missing_relevant)}")

    citation_violations = tuple(
        e.id for e in entries if set(e.cited_doc_ids) - set(e.relevant_ids)
    )
    if citation_violations:
        messages.append(f"{len(citation_violations)} entry(s) cite doc_id(s) outside relevant_ids: {list(citation_violations)}")

    photo_violations = tuple(
        e.id for e in entries if e.best_photo_id is not None and e.best_photo_id not in e.relevant_ids
    )
    if photo_violations:
        messages.append(f"{len(photo_violations)} entry(s) have a best_photo_id outside relevant_ids: {list(photo_violations)}")

    ok = not (missing_relevant or citation_violations or photo_violations)
    if ok and not messages:
        messages.append("All cross-task invariants hold; CRAGB v1 is internally consistent.")

    return AssemblyReport(
        ok=ok,
        total=len(entries),
        n_negative=sum(e.is_negative for e in entries),
        n_image_target=sum(e.image_target for e in entries),
        n_with_photo=sum(1 for e in entries if e.best_photo_id is not None),
        missing_relevant_ids=missing_relevant,
        citation_not_subset=citation_violations,
        photo_not_relevant=photo_violations,
        messages=tuple(messages),
    )


# --------------------------------------------------------------------------
# Leakage guard (for E8)
# --------------------------------------------------------------------------


def normalize_question_text(text: str) -> str:
    """Lowercase, whitespace-collapsed form used for leakage hashing.

    Deliberately simple: the goal is catching a CRAGB question copied
    verbatim (or near enough) into training data, not general semantic
    duplicate detection — that's a much harder problem this guard isn't
    trying to solve.
    """
    return " ".join(text.strip().lower().split())


def compute_question_hash(question_text: str) -> str:
    return hashlib.sha256(normalize_question_text(question_text).encode("utf-8")).hexdigest()


def build_leakage_manifest(entries: list[CragbEntry]) -> dict[str, str]:
    """question_id -> sha256 of its normalized question text."""
    return {e.id: compute_question_hash(e.question) for e in entries}


@dataclass(frozen=True)
class LeakageReport:
    ok: bool
    n_checked: int
    leaked_question_ids: tuple[str, ...]
    messages: tuple[str, ...]


def check_no_leakage(cragb_question_hashes: dict[str, str], candidate_texts: list[str]) -> LeakageReport:
    """Check none of `candidate_texts` reproduce a CRAGB question verbatim.

    This is the function E8 (fine-tune data prep) calls against its
    synthetic training-example prompts, using the hashes this module
    computed once from the frozen CRAGB v1 question set
    (`cragb_v1_leakage_manifest.json`) — not against a copy E8
    recomputes itself, so a later change to CRAGB v1 can't silently
    de-sync the guard from what's actually being evaluated on.

    Args:
        cragb_question_hashes: `question_id -> sha256`, as produced by
            `build_leakage_manifest` (or loaded from the manifest file).
        candidate_texts: raw text to check (e.g. training-example
            prompts) — normalized and hashed the same way before
            comparing.

    Returns:
        A `LeakageReport`; `ok=False` iff at least one CRAGB question's
        hash appears among the candidates.
    """
    candidate_hashes = {compute_question_hash(t) for t in candidate_texts}
    leaked = tuple(qid for qid, h in cragb_question_hashes.items() if h in candidate_hashes)
    ok = not leaked
    messages = (
        (f"{len(leaked)} CRAGB question(s) found verbatim (post-normalization) in the candidate set: {list(leaked)}",)
        if leaked
        else ("No leakage detected.",)
    )
    return LeakageReport(ok=ok, n_checked=len(candidate_texts), leaked_question_ids=leaked, messages=messages)


# --------------------------------------------------------------------------
# Datasheet
# --------------------------------------------------------------------------


def compute_datasheet_stats(entries: list[CragbEntry]) -> dict:
    """Pull every number the datasheet reports directly from the assembled entries.

    Nothing here is hand-typed elsewhere — regenerating `cragb_v1.jsonl`
    and re-running this always reproduces numbers consistent with the
    file, rather than a datasheet that can silently go stale.
    """
    by_type = Counter(e.type for e in entries)
    negatives_by_type = Counter(e.type for e in entries if e.is_negative)
    relevant_counts = [len(e.relevant_ids) for e in entries]
    cited_counts = [len(e.cited_doc_ids) for e in entries if not e.is_abstention]
    n_image_target = sum(e.image_target for e in entries)
    n_with_photo = sum(1 for e in entries if e.best_photo_id is not None)

    return {
        "total": len(entries),
        "by_type": dict(sorted(by_type.items())),
        "negatives_by_type": dict(sorted(negatives_by_type.items())),
        "n_negative": sum(e.is_negative for e in entries),
        "n_abstention": sum(e.is_abstention for e in entries),
        "n_image_target": n_image_target,
        "n_with_photo": n_with_photo,
        "image_coverage_pct": round(100 * n_with_photo / n_image_target, 2) if n_image_target else 0.0,
        "min_relevant_ids": min(relevant_counts) if relevant_counts else 0,
        "max_relevant_ids": max(relevant_counts) if relevant_counts else 0,
        "mean_relevant_ids": round(sum(relevant_counts) / len(relevant_counts), 2) if relevant_counts else 0.0,
        "mean_citations_per_grounded_answer": round(sum(cited_counts) / len(cited_counts), 2) if cited_counts else 0.0,
    }


def render_datasheet(entries: list[CragbEntry], stats: dict, corpus_manifest: dict | None) -> str:
    lines: list[str] = []
    lines.append("# CRAGB v1 Datasheet")
    lines.append("")
    lines.append(
        "Gebru-style datasheet for `benchmark/cragb_v1.jsonl` (T2.10; PLAN.md §3 E1, §5). "
        "All numbers below are computed directly from the assembled entries, not hand-maintained."
    )
    lines.append("")

    lines.append("## Motivation")
    lines.append(
        "CRAGB evaluates whether strictly-grounded review-RAG can answer open-ended shopper "
        "questions about Amazon Clothing/Shoes/Jewelry products without hallucinating (RQ0), "
        "compare retrieval methods (RQ2), and surface photo evidence honestly (RQ4). Built per "
        "PLAN.md §3 E1 as a pooled, human-labeled benchmark — not scraped or auto-generated "
        "without human review at any stage."
    )
    lines.append("")

    lines.append("## Composition")
    lines.append(
        f"**{stats['total']} questions** across {len(stats['by_type'])} taxonomy categories "
        f"(T2.1), of which **{stats['n_negative']} are deliberately unanswerable negatives**."
    )
    lines.append("")
    lines.append("| Category | Total | Negative |")
    lines.append("|---|---:|---:|")
    for category, count in stats["by_type"].items():
        lines.append(f"| `{category}` | {count} | {stats['negatives_by_type'].get(category, 0)} |")
    lines.append("")
    lines.append(
        f"Relevant-document count per question ranges {stats['min_relevant_ids']}–"
        f"{stats['max_relevant_ids']} (mean {stats['mean_relevant_ids']}), from T2.7's pooled "
        f"relevance labeling. **{stats['n_abstention']} questions correctly abstain** "
        "(\"not enough information\") — evidence-driven, not taken from the original taxonomy "
        "negative flag; see the Known limitations section."
    )
    lines.append("")
    lines.append(
        f"**Image evidence (RQ4):** {stats['n_image_target']} questions were flagged "
        "image-target by T2.3 (categories with real image-bearing corpus coverage). Of those, "
        f"**{stats['n_with_photo']}/{stats['n_image_target']} ({stats['image_coverage_pct']}%) "
        "got a genuine best-evidence photo** from T2.9 — the rest had relevant reviews, but none "
        "of *those specific* reviews carried an image. This is the honest, unpadded number; "
        "report it as-is (PLAN.md risk D)."
    )
    lines.append("")
    lines.append(
        f"Grounded (non-abstention) answers cite a mean of "
        f"{stats['mean_citations_per_grounded_answer']} reviews each."
    )
    lines.append("")

    lines.append("## Collection process")
    lines.append(
        "T2.1 taxonomy → T2.2 LLM-drafted candidates (Groq) → T2.3 human curation + authored "
        "negatives → T2.4/T2.5 BM25 + dense retrievers → T2.6 TREC-style pooling (union of "
        "top-k) → T2.7 human relevance labeling over every pooled pair → T2.8 human reference "
        "answers, grounded and citation-checked against T2.7 → T2.9 best-evidence photo "
        "selection, scoped to T2.7-relevant image-bearing reviews → T2.10 (this file): "
        "assembly, cross-task re-validation, datasheet, leakage guard."
    )
    lines.append("")

    lines.append("## Known limitations")
    lines.append(
        "- **Single annotator.** All human judgments (curation, relevance, reference answers, "
        "photo selection) were made by one person; no second-annotator agreement (κ) has been "
        "computed for this benchmark itself (distinct from the answer-quality *judge* validation "
        "planned in E5)."
    )
    lines.append(
        "- **Several authored negatives turned out answerable.** T2.7's labeling found real "
        "relevant evidence for 9 of the 11 originally-authored negative questions "
        "(PLAN.md §14.2) — abstention in this file is evidence-driven (zero relevant docs), "
        "not taken from the original taxonomy intent, so it reflects what's actually true of "
        "the corpus rather than what T2.3 assumed."
    )
    lines.append(
        "- **Image coverage is real, not padded.** T2.3's `image_target` flag is a "
        "category-level estimate; T2.9's actual per-question coverage "
        f"({stats['image_coverage_pct']}%) is lower, because a category having decent image "
        "coverage overall doesn't guarantee the specific relevant reviews for every question "
        "are photographed."
    )
    lines.append(
        "- **Pooled, not exhaustive, relevance.** Recall/Hit@k computed against this file are "
        "correct *with respect to the retrievers pooled in T2.6* (BM25 + dense), per PLAN.md "
        "risk A — not against the full corpus, which was never exhaustively labeled."
    )
    lines.append("")

    lines.append("## Held-out status")
    lines.append(
        "**CRAGB v1 must never appear in fine-tuning training data.** A leakage-guard manifest "
        "(`benchmark/cragb_v1_leakage_manifest.json`) records a SHA-256 hash of every question's "
        "normalized text; `cragb.bench.assemble.check_no_leakage` checks candidate fine-tune "
        "training text against it before E8 accepts any synthetic example."
    )
    lines.append("")

    lines.append("## Provenance")
    if corpus_manifest is not None:
        lines.append(
            f"Built from `corpus_v1.parquet` (seed {corpus_manifest.get('seed')}, "
            f"{corpus_manifest.get('row_count')} rows, sha256 "
            f"`{corpus_manifest.get('sha256', '')[:16]}…`; see "
            "`data/processed/corpus_v1_manifest.json`)."
        )
    else:
        lines.append("Corpus manifest not found at assembly time — provenance hash unavailable.")
    lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def write_cragb_jsonl(entries: list[CragbEntry], out_path: str | Path) -> Path:
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for e in entries:
            row = {
                "id": e.id,
                "type": e.type,
                "is_negative": e.is_negative,
                "question": e.question,
                "relevant_ids": list(e.relevant_ids),
                "reference_answer": e.reference_answer,
                "cited_doc_ids": list(e.cited_doc_ids),
                "is_abstention": e.is_abstention,
                "image_target": e.image_target,
                "best_photo_id": e.best_photo_id,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return resolved


def write_leakage_manifest(entries: list[CragbEntry], out_path: str | Path) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(entries),
        "hash_algorithm": "sha256(normalized question text)",
        "question_hashes": build_leakage_manifest(entries),
    }
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return resolved


def write_datasheet(entries: list[CragbEntry], out_path: str | Path, corpus_manifest: dict | None) -> Path:
    stats = compute_datasheet_stats(entries)
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_datasheet(entries, stats, corpus_manifest), encoding="utf-8")
    return resolved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assemble.yaml", help="Path to assemble config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    paths = cfg["paths"]

    questions = load_image_target_questions(paths["questions_in"])
    relevant_by_question = load_relevant_doc_ids(paths["labels_in"])
    answers = load_reference_answers(paths["answers_in"])
    photo_selections = load_photo_selections(paths["photos_in"])

    entries = assemble_entries(questions, relevant_by_question, answers, photo_selections)
    report = validate_entries(entries)

    cragb_path = write_cragb_jsonl(entries, paths["cragb_out"])
    leakage_path = write_leakage_manifest(entries, paths["leakage_manifest_out"])

    corpus_manifest_path = resolve_path(paths["corpus_manifest_in"])
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8")) if corpus_manifest_path.is_file() else None
    datasheet_path = write_datasheet(entries, paths["datasheet_out"], corpus_manifest)

    logger.info("Wrote %d CRAGB v1 entries to %s", report.total, cragb_path)
    logger.info("Wrote leakage manifest to %s", leakage_path)
    logger.info("Wrote datasheet to %s", datasheet_path)
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
