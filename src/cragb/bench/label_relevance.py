"""Human relevance labeling over pooled candidates (T2.7; PLAN.md §3 E1, §1.4 bottleneck #2).

T2.6 pooled the union of BM25 + dense top-k per question — but pooling
only *proposes* candidates; it says nothing about whether any of them
actually answer the question (top-k retrieval returns its k
highest-scoring documents whether or not they're relevant). This module
is the other half of the TREC pooling protocol: every (question, doc_id)
pair in every pool gets an explicit human relevant/not-relevant label,
so Recall/Hit@k in E3/E5 are computed against trustworthy, complete
judgments instead of silently treating an unlabeled true positive as a
miss.

This is the project's single scarcest resource (PLAN.md §1.4 bottleneck
#2: "keep the question count at the lower end done *well* rather than
100 done shallowly") — at ~20 pooled candidates x 60 questions, that's
over a thousand individual judgments. Nothing here is auto-labeled: a
keyword-overlap heuristic would defeat the entire point of building
trustworthy ground truth (it's also what the *retrievers themselves*
already do, so it would just be pooling twice under a different name).

Two-step workflow:
  1. `export` — render `benchmark/relevance_worksheet_v1.md`, a
     human-readable worksheet grouping every question with its pooled
     candidates' review text, for a labeler to read from.
  2. A human (or, in this project's solo-author context, the author
     reading every pooled review against its question) writes
     `benchmark/relevance_labels_v1.jsonl` — one `{question_id, doc_id,
     relevant, reason}` row per pooled pair.
  3. `validate` — checks every pooled pair got a label (no silent gaps),
     computes per-question positive counts and the "% questions with
     >=1 relevant doc" stat PLAN.md §3-E1 requires, and flags anomalies:
     a non-negative question with zero relevant docs (it may need to be
     re-pooled at a higher k, rewritten, or reclassified as a deliberate
     negative), and a *negative* question that unexpectedly got a
     positive label (the authored negative wasn't actually unanswerable
     after all — see T2.3's `curation_decisions_v1.yaml`).

Usage:
    python -m cragb.bench.label_relevance export
    python -m cragb.bench.label_relevance validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pools + corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolRecord:
    question_id: str
    category: str
    is_negative: bool
    question: str
    doc_ids: tuple[str, ...]


def load_pools(pools_path: str | Path, questions_path: str | Path) -> list[PoolRecord]:
    """Load T2.6's `pools_v1.jsonl`, joined with question text from T2.3.

    Pools carry `question_id`/`category`/`is_negative`/`doc_ids` but not
    the question text itself (T2.6 didn't need it downstream); the
    worksheet does, so it's joined in here from `cragb_questions_v1.jsonl`.
    """
    questions: dict[str, str] = {}
    with resolve_path(questions_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions[obj["id"]] = obj["question"]

    records: list[PoolRecord] = []
    with resolve_path(pools_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["question_id"]
            if qid not in questions:
                raise ValueError(f"Pool {qid!r} has no matching entry in {questions_path}")
            records.append(
                PoolRecord(
                    question_id=qid,
                    category=obj["category"],
                    is_negative=bool(obj["is_negative"]),
                    question=questions[qid],
                    doc_ids=tuple(obj["doc_ids"]),
                )
            )
    return records


@dataclass(frozen=True)
class CorpusSnippet:
    text: str
    rating: float
    has_image: bool


def load_corpus_snippets(
    corpus: pd.DataFrame,
    doc_ids: set[str],
    text_col: str = "text",
    max_chars: int = 600,
) -> dict[str, CorpusSnippet]:
    """Look up review text/rating/image-presence for a set of doc ids.

    Doc ids are the corpus `DataFrame`'s index, stringified (see
    `cragb.retrieval.base.Retriever.index`'s `id_col=None` convention,
    which T2.6's pooling run used) — so lookup is `corpus.loc[int(doc_id)]`.
    """
    wanted = {int(doc_id) for doc_id in doc_ids}
    subset = corpus.loc[corpus.index.isin(wanted)]
    return {
        str(idx): CorpusSnippet(
            text=str(row[text_col])[:max_chars],
            rating=float(row["rating"]),
            has_image=bool(row["has_image"]),
        )
        for idx, row in subset.iterrows()
    }


# --------------------------------------------------------------------------
# Worksheet export
# --------------------------------------------------------------------------


def render_worksheet(pools: list[PoolRecord], snippets: dict[str, CorpusSnippet]) -> str:
    """Render a human-readable Markdown worksheet, grouped by question."""
    lines: list[str] = []
    lines.append("# CRAGB Relevance Labeling Worksheet (v1)")
    lines.append("")
    lines.append(
        "One block per pooled question; each numbered candidate is a review "
        "text a retriever surfaced for that question (T2.6). For each "
        "candidate, judge: does this review's text actually support an "
        "answer to the question? Labels go in `benchmark/relevance_labels_v1.jsonl`, "
        "not in this file — this is a reading aid, not the label store."
    )
    lines.append("")
    for pool in pools:
        tag = " (NEGATIVE — expected to have few/no relevant candidates)" if pool.is_negative else ""
        lines.append(f"## {pool.question_id} [{pool.category}]{tag}")
        lines.append(f"**Q:** {pool.question}")
        lines.append("")
        for doc_id in pool.doc_ids:
            snippet = snippets.get(doc_id)
            if snippet is None:
                lines.append(f"- `{doc_id}`: (text not found in corpus sample)")
                continue
            image_tag = " [has image]" if snippet.has_image else ""
            lines.append(f"- `{doc_id}` ({snippet.rating:g}★{image_tag}): {snippet.text}")
        lines.append("")
    return "\n".join(lines) + "\n"


def export_worksheet(
    pools: list[PoolRecord], corpus: pd.DataFrame, out_path: str | Path, text_col: str, max_chars: int
) -> Path:
    """Render and write the labeling worksheet."""
    all_doc_ids = {doc_id for pool in pools for doc_id in pool.doc_ids}
    snippets = load_corpus_snippets(corpus, all_doc_ids, text_col=text_col, max_chars=max_chars)
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_worksheet(pools, snippets), encoding="utf-8")
    return resolved


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelDecision:
    question_id: str
    doc_id: str
    relevant: bool
    reason: str


def load_labels(path: str | Path) -> dict[tuple[str, str], LabelDecision]:
    """Load a labels JSONL into a `(question_id, doc_id) -> LabelDecision` map.

    Raises:
        ValueError: if the file contains a duplicate `(question_id,
            doc_id)` pair — a sign of a labeling-file editing mistake,
            not something to silently overwrite.
    """
    labels: dict[tuple[str, str], LabelDecision] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["question_id"], obj["doc_id"])
            if key in labels:
                raise ValueError(f"Duplicate label for {key} in {path}")
            labels[key] = LabelDecision(
                question_id=obj["question_id"],
                doc_id=obj["doc_id"],
                relevant=bool(obj["relevant"]),
                reason=obj.get("reason", ""),
            )
    return labels


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelingReport:
    ok: bool
    total_pairs: int
    labeled_pairs: int
    missing_pairs: tuple[tuple[str, str], ...]
    n_questions: int
    n_questions_with_relevant: int
    pct_questions_with_relevant: float
    zero_positive_questions: tuple[str, ...]  # non-negative, 0 relevant docs
    unexpected_positive_negatives: tuple[str, ...]  # negative, >=1 relevant doc
    messages: tuple[str, ...]


def validate_labels(pools: list[PoolRecord], labels: dict[tuple[str, str], LabelDecision]) -> LabelingReport:
    """Check label completeness and compute the required coverage stats.

    Hard failure (`ok=False`): any pooled `(question_id, doc_id)` pair
    has no label — per M2.md T2.7, every pair in every pool must be
    labeled, not just a convenient subset.

    Flags (don't affect `ok`, but are reported): a non-negative question
    with zero relevant docs in its pool (M2.md's own verify bar: "else
    the question is unanswerable and should be flagged as a deliberate
    negative or dropped" — this is exactly that check), and a negative
    question that unexpectedly got a positive label (worth revisiting
    whether that authored negative, from T2.3, is actually unanswerable).
    """
    all_pairs: list[tuple[str, str]] = [
        (pool.question_id, doc_id) for pool in pools for doc_id in pool.doc_ids
    ]
    missing = tuple(pair for pair in all_pairs if pair not in labels)

    positive_count_by_question: dict[str, int] = {pool.question_id: 0 for pool in pools}
    for (question_id, _doc_id), decision in labels.items():
        if decision.relevant and question_id in positive_count_by_question:
            positive_count_by_question[question_id] += 1

    is_negative_by_question = {pool.question_id: pool.is_negative for pool in pools}

    zero_positive = tuple(
        qid
        for qid, count in positive_count_by_question.items()
        if count == 0 and not is_negative_by_question[qid]
    )
    unexpected_positive_negatives = tuple(
        qid
        for qid, count in positive_count_by_question.items()
        if count > 0 and is_negative_by_question[qid]
    )

    n_questions = len(pools)
    n_with_relevant = sum(1 for count in positive_count_by_question.values() if count > 0)
    pct_with_relevant = round(100 * n_with_relevant / n_questions, 2) if n_questions else 0.0

    messages: list[str] = []
    if missing:
        messages.append(f"{len(missing)} pooled pair(s) have no label: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if zero_positive:
        messages.append(
            f"{len(zero_positive)} answerable question(s) have zero relevant docs "
            f"(consider re-pooling at higher k, rewriting, or reclassifying as negative): {list(zero_positive)}"
        )
    if unexpected_positive_negatives:
        messages.append(
            f"{len(unexpected_positive_negatives)} negative question(s) unexpectedly have a "
            f"relevant doc (re-check whether these are actually unanswerable): {list(unexpected_positive_negatives)}"
        )
    if not messages:
        messages.append("All pairs labeled; every answerable question has >=1 relevant doc.")

    return LabelingReport(
        ok=not missing,
        total_pairs=len(all_pairs),
        labeled_pairs=len(all_pairs) - len(missing),
        missing_pairs=missing,
        n_questions=n_questions,
        n_questions_with_relevant=n_with_relevant,
        pct_questions_with_relevant=pct_with_relevant,
        zero_positive_questions=zero_positive,
        unexpected_positive_negatives=unexpected_positive_negatives,
        messages=tuple(messages),
    )


def write_labels_jsonl(
    pools: list[PoolRecord], labels: dict[tuple[str, str], LabelDecision], out_path: str | Path
) -> Path:
    """Write the final `relevance_labels_v1.jsonl`, one row per pooled pair.

    Only pairs that are both pooled *and* labeled are written — a stray
    label for a doc_id no longer in any pool (e.g. after re-pooling)
    is silently dropped rather than propagated into the benchmark.
    """
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for pool in pools:
            for doc_id in pool.doc_ids:
                decision = labels.get((pool.question_id, doc_id))
                if decision is None:
                    continue
                row = {
                    "question_id": decision.question_id,
                    "doc_id": decision.doc_id,
                    "relevant": int(decision.relevant),
                    "reason": decision.reason,
                }
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
    return resolved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/labeling.yaml", help="Path to labeling config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("command", choices=["export", "validate"], help="export the worksheet, or validate labels")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    pools = load_pools(cfg["paths"]["pools_in"], cfg["paths"]["questions_in"])

    if args.command == "export":
        corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
        out_path = export_worksheet(
            pools,
            corpus,
            cfg["paths"]["worksheet_out"],
            text_col=cfg["corpus"]["text_col"],
            max_chars=cfg["worksheet"]["max_chars"],
        )
        logger.info("Wrote worksheet (%d questions) to %s", len(pools), out_path)
        return 0

    # args.command == "validate"
    labels = load_labels(cfg["paths"]["labels_in"])
    report = validate_labels(pools, labels)

    write_labels_jsonl(pools, labels, cfg["paths"]["labels_out"])
    logger.info(
        "Labeled %d/%d pooled pairs across %d questions (%.2f%% of questions have >=1 relevant doc)",
        report.labeled_pairs,
        report.total_pairs,
        report.n_questions,
        report.pct_questions_with_relevant,
    )
    for msg in report.messages:
        logger.info("  - %s", msg)
    print("PASS" if report.ok else "FAIL")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
