"""Load CRAGB questions for retrieval evaluation (T3.4; PLAN.md §3 E2/E3).

The final assembled benchmark (`benchmark/cragb_v1.jsonl`, T2.10) carries
fields for every downstream use — reference answers, citation ids, best
photo — but retrieval evaluation (the chunking study T3.4, RQ2 in T3.6+)
only ever needs a question's id, taxonomy type, and its pooled relevant
document ids. `cragb.bench.assemble.CragbEntry` exists already, but it's
the assembly pipeline's own internal shape (coupled to the fields it
*writes*, e.g. `reference_answer`, `cited_doc_ids`, `best_photo_id`);
reading the finished artifact back for retrieval eval is a different,
narrower need, so this module defines its own minimal, read-only view
rather than importing across from `cragb.bench` into `cragb.eval`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cragb.utils.io import resolve_path


@dataclass(frozen=True)
class RetrievalQuestion:
    """The slice of a CRAGB question that retrieval evaluation needs."""

    id: str
    type: str
    question: str
    is_negative: bool
    relevant_ids: frozenset[str]


def load_retrieval_questions(
    path: str | Path = "benchmark/cragb_v1.jsonl",
) -> list[RetrievalQuestion]:
    """Load every question in `path` (default: `cragb_v1.jsonl`) as `RetrievalQuestion`s.

    Args:
        path: path to a CRAGB jsonl file, one question object per line,
            absolute or relative to the repo root.

    Returns:
        One `RetrievalQuestion` per line, in file order. Includes
        negatives with empty `relevant_ids` — use `filter_scorable` to
        drop those before computing retrieval metrics.

    Raises:
        FileNotFoundError: if `path` does not exist.
    """
    questions: list[RetrievalQuestion] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(
                RetrievalQuestion(
                    id=obj["id"],
                    type=obj["type"],
                    question=obj["question"],
                    is_negative=bool(obj["is_negative"]),
                    relevant_ids=frozenset(obj["relevant_ids"]),
                )
            )
    return questions


def filter_scorable(questions: list[RetrievalQuestion]) -> list[RetrievalQuestion]:
    """Questions with at least one relevant document.

    `cragb.eval.metrics_retrieval` raises `ValueError` on empty
    `relevant_ids` deliberately (Recall/Hit/nDCG/MRR are undefined with
    no relevant documents to find) — this is the filter every retrieval
    eval caller (T3.4, T3.6+) applies before scoring, so a genuinely
    unanswerable CRAGB negative (e.g. `fabric_quality_neg_000`,
    `defects_neg_000` — see PLAN.md §14.2) is excluded from Recall/nDCG/
    MRR/Hit averages rather than crashing the eval run or silently
    contributing an undefined value.
    """
    return [q for q in questions if q.relevant_ids]
