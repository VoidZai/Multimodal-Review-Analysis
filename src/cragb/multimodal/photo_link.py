"""Surfaced-vs-control photo pairing for the multimodal pilot (T6.3; PLAN.md
§3 E7, M6.md T6.3).

E7's vision judge (T6.4) needs, for every image-target CRAGB question, two
things: the photo the RAG-small pipeline would *actually* surface next to
its answer, and a random control photo to judge it against. This module
builds exactly that pair set, plus the coverage funnel that makes the
pilot's honesty auditable — RQ4's whole framing (PLAN.md §1.4 risk D,
"evidence surfacing, not image-conditioned generation") depends on
measuring the system as built, not an oracle.

**"Surfaced" means the pipeline's own retrieval rank, not a citation.**
`answer_gen_rag_small_v1.jsonl`'s `cited_photo_ids` is empty across the
inspected transcripts (`reports/grounded_qa_transcripts_v1.md` already
flags the model under-using `[photo of doc_id]` citations even when the
prompt permits them) — keying on citations would collapse the usable
sample to near zero. `surfaced_photo` instead walks `context_doc_ids` in
the order T4a.2's `context_builder` actually retrieved them (confirmed:
this matches `pools_v1.jsonl`'s per-retriever rank order) and takes the
first one flagged `context_photo_flags[doc_id] == True` — the photo a
user would actually see if this pipeline shipped today.

**A photo-bearing doc that turns out to be unfetchable is skipped, not
fatal.** T6.1 already fetched 99.2% of the relevant candidate set, so this
is rare, but `surfaced_photo` keeps scanning to the next photo-bearing doc
in context rather than dropping the question outright — still "what the
pipeline would show", just the next rank down when the very first choice
is a dead link. Every unfetchable candidate is a live network check
(`PhotoStore.fetch_photo`, cached), not a guess from the T6.1 manifest
alone, since a URL can also die *between* T6.1's run and this one.

**The control is drawn from outside both the question's gold relevance
and its retrieved context**, so it cannot accidentally be genuine
evidence, and is resampled (seeded, so reproducibly) until a fetchable
one is found or the attempt budget is exhausted — again a live check via
`PhotoStore`, since T6.1 only pre-fetched the *pool* candidates, not the
much larger space controls are drawn from.

Usage:
    python -m cragb.multimodal.photo_link build --config configs/multimodal.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cragb.multimodal.photo_store import PhotoStore
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

# Coverage-funnel stage names, in the fixed order M6.md's T6.3 spec lists
# them ("60 total -> 49 image_target -> N with a photo-bearing retrieved
# doc -> M with fetchable bytes -> K judged") -- a `pandas.Categorical`
# over this exact sequence is what keeps `build_coverage_funnel`'s stage
# counts checkably monotonic rather than accidentally reordered.
FUNNEL_STAGES = (
    "total_questions",
    "image_target",
    "photo_in_context",
    "fetchable_bytes",
    "usable_pairs",
)

DROP_REASON_NO_PHOTO_IN_CONTEXT = "no_photo_in_context"
DROP_REASON_SURFACED_UNFETCHABLE = "surfaced_unfetchable"
DROP_REASON_CONTROL_EXHAUSTED = "control_exhausted"
DROP_REASON_NO_TRANSCRIPT = "no_rag_small_transcript"


# --------------------------------------------------------------------------
# Loading (CRAGB questions + RAG-small transcripts)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CragbQuestion:
    """The slice of a CRAGB question this module needs.

    A narrow, module-local view rather than a shared dataclass, matching
    this project's own convention (`cragb.eval.cragb_questions.RetrievalQuestion`,
    `cragb.bench.best_photo.ImageQuestionRecord`): each consumer of
    `cragb_v1.jsonl` keeps only the fields it actually uses.
    """

    id: str
    type: str
    question: str
    relevant_ids: frozenset[str]
    image_target: bool


def load_cragb_questions(path: str | Path) -> list[CragbQuestion]:
    """Load every question in `path` (default `benchmark/cragb_v1.jsonl`)."""
    questions: list[CragbQuestion] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(
                CragbQuestion(
                    id=obj["id"],
                    type=obj["type"],
                    question=obj["question"],
                    relevant_ids=frozenset(obj["relevant_ids"]),
                    image_target=bool(obj["image_target"]),
                )
            )
    return questions


@dataclass(frozen=True)
class RagContext:
    """The slice of a RAG-small transcript this module needs: what the
    pipeline actually retrieved and which of those docs have a photo."""

    context_doc_ids: tuple[str, ...]
    context_photo_flags: dict[str, bool]


def load_rag_small_context(path: str | Path) -> dict[str, RagContext]:
    """Load `answer_gen_rag_small_v1.jsonl` into a `question_id -> RagContext` map."""
    contexts: dict[str, RagContext] = {}
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            contexts[obj["question_id"]] = RagContext(
                context_doc_ids=tuple(obj["context_doc_ids"]),
                context_photo_flags=dict(obj["context_photo_flags"]),
            )
    return contexts


def _first_image_url(corpus: pd.DataFrame, doc_id: str, image_urls_col: str) -> str | None:
    """The first (highest-preference) image URL for `doc_id`, or None if it has none."""
    row = corpus.loc[int(doc_id)]
    urls = row[image_urls_col]
    if urls is not None and len(urls) > 0:
        return str(urls[0])
    return None


# --------------------------------------------------------------------------
# Surfaced photo
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfacedResult:
    """Outcome of resolving the pipeline's surfaced photo for one question."""

    doc_id: str | None
    photo_id: str | None
    url: str | None
    status: str  # "ok" | "no_photo_in_context" | "unfetchable"


def surfaced_photo(
    context: RagContext,
    corpus: pd.DataFrame,
    store: PhotoStore,
    *,
    image_urls_col: str = "image_urls",
    top_k: int | None = None,
) -> SurfacedResult:
    """The photo the RAG-small pipeline would actually surface for this question.

    Walks `context.context_doc_ids` in retrieval-rank order (the order
    T4a.2's `context_builder` retrieved them, matching `pools_v1.jsonl`'s
    per-retriever rank) and returns the first one flagged
    `context_photo_flags[doc_id] == True` whose photo is actually
    fetchable. A photo-bearing doc whose URL turns out to be dead is
    skipped in favour of the next photo-bearing doc in context, not
    treated as fatal — see the module docstring.

    Args:
        context: this question's RAG-small retrieved context.
        corpus: `corpus_v1.parquet`, indexed by doc_id (int).
        store: a `PhotoStore` used to verify/fetch the candidate photo.
        image_urls_col: corpus column holding each review's URL list.
        top_k: scan only the first `top_k` entries of `context_doc_ids`
            (default `None`: scan the full context, i.e. everything the
            pipeline actually retrieved — capping this would understate
            what the real pipeline shows).

    Returns:
        A `SurfacedResult`. `status="ok"` is the only outcome with a
        usable `photo_id`; the other two are real, reportable outcomes.
    """
    doc_ids = context.context_doc_ids[:top_k] if top_k is not None else context.context_doc_ids
    saw_any_photo_flag = False
    for doc_id in doc_ids:
        if not context.context_photo_flags.get(doc_id):
            continue
        saw_any_photo_flag = True
        url = _first_image_url(corpus, doc_id, image_urls_col)
        if url is None:
            continue
        record = store.fetch_photo(url)
        if record.status == "ok":
            return SurfacedResult(doc_id=doc_id, photo_id=record.photo_id, url=url, status="ok")
        logger.debug("surfaced candidate doc_id=%s unfetchable (status=%s)", doc_id, record.status)

    status = "unfetchable" if saw_any_photo_flag else "no_photo_in_context"
    return SurfacedResult(doc_id=None, photo_id=None, url=None, status=status)


# --------------------------------------------------------------------------
# Control photo
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlResult:
    """Outcome of resolving a random control photo for one question."""

    doc_id: str | None
    photo_id: str | None
    url: str | None
    status: str  # "ok" | "no_eligible_candidates" | "exhausted"


def control_photo(
    exclude_doc_ids: frozenset[str],
    corpus: pd.DataFrame,
    store: PhotoStore,
    rng: np.random.Generator,
    *,
    image_col: str = "has_image",
    image_urls_col: str = "image_urls",
    max_attempts: int = 200,
) -> ControlResult:
    """A random, verifiably-fetchable photo that is not evidence for this question.

    Args:
        exclude_doc_ids: doc ids the control must not be drawn from --
            the question's `relevant_ids` plus every doc in its retrieved
            context, so the control cannot accidentally be genuine
            evidence.
        corpus: `corpus_v1.parquet`, indexed by doc_id (int).
        store: a `PhotoStore` used to verify/fetch each trial candidate.
        image_col: corpus boolean column flagging an image-bearing review.
        image_urls_col: corpus column holding each review's URL list.
        rng: seeded `numpy.random.Generator` -- the same seed reproduces
            byte-identical control draws across runs, since sampling order
            depends only on it and on `corpus`'s (fixed) row order.
        max_attempts: number of distinct eligible candidates to try before
            giving up.

    Returns:
        A `ControlResult`. `status="ok"` is the only outcome with a usable
        `photo_id`.
    """
    is_image_bearing = corpus[image_col].astype(bool)
    is_excluded = corpus.index.astype(str).isin(exclude_doc_ids)
    eligible = corpus.index[is_image_bearing & ~is_excluded].to_numpy()

    if len(eligible) == 0:
        return ControlResult(None, None, None, status="no_eligible_candidates")

    n_try = min(max_attempts, len(eligible))
    trial_positions = rng.choice(len(eligible), size=n_try, replace=False)

    for pos in trial_positions:
        doc_id = str(eligible[pos])
        url = _first_image_url(corpus, doc_id, image_urls_col)
        if url is None:
            continue
        record = store.fetch_photo(url)
        if record.status == "ok":
            return ControlResult(doc_id=doc_id, photo_id=record.photo_id, url=url, status="ok")
        logger.debug("control candidate doc_id=%s unfetchable (status=%s)", doc_id, record.status)

    return ControlResult(None, None, None, status="exhausted")


# --------------------------------------------------------------------------
# Pairing + coverage funnel
# --------------------------------------------------------------------------


def build_pairs(
    questions: list[CragbQuestion],
    contexts: dict[str, RagContext],
    corpus: pd.DataFrame,
    store: PhotoStore,
    rng: np.random.Generator,
    *,
    image_col: str = "has_image",
    image_urls_col: str = "image_urls",
    top_k: int | None = None,
    max_control_attempts: int = 200,
) -> pd.DataFrame:
    """Build one row per image-target question: its surfaced/control photo pair, or why not.

    Args:
        questions: every CRAGB question (image-target and not) -- only the
            image-target ones produce a row; the rest inform
            `build_coverage_funnel`'s first two stages by their absence.
        contexts: `question_id -> RagContext`, from
            `load_rag_small_context`.
        corpus: `corpus_v1.parquet`, indexed by doc_id (int).
        store: a `PhotoStore` used to verify/fetch every candidate photo.
        rng: seeded `numpy.random.Generator`, consumed in `questions` order
            (one `control_photo` draw per image-target question) --
            reproducibility depends on this order being stable, which it
            is: `questions` is read from `cragb_v1.jsonl` in file order.
        image_col, image_urls_col: corpus column names.
        top_k: forwarded to `surfaced_photo`.
        max_control_attempts: forwarded to `control_photo`.

    Returns:
        A DataFrame with one row per image-target question:
        `question_id, type, question, surfaced_photo_id, surfaced_doc_id,
        control_photo_id, control_doc_id, drop_reason`. `drop_reason` is
        `None` for a usable pair (both surfaced and control resolved) and
        one of the `DROP_REASON_*` constants otherwise -- every
        image-target question produces exactly one row, usable or not, so
        nothing is silently omitted.
    """
    rows: list[dict[str, Any]] = []
    for q in questions:
        if not q.image_target:
            continue

        context = contexts.get(q.id)
        if context is None:
            rows.append(
                {
                    "question_id": q.id,
                    "type": q.type,
                    "question": q.question,
                    "surfaced_photo_id": None,
                    "surfaced_doc_id": None,
                    "control_photo_id": None,
                    "control_doc_id": None,
                    "drop_reason": DROP_REASON_NO_TRANSCRIPT,
                }
            )
            continue

        surfaced = surfaced_photo(context, corpus, store, image_urls_col=image_urls_col, top_k=top_k)
        if surfaced.status != "ok":
            drop_reason = (
                DROP_REASON_NO_PHOTO_IN_CONTEXT
                if surfaced.status == "no_photo_in_context"
                else DROP_REASON_SURFACED_UNFETCHABLE
            )
            rows.append(
                {
                    "question_id": q.id,
                    "type": q.type,
                    "question": q.question,
                    "surfaced_photo_id": None,
                    "surfaced_doc_id": None,
                    "control_photo_id": None,
                    "control_doc_id": None,
                    "drop_reason": drop_reason,
                }
            )
            continue

        exclude = q.relevant_ids | set(context.context_doc_ids)
        control = control_photo(
            frozenset(exclude),
            corpus,
            store,
            rng,
            image_col=image_col,
            image_urls_col=image_urls_col,
            max_attempts=max_control_attempts,
        )
        if control.status != "ok":
            rows.append(
                {
                    "question_id": q.id,
                    "type": q.type,
                    "question": q.question,
                    "surfaced_photo_id": surfaced.photo_id,
                    "surfaced_doc_id": surfaced.doc_id,
                    "control_photo_id": None,
                    "control_doc_id": None,
                    "drop_reason": DROP_REASON_CONTROL_EXHAUSTED,
                }
            )
            continue

        rows.append(
            {
                "question_id": q.id,
                "type": q.type,
                "question": q.question,
                "surfaced_photo_id": surfaced.photo_id,
                "surfaced_doc_id": surfaced.doc_id,
                "control_photo_id": control.photo_id,
                "control_doc_id": control.doc_id,
                "drop_reason": None,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "question_id",
            "type",
            "question",
            "surfaced_photo_id",
            "surfaced_doc_id",
            "control_photo_id",
            "control_doc_id",
            "drop_reason",
        ],
    )


def build_coverage_funnel(all_questions: list[CragbQuestion], pairs: pd.DataFrame) -> pd.DataFrame:
    """The five-stage coverage funnel M6.md's T6.3 spec requires.

    Args:
        all_questions: every CRAGB question (not just image-target ones) --
            stage 1's denominator.
        pairs: `build_pairs`'s output.

    Returns:
        A DataFrame with columns `stage, count, description`, in
        `FUNNEL_STAGES` order. Counts are monotonically non-increasing by
        construction (each stage is a subset of the previous one), and the
        final stage (`usable_pairs`) equals the number of rows with
        `drop_reason is None` -- exactly what `write_pairs_jsonl` writes to
        `mm_pairs_v1.jsonl`.
    """
    n_total = len(all_questions)
    n_image_target = len(pairs)
    # Excludes both "no photo in context" and the (rare/never-expected) missing-
    # transcript rows, since without a transcript we don't even know whether a
    # photo was in context.
    n_photo_in_context = (
        int((~pairs["drop_reason"].isin([DROP_REASON_NO_PHOTO_IN_CONTEXT, DROP_REASON_NO_TRANSCRIPT])).sum())
        if n_image_target
        else 0
    )
    n_fetchable_bytes = int((pairs["drop_reason"].isna()).sum())
    n_usable_pairs = n_fetchable_bytes  # T6.3 hands every fetchable-bytes pair to T6.5 unchanged.

    descriptions = {
        "total_questions": "all CRAGB v1 questions",
        "image_target": "questions flagged image_target=true (T2.3)",
        "photo_in_context": "image-target questions whose RAG-small retrieved context "
        "contains >=1 has-image doc, with a usable transcript",
        "fetchable_bytes": "of those, the surfaced photo AND a control photo both "
        "resolved to real, fetchable bytes",
        "usable_pairs": "pairs written to mm_pairs_v1.jsonl (== fetchable_bytes; every "
        "fetchable pair is handed to T6.5's judge)",
    }
    counts = {
        "total_questions": n_total,
        "image_target": n_image_target,
        "photo_in_context": n_photo_in_context,
        "fetchable_bytes": n_fetchable_bytes,
        "usable_pairs": n_usable_pairs,
    }
    return pd.DataFrame(
        {
            "stage": list(FUNNEL_STAGES),
            "count": [counts[s] for s in FUNNEL_STAGES],
            "description": [descriptions[s] for s in FUNNEL_STAGES],
        }
    )


def write_pairs_jsonl(pairs: pd.DataFrame, path: str | Path) -> int:
    """Write only the usable rows (`drop_reason is None`) -- "the pair set
    actually judged" -- to `path`. Returns the number of rows written."""
    usable = pairs[pairs["drop_reason"].isna()].drop(columns=["drop_reason"])
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in usable.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return len(usable)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal.yaml", help="Path to multimodal config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("command", nargs="?", default="build", choices=["build"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    paths = cfg["paths"]
    corpus_cfg = cfg["corpus"]
    store_cfg = cfg["photo_store"]
    pairing_cfg = cfg.get("pairing", {})

    all_questions = load_cragb_questions(paths["questions_in"])
    contexts = load_rag_small_context(paths["rag_small_transcripts_in"])
    corpus = pd.read_parquet(resolve_path(paths["corpus_in"]))
    store = PhotoStore(
        photos_dir=store_cfg["photos_dir"],
        max_bytes=store_cfg["max_bytes"],
        timeout_s=store_cfg["timeout_s"],
        max_retries=store_cfg["max_retries"],
        request_delay_s=store_cfg["request_delay_s"],
        allowed_mime=tuple(store_cfg["allowed_mime"]),
    )
    rng = np.random.default_rng(cfg["seed"])

    pairs = build_pairs(
        all_questions,
        contexts,
        corpus,
        store,
        rng,
        image_col=corpus_cfg["image_col"],
        image_urls_col=corpus_cfg["image_urls_col"],
        top_k=pairing_cfg.get("top_k"),
        max_control_attempts=pairing_cfg.get("max_control_attempts", 200),
    )
    funnel = build_coverage_funnel(all_questions, pairs)

    n_written = write_pairs_jsonl(pairs, paths["pairs_out"])
    funnel_path = resolve_path(paths["coverage_out"])
    funnel_path.parent.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(funnel_path, index=False)

    logger.info("coverage funnel:")
    for _, row in funnel.iterrows():
        logger.info("  %-16s %3d  (%s)", row["stage"], row["count"], row["description"])
    logger.info("wrote %d usable pairs to %s", n_written, paths["pairs_out"])

    drop_counts = pairs.loc[pairs["drop_reason"].notna(), "drop_reason"].value_counts()
    for reason, count in drop_counts.items():
        logger.info("  dropped %d question(s): %s", count, reason)

    return 0


if __name__ == "__main__":
    sys.exit(main())
