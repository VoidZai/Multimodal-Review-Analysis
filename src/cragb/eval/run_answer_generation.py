"""Full-benchmark answer generation batch driver (T4b.2; PLAN.md §3 E5, M4b.md T4b.2).

M4a's grounded-QA pipeline (`cragb.eval.run_grounded_qa_pilot`) only ever ran a curated
~10-12 question slice of CRAGB v1 — enough for worked-transcript appendix material, not
enough to compute RQ0/RQ1 answer-quality tables over. T4b.1 built the closed-book baseline
arm but has never run it at all. This module is the thin, shared driver that runs **every**
CRAGB v1 question through **each of the three arms** RQ0/RQ1 need:

- ``closed_book`` — T4b.1's no-retrieval baseline (`cragb.generate.closed_book_qa`).
- ``rag_small``  — M4a's grounded pipeline (`cragb.generate.grounded_qa`) at
  `configs/grounded_qa.yaml`'s model, `openai/gpt-oss-20b`.
- ``rag_large``  — the same grounded pipeline at `configs/grounded_qa_large.yaml`'s model,
  `openai/gpt-oss-120b` (PLAN.md §14.4: same family, confirmed live on Groq at T4b.2 build
  time) — RQ1's "larger model, same family, same API, same retriever/k/prompt" arm.

Nothing here reimplements generation, parsing, or prompt rendering — this module only
sequences existing pipeline calls (`run_closed_book_qa`, `run_grounded_qa`) across three
configs and writes each arm's output to the path M4b.md's task spec names, then checks the
result is actually complete before declaring success.

**Why output paths are decided here, not read from each config's own
`paths.transcripts_out`:** `configs/grounded_qa.yaml`'s `transcripts_out` still points at
M4a's curated pilot file (`results/tables/grounded_qa_transcripts_v1.jsonl`), which this
full 60-question run must never overwrite — T4a.6's appendix and notebook depend on that
file staying exactly the curated 10-12 question slice it is. This driver always writes to
`results/tables/answer_gen_<arm>_v1.jsonl` (overridable via `--out` for single-arm runs)
regardless of what a given config's own `paths.transcripts_out` says.

Usage:
    python -m cragb.eval.run_answer_generation --arm all
    python -m cragb.eval.run_answer_generation --arm rag_large --question-ids fit_sizing_000
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cragb.eval.cragb_questions import RetrievalQuestion, load_retrieval_questions
from cragb.generate.api_clients import GroqClient
from cragb.generate.closed_book_qa import ClosedBookTranscript, run_closed_book_qa
from cragb.generate.closed_book_qa import write_transcripts_jsonl as write_closed_book_transcripts_jsonl
from cragb.generate.context_builder import CorpusLookup, build_corpus_lookup, index_bm25_retriever
from cragb.generate.grounded_qa import GroundedQATranscript, load_prompt_template, run_grounded_qa
from cragb.generate.grounded_qa import write_transcripts_jsonl as write_grounded_qa_transcripts_jsonl
from cragb.retrieval.base import Retriever
from cragb.retrieval.chunking import load_chunking_config
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

ARMS: tuple[str, ...] = ("closed_book", "rag_small", "rag_large")

# Each arm's default config and output path. Together these *are* M4b.md's T4b.2 artifact
# list: three configs, three output files.
_ARM_DEFAULT_CONFIG: dict[str, str] = {
    "closed_book": "configs/closed_book_qa.yaml",
    "rag_small": "configs/grounded_qa.yaml",
    "rag_large": "configs/grounded_qa_large.yaml",
}
_ARM_DEFAULT_OUT: dict[str, str] = {
    "closed_book": "results/tables/answer_gen_closed_book_v1.jsonl",
    "rag_small": "results/tables/answer_gen_rag_small_v1.jsonl",
    "rag_large": "results/tables/answer_gen_rag_large_v1.jsonl",
}


def validate_full_run(
    transcripts: list[GroundedQATranscript] | list[ClosedBookTranscript],
    expected_question_ids: tuple[str, ...],
) -> None:
    """Fail loudly on the two concrete ways a batch run can go quietly wrong.

    Generalizes `cragb.eval.run_grounded_qa_pilot.validate_pilot_run` from a fixed
    curated slice to an arbitrary (normally: the full 60-question benchmark) question
    set, and to either transcript type — both `GroundedQATranscript` and
    `ClosedBookTranscript` carry the same `question_id`/`answer_text` fields this check
    needs.

    Args:
        transcripts: one arm's generated transcripts, in request order.
        expected_question_ids: the question ids that were actually requested, in order.

    Raises:
        ValueError: if the transcripts' ids don't exactly match
            `expected_question_ids` (same count, same ids, same order — e.g. a crashed
            or partial run), or if any transcript's `answer_text` is empty. An empty
            answer is the `max_tokens`-too-low-for-a-reasoning-model failure mode
            PLAN.md §14 already documents once for `gpt-oss-20b`; nothing guarantees
            `gpt-oss-120b` or the (differently-shaped) closed-book prompt can't hit it
            too, so this run checks for it directly rather than assuming it away.
    """
    got_ids = tuple(t.question_id for t in transcripts)
    if got_ids != tuple(expected_question_ids):
        missing = sorted(set(expected_question_ids) - set(got_ids))
        extra = sorted(set(got_ids) - set(expected_question_ids))
        raise ValueError(
            f"Transcript ids do not match what was requested "
            f"(got {len(got_ids)}, expected {len(expected_question_ids)}); "
            f"missing={missing} extra={extra}"
        )
    empty = [t.question_id for t in transcripts if not t.answer_text.strip()]
    if empty:
        raise ValueError(
            f"{len(empty)} transcript(s) have an empty answer_text: {empty}. Likely a "
            "max_tokens cap too low for the model to finish its visible answer -- see "
            "the provider config's max_tokens comment."
        )


@dataclass(frozen=True)
class _RagIndex:
    """A BM25 index + lookups built once per corpus, shared across RAG arms that agree on it."""

    corpus: pd.DataFrame
    retriever: Retriever
    chunk_to_parent: dict[str, str]
    lookup: CorpusLookup


def _build_client(cfg: dict) -> GroqClient:
    provider_cfg = cfg["provider"]
    return GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )


def _build_rag_index(cfg: dict) -> _RagIndex:
    corpus = pd.read_parquet(resolve_path(cfg["paths"]["corpus_in"]))
    chunking_config = load_chunking_config(cfg["retrieval"]["chunking_config"])
    retriever, chunk_to_parent = index_bm25_retriever(corpus, chunking_config)
    lookup = build_corpus_lookup(corpus)
    return _RagIndex(corpus=corpus, retriever=retriever, chunk_to_parent=chunk_to_parent, lookup=lookup)


def run_arm(
    arm: str,
    config_path: str,
    out_path: str,
    questions: list[RetrievalQuestion],
    rag_index_cache: dict[str, _RagIndex],
) -> Path:
    """Generate and write the full set of `questions`' transcripts for one arm.

    Args:
        arm: one of `ARMS`.
        config_path: the arm's config YAML (provider + paths, and for RAG arms a
            `retrieval` block).
        out_path: where to write the transcripts JSONL — always used as-is, never
            `cfg["paths"]["transcripts_out"]` (see module docstring).
        questions: which CRAGB questions to answer, in order.
        rag_index_cache: keyed by the resolved `paths.corpus_in` a RAG arm's config
            points at, so `rag_small` and `rag_large` — which share the same corpus and
            chunking scheme by construction (`configs/grounded_qa_large.yaml` is a copy
            of `configs/grounded_qa.yaml` with only the model changed) — build the BM25
            index once per `--arm all` invocation, not twice. Mutated in place; pass the
            same dict across a multi-arm run to get the sharing.

    Returns:
        The resolved path written.

    Raises:
        ValueError: if `arm` is not one of `ARMS`, or if `validate_full_run` rejects
            the generated transcripts.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    cfg = load_config(config_path)
    set_global_seed(cfg["seed"])
    client = _build_client(cfg)
    expected_ids = tuple(q.id for q in questions)

    if arm == "closed_book":
        template = load_prompt_template(cfg["paths"]["prompt_template"])
        transcripts = run_closed_book_qa(questions, template, client.complete)
        validate_full_run(transcripts, expected_ids)
        return write_closed_book_transcripts_jsonl(transcripts, out_path)

    if arm in ("rag_small", "rag_large"):
        cache_key = str(resolve_path(cfg["paths"]["corpus_in"]))
        if cache_key not in rag_index_cache:
            rag_index_cache[cache_key] = _build_rag_index(cfg)
        index = rag_index_cache[cache_key]
        template = load_prompt_template(cfg["paths"]["prompt_template"])
        transcripts = run_grounded_qa(
            questions,
            index.retriever,
            index.chunk_to_parent,
            index.lookup,
            template,
            client.complete,
            k=cfg["retrieval"]["k"],
        )
        validate_full_run(transcripts, expected_ids)
        return write_grounded_qa_transcripts_jsonl(transcripts, out_path)

    # Unreachable: `arm` was already checked against `ARMS` above, and every member of
    # `ARMS` is handled by one of the two branches. Kept only as a defensive last resort.
    raise AssertionError(f"arm {arm!r} passed the ARMS check but matched no branch")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS, "all"), required=True, help="Which arm(s) to run.")
    parser.add_argument(
        "--config", default=None, help="Override the arm's default config path (single-arm runs only)."
    )
    parser.add_argument("--out", default=None, help="Override the arm's default output path (single-arm runs only).")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Subset of CRAGB question ids to run. Default: every question in "
        "--questions-in (the full benchmark this task exists to cover).",
    )
    parser.add_argument(
        "--questions-in", default="benchmark/cragb_v1.jsonl", help="Path to the CRAGB v1 questions file."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    if args.arm == "all" and (args.config or args.out):
        raise ValueError("--config/--out only apply to a single --arm run, not --arm all.")

    all_questions = load_retrieval_questions(args.questions_in)
    if args.question_ids:
        by_id = {q.id: q for q in all_questions}
        missing = [qid for qid in args.question_ids if qid not in by_id]
        if missing:
            raise ValueError(f"Unknown question id(s): {missing}")
        questions = [by_id[qid] for qid in args.question_ids]
    else:
        questions = all_questions

    arms_to_run = ARMS if args.arm == "all" else (args.arm,)
    rag_index_cache: dict[str, _RagIndex] = {}
    written: dict[str, Path] = {}

    for arm in arms_to_run:
        config_path = args.config or _ARM_DEFAULT_CONFIG[arm]
        out_path = args.out or _ARM_DEFAULT_OUT[arm]
        logger.info("arm=%s: %d question(s), config=%s -> %s", arm, len(questions), config_path, out_path)
        written[arm] = run_arm(arm, config_path, out_path, questions, rag_index_cache)
        logger.info("arm=%s: wrote %d transcript(s) to %s", arm, len(questions), written[arm])

    logger.info("Done. %d arm(s) run, %d question(s) each.", len(written), len(questions))
    for arm, path in written.items():
        logger.info("  %s -> %s", arm, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
