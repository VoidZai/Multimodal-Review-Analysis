"""Batch judge run over all three answer-generation arms (T4b.5; PLAN.md §3 E5,
M4b.md T4b.5).

T4b.4 built the rubric judge (prompt + JSON scorer); T4b.2 generated 60 questions'
answers for each of three arms (closed-book, RAG-small, RAG-large). This module is the
thin driver connecting them: load every arm's transcripts, load CRAGB's reference
answers, score every (arm, question) pair with the judge, and write the combined
per-question score table T4b.7's RQ0/RQ1 aggregation reads.

**The judge is shown a context block only for the two RAG arms, and never told why the
closed-book arm has none.** `context_text_for` returns `None` for `"closed_book"` and
`transcript.context.text` for the RAG arms — `cragb.eval.judge.score_answer` renders
`None` as `judge.NO_CONTEXT_MARKER`, a neutral string that says nothing about which arm
produced the candidate (PLAN.md §9's self-preference-bias caution, already the reason
`configs/judge.yaml` pins a model outside the `gpt-oss` family the RAG/closed-book arms
themselves run on).

**Transcript paths are read from `cragb.eval.run_answer_generation.ARM_DEFAULT_OUT`**,
not re-hardcoded here — T4b.2's driver is the single source of truth for where each
arm's generated answers live; duplicating those three paths a second time here would
only be a second place for them to silently drift apart.

Usage:
    python -m cragb.eval.run_judge_eval
    python -m cragb.eval.run_judge_eval --arm closed_book rag_small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cragb.bench.reference_answers import ReferenceAnswer, load_reference_answers
from cragb.eval.judge import score_answer
from cragb.eval.run_answer_generation import ARM_DEFAULT_OUT, ARMS
from cragb.eval.run_grounded_qa_pilot import write_csv
from cragb.generate.api_clients import GroqClient
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.closed_book_qa import load_transcripts_jsonl as load_closed_book_transcripts
from cragb.generate.grounded_qa import GroundedQATranscript, load_prompt_template
from cragb.generate.grounded_qa import load_transcripts_jsonl as load_grounded_qa_transcripts
from cragb.utils.io import load_config
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_CONFIG = "configs/judge.yaml"
DEFAULT_REFERENCES_IN = "benchmark/reference_answers_v1.jsonl"
DEFAULT_SCORES_OUT = "results/tables/judge_scores_v1.csv"

_RUBRIC_COLUMNS: tuple[str, ...] = ("correctness", "faithfulness", "completeness", "conciseness")
_SCORE_COLUMNS: tuple[str, ...] = ("arm", "question_id", *_RUBRIC_COLUMNS, "rationale")


def load_arm_transcripts(
    arm: str, path: str | Path
) -> list[GroundedQATranscript] | list[ClosedBookTranscript]:
    """Load `arm`'s transcripts from `path`, using that arm's own transcript type."""
    if arm == "closed_book":
        return load_closed_book_transcripts(path)
    if arm in ("rag_small", "rag_large"):
        return load_grounded_qa_transcripts(path)
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")


def context_text_for(arm: str, transcript: GroundedQATranscript | ClosedBookTranscript) -> str | None:
    """The context text to show the judge for one transcript, or `None` for no context.

    `arm` decides *whether* a context block exists at all -- it is never itself shown to
    the judge (see module docstring).
    """
    if arm == "closed_book":
        return None
    return transcript.context.text  # type: ignore[union-attr]  # rag arms only


def score_transcripts(
    arm: str,
    transcripts: list[GroundedQATranscript] | list[ClosedBookTranscript],
    references: dict[str, ReferenceAnswer],
    template,
    chat_fn,
) -> pd.DataFrame:
    """Score one arm's every transcript against its CRAGB reference answer.

    Args:
        arm: one of `ARMS` -- decides whether a context block is shown to the judge
            (`context_text_for`); never passed to the judge itself.
        transcripts: that arm's loaded transcripts (`load_arm_transcripts`).
        references: from `cragb.bench.reference_answers.load_reference_answers`, keyed
            by `question_id`.
        template: T4b.4's loaded judge prompt template.
        chat_fn: `GroqClient.complete`, or a stand-in for testing.

    Returns:
        One row per transcript: `arm`, `question_id`, `correctness`, `faithfulness`,
        `completeness`, `conciseness`, `rationale`.

    Raises:
        KeyError: if a transcript's `question_id` has no reference answer -- scoring
            against a missing reference would silently produce a meaningless
            comparison, mirroring `cragb.eval.metrics_answer.score_arm`'s convention
            for the same class of "missing ground truth" gap.
    """
    missing = [t.question_id for t in transcripts if t.question_id not in references]
    if missing:
        raise KeyError(f"No reference answer for question_id(s): {missing}")

    rows = []
    for t in transcripts:
        context_text = context_text_for(arm, t)
        reference_text = references[t.question_id].answer
        score = score_answer(t.question, context_text, t.answer_text, reference_text, template, chat_fn)
        row = score.to_dict()
        row["arm"] = arm
        row["question_id"] = t.question_id
        rows.append(row)

    return pd.DataFrame(rows, columns=list(_SCORE_COLUMNS))


def validate_judge_scores(
    scores: pd.DataFrame, expected_arms: tuple[str, ...], expected_question_ids: tuple[str, ...]
) -> None:
    """Fail loudly if the batch judge run is incomplete or any score is out of range.

    Args:
        scores: the combined per-arm score table (`pd.concat` of `score_transcripts`
            outputs).
        expected_arms: every arm that should appear, e.g. `ARMS`.
        expected_question_ids: every question id that should appear per arm.

    Raises:
        ValueError: if the row count doesn't match `len(expected_arms) *
            len(expected_question_ids)`; if any `(arm, question_id)` pair is missing,
            duplicated, or unexpected; if any rubric score is null or outside `[1, 5]`;
            or if any `rationale` is null or empty. Mirrors
            `cragb.eval.run_answer_generation.validate_full_run`'s "fail before write,
            not after" discipline, generalized from one arm's transcripts to the full
            arm x question grid this batch run must cover.
    """
    expected_n = len(expected_arms) * len(expected_question_ids)
    if len(scores) != expected_n:
        raise ValueError(
            f"Expected {expected_n} judge score row(s) ({len(expected_arms)} arm(s) x "
            f"{len(expected_question_ids)} question(s)), got {len(scores)}"
        )

    expected_pairs = {(arm, qid) for arm in expected_arms for qid in expected_question_ids}
    got_pairs = list(zip(scores["arm"], scores["question_id"]))
    got_pairs_set = set(got_pairs)
    if len(got_pairs) != len(got_pairs_set):
        raise ValueError("Duplicate (arm, question_id) rows in judge scores.")

    missing = sorted(expected_pairs - got_pairs_set)
    extra = sorted(got_pairs_set - expected_pairs)
    if missing or extra:
        raise ValueError(f"Judge scores do not match the expected arm x question grid; missing={missing} extra={extra}")

    if scores[list(_RUBRIC_COLUMNS)].isnull().any().any():
        raise ValueError("Judge scores contain null rubric value(s).")
    for col in _RUBRIC_COLUMNS:
        out_of_range = scores.loc[(scores[col] < 1) | (scores[col] > 5), ["arm", "question_id", col]]
        if not out_of_range.empty:
            raise ValueError(f"Judge scores out of [1, 5] range in {col!r}: {out_of_range.to_dict('records')}")

    empty_rationale = scores["rationale"].isnull() | (scores["rationale"].astype(str).str.strip() == "")
    if empty_rationale.any():
        raise ValueError(
            f"Judge scores contain empty/null rationale(s) for: "
            f"{scores.loc[empty_rationale, ['arm', 'question_id']].to_dict('records')}"
        )


def _build_judge_client(cfg: dict) -> GroqClient:
    provider_cfg = cfg["provider"]
    return GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        reasoning_effort=provider_cfg.get("reasoning_effort"),
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-config", default=DEFAULT_JUDGE_CONFIG, help="Path to judge config YAML.")
    parser.add_argument(
        "--references-in", default=DEFAULT_REFERENCES_IN, help="Path to CRAGB's reference-answers JSONL."
    )
    parser.add_argument("--out", default=DEFAULT_SCORES_OUT, help="Where to write the combined judge scores CSV.")
    parser.add_argument(
        "--arm",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="Which arm(s) to score (default: all three).",
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

    judge_cfg = load_config(args.judge_config)
    set_global_seed(judge_cfg["seed"])
    template = load_prompt_template(judge_cfg["paths"]["prompt_template"])
    client = _build_judge_client(judge_cfg)

    references = load_reference_answers(args.references_in)
    expected_question_ids = tuple(references)

    per_arm_scores = []
    for arm in args.arm:
        transcripts_path = ARM_DEFAULT_OUT[arm]
        transcripts = load_arm_transcripts(arm, transcripts_path)
        logger.info("arm=%s: scoring %d transcript(s) from %s", arm, len(transcripts), transcripts_path)
        arm_scores = score_transcripts(arm, transcripts, references, template, client.complete)
        per_arm_scores.append(arm_scores)
        logger.info("arm=%s: scored %d transcript(s)", arm, len(arm_scores))

    result = pd.concat(per_arm_scores, ignore_index=True)
    validate_judge_scores(result, expected_arms=tuple(args.arm), expected_question_ids=expected_question_ids)

    out_path = write_csv(result, args.out)
    logger.info("Wrote %d judge score row(s) to %s", len(result), out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
