"""Rubric answer-quality judge: prompt + JSON scorer (T4b.4; PLAN.md §3 E5, §8, §9,
M4b.md T4b.4).

Renders T4b.4's prompt (`prompts/answer_judge_v1.md`) around a question, the context a
candidate answer was written against (or none, for the closed-book arm), the candidate
itself, and the CRAGB reference answer; calls the judge model; and parses its completion
into a structured `JudgeScore` — four 1-5 rubric criteria plus a short rationale.

**The judge is never told which system produced the candidate.** PLAN.md §9 names this
directly as a self-preference-bias risk ("letting the judge see which system produced an
answer"), and PLAN.md §14.4 sharpens why it matters concretely here: two of the three
arms this judge will score (T4b.2's closed-book and RAG-small) share a model family with
one RQ0/RQ1 comparison each, so if the prompt leaked which arm a candidate came from, the
judge's own training could bias toward or against a family it recognizes. Concretely:
`build_judge_prompt` never receives or renders an arm name, model name, or system
identifier — a caller with no context to show passes `context_text=None`, and this module
substitutes a neutral `NO_CONTEXT_MARKER` string that says nothing about *why* there is no
context (never the literal words "closed-book" or "RAG").

**Model:** `configs/judge.yaml` pins this to `qwen/qwen3.6-27b` — a distinct family from
`openai/gpt-oss-20b`/`-120b` (T4b.2's closed-book/RAG-small/RAG-large models), for the
same self-preference reason. That model needs `reasoning_effort: "none"` (also in the
config) to suppress visible `<think>...</think>` reasoning it otherwise emits inline in
its completion (PLAN.md §14.4) — without it, `parse_judge_response` would have to parse
through that text to find the JSON object.

Testability mirrors every other generation module in this project
(`cragb.generate.grounded_qa`, `cragb.generate.closed_book_qa`): every function that
would otherwise need a live API call takes an injected `chat_fn` standing in for
`GroqClient.complete`, so prompt rendering and response parsing are fully unit-testable
with no network access or API key. Only `main()` constructs a real `GroqClient`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from string import Template
from typing import Callable

from dotenv import load_dotenv

from cragb.generate.api_clients import GroqClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import load_config

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]

# Rendered in the judge prompt's "Context available..." section whenever a candidate had
# no retrieved context at all. Deliberately says only *that* there was no context, never
# *why* (i.e. never "closed-book") -- see module docstring on why that distinction is the
# entire point of this constant existing.
NO_CONTEXT_MARKER = "No review context was available when this answer was written."

_CRITERIA: tuple[str, ...] = ("correctness", "faithfulness", "completeness", "conciseness")

# Same code-fence/JSON tolerance cragb.generate.draft_questions.parse_llm_questions
# already established for LLM responses that add a markdown fence despite being told not
# to -- this prompt asks for a JSON *object*, not an array, so the object-matching regex
# differs, but the reasoning (and the ValueError-with-raw-response-attached behavior on
# genuine failure) is identical.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_LANG_RE = re.compile(r"^json\s*", re.IGNORECASE)


def build_judge_prompt(
    question: str,
    context_text: str | None,
    candidate_answer: str,
    reference_answer: str,
    template: Template,
) -> str:
    """Render T4b.4's judge prompt for one (question, candidate, reference) triple.

    Args:
        question: the CRAGB question text.
        context_text: the context block shown to whichever generator produced
            `candidate_answer` (e.g. `ContextBlock.text`), or `None` if no context was
            available at all -- rendered as `NO_CONTEXT_MARKER` in that case, never as
            a literal arm or system name (see module docstring).
        candidate_answer: the answer being scored.
        reference_answer: the CRAGB reference answer (ground truth).
        template: T4b.4's loaded prompt template (see `cragb.generate.grounded_qa
            .load_prompt_template`, reused here rather than reimplemented again).

    Returns:
        The rendered prompt text.
    """
    return template.substitute(
        question=question,
        context_block=context_text if context_text is not None else NO_CONTEXT_MARKER,
        candidate_answer=candidate_answer,
        reference_answer=reference_answer,
    )


@dataclass(frozen=True)
class JudgeScore:
    """One judge call's rubric scores for a single candidate answer.

    Every criterion is an integer in `[1, 5]` (enforced by `parse_judge_response`, the
    only place a `JudgeScore` is ever constructed from model output).
    """

    correctness: int
    faithfulness: int
    completeness: int
    conciseness: int
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_judge_response(raw_response: str) -> JudgeScore:
    """Parse the judge model's raw completion into a `JudgeScore`.

    Tolerant of an accidental markdown code fence around the JSON object (models
    sometimes add one despite being told not to -- same allowance
    `cragb.generate.draft_questions.parse_llm_questions` makes for the same reason).
    Anything else that fails to parse, or parses but doesn't carry a valid score, raises
    with the raw response attached, so a bad judge call is visible and re-runnable
    rather than silently producing a meaningless score that would corrupt T4b.5's batch
    run or T4b.6's reliability measurement.

    Args:
        raw_response: the judge model's raw completion text.

    Returns:
        A `JudgeScore`.

    Raises:
        ValueError: if no JSON object can be extracted from `raw_response`; if the
            parsed value isn't a JSON object; if any of the four rubric keys is
            missing, not an integer, a bool (JSON `true`/`false` would otherwise pass
            Python's `isinstance(x, int)` check, since `bool` subclasses `int` -- that
            is never a valid rubric score), or outside `[1, 5]`; or if `rationale` is
            missing, not a string, or empty.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = _CODE_FENCE_LANG_RE.sub("", text)

    match = _JSON_OBJECT_RE.search(text)
    candidate_text = match.group(0) if match else text

    try:
        obj = json.loads(candidate_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse a JSON object from the judge's response: {e}\n"
            f"--- raw response ---\n{raw_response}"
        ) from e

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object from the judge, got {type(obj).__name__}: {obj!r}")

    missing = [key for key in (*_CRITERIA, "rationale") if key not in obj]
    if missing:
        raise ValueError(f"Judge response is missing key(s) {missing}: {obj!r}")

    scores: dict[str, int] = {}
    for key in _CRITERIA:
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 5):
            raise ValueError(f"Judge score {key!r}={value!r} is not an integer in [1, 5]: {obj!r}")
        scores[key] = value

    rationale = obj["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"Judge 'rationale' must be a non-empty string, got {rationale!r}: {obj!r}")

    return JudgeScore(
        correctness=scores["correctness"],
        faithfulness=scores["faithfulness"],
        completeness=scores["completeness"],
        conciseness=scores["conciseness"],
        rationale=rationale.strip(),
    )


def score_answer(
    question: str,
    context_text: str | None,
    candidate_answer: str,
    reference_answer: str,
    template: Template,
    chat_fn: ChatFn,
) -> JudgeScore:
    """Render the judge prompt, call `chat_fn`, and parse the result into a `JudgeScore`."""
    prompt = build_judge_prompt(question, context_text, candidate_answer, reference_answer, template)
    raw_response = chat_fn([{"role": "user", "content": prompt}])
    return parse_judge_response(raw_response)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/judge.yaml", help="Path to judge config YAML.")
    parser.add_argument("--question", required=True, help="The CRAGB question text.")
    parser.add_argument("--candidate-answer", required=True, help="The answer to score.")
    parser.add_argument("--reference-answer", required=True, help="The trusted reference answer.")
    parser.add_argument(
        "--context",
        default=None,
        help="Context text the candidate was written against; omit for a closed-book (no-context) candidate.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: score one hand-specified (question, candidate, reference) triple.

    A deliberately small hook for T4b.4 -- confirming the pipeline works end-to-end
    against the real judge model. Batch-scoring every transcript from every arm is
    T4b.5's job, not this CLI's.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    cfg = load_config(args.config)
    template = load_prompt_template(cfg["paths"]["prompt_template"])

    provider_cfg = cfg["provider"]
    client = GroqClient(
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

    score = score_answer(
        args.question, args.context, args.candidate_answer, args.reference_answer, template, client.complete
    )

    logger.info("Judge score: %s", score.to_dict())
    print(json.dumps(score.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
