"""Pairwise vision-evidence judge: prompt + JSON parser + order-swap bias
control (T6.4; PLAN.md §3 E7, §8 G3, §9, M6.md T6.4).

Renders T6.4's prompt (`prompts/vision_judge_v1.md`) around a CRAGB question and two
candidate photos (the RAG-small pipeline's surfaced photo and T6.3's random control),
calls the vision judge model, and parses its completion into a structured
`VisionVerdict`. `judge_pair` is the module's real deliverable: it runs the judge
**twice** per pair — once with the surfaced photo in position A, once with positions
swapped — so a judge with a positional preference (e.g. "always prefer A") cannot
manufacture a surfaced-photo win. This mirrors PLAN.md §9's self-preference-bias caution
for the text judge (T4b.4), applied to the one bias that matters for a *pairwise*
design: which position, not which system, holds the model's favor.

**No self-preference risk from model family here** — unlike T4b.4's judge (which had to
be picked from a different family than the arms it scores, PLAN.md §14.4), this judge
(Gemini, `cragb.generate.gemini_client`) is a different provider entirely from every
model this project has generated with (Groq). The bias this module guards against is
purely positional, not familial.

**Parsing mirrors `cragb.eval.judge.parse_judge_response`** (same JSON-object
extraction, same markdown-code-fence tolerance) with one addition M6.md's T6.4 spec
calls for explicitly: a leading `<think>...</think>` block is stripped before parsing.
T4b.4's judge (PLAN.md §14.4) needed this because `qwen/qwen3.6-27b` emits visible
inline reasoning by default; Gemini's `gemini-3.6-flash` (confirmed live, T6.2) hides its
"thinking" in `usageMetadata.thoughtsTokenCount` instead, so this path is not currently
exercised by the live model in use — it's kept as defensive parity in case a future
model (or a Gemini config change) starts emitting one, exactly the situation T4b.4 hit
with no warning.

Testability mirrors `cragb.eval.judge.score_answer`: every function that would otherwise
need a live API call takes an injected `chat_fn` standing in for
`cragb.generate.gemini_client.GeminiClient.complete`, so prompt rendering, response
parsing, and the order-swap logic are fully unit-testable with no network access or API
key.

Usage:
    python -m cragb.multimodal.vision_judge \\
        --config configs/vision_judge.yaml \\
        --question "Is the colour as pictured?" \\
        --surfaced-photo-id <photo_id> --control-photo-id <photo_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from string import Template
from typing import Any, Callable

from dotenv import load_dotenv

from cragb.generate.gemini_client import GeminiClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.multimodal.photo_store import PhotoStore
from cragb.utils.io import load_config

logger = logging.getLogger(__name__)

VisionChatFn = Callable[[list[dict[str, Any]]], str]

# The template's [[PHOTO_A]]/[[PHOTO_B]] markers are where the two provider-neutral
# image parts get spliced into the flat parts list `build_vision_prompt` returns --
# they are never sent to any model themselves.
_PHOTO_A_MARKER = "[[PHOTO_A]]"
_PHOTO_B_MARKER = "[[PHOTO_B]]"

_VALID_WINNERS = ("A", "B", "tie")

# Same code-fence/JSON tolerance cragb.eval.judge.parse_judge_response established for
# LLM responses that add a markdown fence despite being told not to. `_THINK_BLOCK_RE`
# is this module's addition -- see module docstring for why it's kept as defensive
# parity even though the live model in use doesn't currently need it.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_LANG_RE = re.compile(r"^json\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class VisionJudgeParseError(ValueError):
    """Raised when a vision-judge completion can't be parsed into a `VisionVerdict`."""


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def build_vision_prompt(
    question: str,
    photo_a_part: dict[str, Any],
    photo_b_part: dict[str, Any],
    template: Template,
) -> list[dict[str, Any]]:
    """Render T6.4's vision-judge prompt, interleaved with the two photo parts.

    Args:
        question: the CRAGB question text.
        photo_a_part: provider-neutral image part shown as "Photo A" (e.g.
            `cragb.multimodal.photo_store.PhotoStore.to_data_part`'s output).
        photo_b_part: provider-neutral image part shown as "Photo B".
        template: the loaded vision-judge prompt template (see
            `cragb.generate.grounded_qa.load_prompt_template`, reused here).

    Returns:
        A flat list of provider-neutral parts ready for
        `cragb.generate.gemini_client.GeminiClient.complete`/`complete_with_usage`:
        text, then `photo_a_part`, then text, then `photo_b_part`, then text.
        Empty text segments (e.g. if the markers are adjacent) are omitted.

    Raises:
        ValueError: the rendered template is missing either photo marker.
    """
    rendered = template.substitute(question=question)
    if _PHOTO_A_MARKER not in rendered or _PHOTO_B_MARKER not in rendered:
        raise ValueError(
            f"Vision-judge prompt template must contain both {_PHOTO_A_MARKER!r} and "
            f"{_PHOTO_B_MARKER!r} markers to place the two photos."
        )

    before, rest = rendered.split(_PHOTO_A_MARKER, 1)
    between, after = rest.split(_PHOTO_B_MARKER, 1)

    parts: list[dict[str, Any]] = []
    if before.strip():
        parts.append({"type": "text", "text": before})
    parts.append(photo_a_part)
    if between.strip():
        parts.append({"type": "text", "text": between})
    parts.append(photo_b_part)
    if after.strip():
        parts.append({"type": "text", "text": after})
    return parts


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionVerdict:
    """One vision-judge call's verdict for a single (Photo A, Photo B) pair."""

    winner: str  # "A" | "B" | "tie"
    confidence: int
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_vision_response(raw_response: str) -> VisionVerdict:
    """Parse the vision judge model's raw completion into a `VisionVerdict`.

    Tolerant of a leading `<think>...</think>` reasoning block (stripped before
    parsing) and an accidental markdown code fence around the JSON object --
    see module docstring. Anything else that fails to parse, or parses but
    doesn't carry a valid verdict, raises with the raw response attached, so
    a bad judge call is visible and re-runnable rather than silently
    corrupting T6.5's win-rate.

    Args:
        raw_response: the vision judge model's raw completion text.

    Returns:
        A `VisionVerdict`.

    Raises:
        VisionJudgeParseError: no JSON object can be extracted; the parsed
            value isn't a JSON object; `winner` is missing or not one of
            `"A"`/`"B"`/`"tie"`; `confidence` is missing, not an integer, a
            bool, or outside `[1, 5]`; or `rationale` is missing, not a
            string, or empty.
    """
    text = _THINK_BLOCK_RE.sub("", raw_response).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = _CODE_FENCE_LANG_RE.sub("", text)

    match = _JSON_OBJECT_RE.search(text)
    candidate_text = match.group(0) if match else text

    try:
        obj = json.loads(candidate_text)
    except json.JSONDecodeError as e:
        raise VisionJudgeParseError(
            f"Could not parse a JSON object from the vision judge's response: {e}\n"
            f"--- raw response ---\n{raw_response}"
        ) from e

    if not isinstance(obj, dict):
        raise VisionJudgeParseError(
            f"Expected a JSON object from the vision judge, got {type(obj).__name__}: {obj!r}"
        )

    missing = [key for key in ("winner", "confidence", "rationale") if key not in obj]
    if missing:
        raise VisionJudgeParseError(f"Vision judge response is missing key(s) {missing}: {obj!r}")

    winner = obj["winner"]
    if not isinstance(winner, str) or winner not in _VALID_WINNERS:
        raise VisionJudgeParseError(f"Vision judge 'winner'={winner!r} is not one of {_VALID_WINNERS}: {obj!r}")

    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not (1 <= confidence <= 5):
        raise VisionJudgeParseError(
            f"Vision judge 'confidence'={confidence!r} is not an integer in [1, 5]: {obj!r}"
        )

    rationale = obj["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise VisionJudgeParseError(
            f"Vision judge 'rationale' must be a non-empty string, got {rationale!r}: {obj!r}"
        )

    return VisionVerdict(winner=winner, confidence=confidence, rationale=rationale.strip())


# --------------------------------------------------------------------------
# Order-swap pairwise judging
# --------------------------------------------------------------------------


def _relative_winner(verdict: VisionVerdict, *, surfaced_is_a: bool) -> str:
    """Translate a raw A/B/tie verdict into "surfaced" / "control" / "tie"."""
    if verdict.winner == "tie":
        return "tie"
    a_wins = verdict.winner == "A"
    return "surfaced" if (a_wins == surfaced_is_a) else "control"


@dataclass(frozen=True)
class PairVerdict:
    """Outcome of judging one (surfaced, control) photo pair in both photo orders.

    Attributes:
        verdict_surfaced_as_a: the raw verdict from the call with the
            surfaced photo in position A, control in position B.
        verdict_surfaced_as_b: the raw verdict from the call with positions
            swapped (control in A, surfaced in B).
        order_agreement: whether the two orders agree on the same relative
            winner once the position swap is accounted for -- the
            position-bias diagnostic T6.6 reports on. `False` means the
            judge's answer tracked *which position* it saw, not which
            photo was actually more relevant.
        outcome: `"surfaced_win"` only if surfaced won in both orders,
            `"control_win"` only if control won in both orders, `"tie"`
            otherwise (either call returned an explicit tie, or the two
            orders disagree). A pair can never count as a surfaced-photo
            win on the strength of one order alone.
    """

    verdict_surfaced_as_a: VisionVerdict
    verdict_surfaced_as_b: VisionVerdict
    order_agreement: bool
    outcome: str  # "surfaced_win" | "control_win" | "tie"


def judge_pair(
    question: str,
    surfaced_photo_id: str,
    control_photo_id: str,
    store: PhotoStore,
    template: Template,
    chat_fn: VisionChatFn,
) -> PairVerdict:
    """Judge one surfaced-vs-control photo pair in both photo orders.

    Args:
        question: the CRAGB question text.
        surfaced_photo_id: photo id of the pipeline's surfaced photo (T6.3).
        control_photo_id: photo id of the random control photo (T6.3).
        store: a `PhotoStore` used to resolve each photo id to a
            provider-neutral image part (`to_data_part`) -- both ids must
            already be cached (T6.1/T6.3 fetch on demand, so by this point
            they are).
        template: the loaded vision-judge prompt template.
        chat_fn: stands in for `GeminiClient.complete` -- takes a flat parts
            list, returns the raw completion text. Injected so this
            function needs no network/API key to test.

    Returns:
        A `PairVerdict`.
    """
    surfaced_part = store.to_data_part(surfaced_photo_id)
    control_part = store.to_data_part(control_photo_id)

    prompt_ab = build_vision_prompt(question, surfaced_part, control_part, template)
    verdict_ab = parse_vision_response(chat_fn(prompt_ab))  # A = surfaced, B = control

    prompt_ba = build_vision_prompt(question, control_part, surfaced_part, template)
    verdict_ba = parse_vision_response(chat_fn(prompt_ba))  # A = control, B = surfaced

    rel_ab = _relative_winner(verdict_ab, surfaced_is_a=True)
    rel_ba = _relative_winner(verdict_ba, surfaced_is_a=False)
    order_agreement = rel_ab == rel_ba

    if order_agreement and rel_ab in ("surfaced", "control"):
        outcome = f"{rel_ab}_win"
    else:
        outcome = "tie"

    return PairVerdict(
        verdict_surfaced_as_a=verdict_ab,
        verdict_surfaced_as_b=verdict_ba,
        order_agreement=order_agreement,
        outcome=outcome,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vision_judge.yaml", help="Path to vision-judge config YAML.")
    parser.add_argument("--question", required=True, help="The CRAGB question text.")
    parser.add_argument("--surfaced-photo-id", required=True, help="Photo id of the pipeline's surfaced photo.")
    parser.add_argument("--control-photo-id", required=True, help="Photo id of the random control photo.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: judge one hand-specified pair against the real model.

    A deliberately small hook for T6.4 -- confirming the pipeline works
    end-to-end against the real judge before committing quota to a full
    run. Batch-judging every pair from T6.3's `mm_pairs_v1.jsonl` is T6.5's
    job, not this CLI's.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    cfg = load_config(args.config)
    template = load_prompt_template(cfg["paths"]["prompt_template"])
    store_cfg = cfg["photo_store"]
    store = PhotoStore(
        photos_dir=store_cfg["photos_dir"],
        max_bytes=store_cfg["max_bytes"],
        timeout_s=store_cfg["timeout_s"],
        max_retries=store_cfg["max_retries"],
        request_delay_s=store_cfg["request_delay_s"],
        allowed_mime=tuple(store_cfg["allowed_mime"]),
    )

    provider_cfg = cfg["provider"]
    client = GeminiClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=provider_cfg["cache_dir"],
        call_log_path=provider_cfg["call_log_path"],
    )

    pair_verdict = judge_pair(
        args.question, args.surfaced_photo_id, args.control_photo_id, store, template, client.complete
    )

    logger.info("outcome=%s order_agreement=%s", pair_verdict.outcome, pair_verdict.order_agreement)
    print(
        json.dumps(
            {
                "outcome": pair_verdict.outcome,
                "order_agreement": pair_verdict.order_agreement,
                "verdict_surfaced_as_a": pair_verdict.verdict_surfaced_as_a.to_dict(),
                "verdict_surfaced_as_b": pair_verdict.verdict_surfaced_as_b.to_dict(),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
