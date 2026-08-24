"""Quality filter: deterministic citation check + the one validated judge criterion
(T7.5; PLAN.md §3 E8, §10, M7.md T7.5).

Drops the training examples that would teach the fine-tuned model the wrong thing, using
only instruments whose reliability has actually been measured. Two independent stages,
run in order, per example:

- **Stage 1, deterministic and free.** Re-verifies -- from scratch, not by trusting
  T7.1/T7.3/T7.4's own construction-time guarantees -- that every cited id exists in the
  example's own context, that `cited_doc_ids` still matches what `example.answer`'s
  bracket citations actually say (catching any hypothetical drift between the two), that
  an abstention carries no citations and a non-abstention carries at least one, and that
  the answer is a well-formed single paragraph (no malformed citation brackets, no
  markdown headings/bullets, no stray braces suggesting JSON). This mirrors
  `cragb.bench.assemble`'s own stated principle -- "re-check that nothing drifted between
  independently-run tasks" -- applied to a pipeline stage that has already, in principle,
  filtered these things once before (T7.3's `parse_generated_pairs` drops fabricated
  citations at generation time; T7.4's abstentions are structurally guaranteed). Any
  failure here is a hard drop, no API call spent.
- **Stage 2, judged.** Runs `qwen/qwen3.6-27b` (the same distinct-family judge T4b.4
  uses, `configs/judge.yaml`) on every stage-1 survivor that isn't an abstention, and
  keeps only its `faithfulness` score -- the one rubric criterion
  `results/tables/judge_validation_v1.csv` (T4b.6) measured above the 0.4 usability bar
  (κ=0.597; correctness κ=-0.151, completeness 0.243, conciseness 0.321 are all below
  it). `configs/finetune.yaml`'s `filter:` block already documents, in a comment, that
  the other three criteria are *deliberately* unused here, not an oversight.

  **No independent reference answer exists for synthetic training data**, unlike T4b's
  CRAGB evaluation (which always has a human-written reference). This module passes
  `example.answer` as *both* the candidate and the reference to
  `cragb.eval.judge.build_judge_prompt` -- a deliberate choice, safe specifically because
  the prompt defines `faithfulness` (the only score this module reads) purely in terms of
  "is every claim traceable to the context shown", never in terms of the reference
  (`correctness`/`completeness`, which do compare against the reference, become trivially
  self-referential and are simply discarded). This also sidesteps the abstention
  special-rule in `answer_judge_v1.md` (candidate == reference means it can never fire
  unexpectedly), which is moot anyway since abstentions never reach this stage.

  Bypasses `cragb.eval.judge.score_answer`'s convenience wrapper and calls
  `build_judge_prompt`/`parse_judge_response` directly instead, through
  `GroqClient.complete_with_usage` rather than `.complete` -- `score_answer`'s `chat_fn`
  interface returns only text, with no path to the token/latency telemetry
  `build_filter_funnel`'s judge-cost columns need (PLAN.md §14.5's lesson: log usage from
  the first API-calling line of code). Both of `score_answer`'s constituent calls are
  still reused directly; only its thin orchestration wrapper is bypassed.

- **Abstentions bypass stage 2 entirely** -- there is nothing to be faithful *to* when
  the correct answer is a fixed, context-independent phrase. They still go through stage
  1, plus whatever structural guarantees `TrainingExample.__post_init__` and T7.4's
  construction methods already gave them.

This module makes no attempt at T7.3-style incremental resumability: every call to
`filter_examples` re-processes its full input list from scratch. That's the right choice
here (not an oversight) -- unlike T7.3's many-calls-per-context-group generation sweep,
this stage makes at most one judge call per non-abstention example, and
`GroqClient.complete_with_usage`'s disk cache already makes a full re-run of unchanged
input a 100%-cache-hit, free replay; there is no partial-progress state worth tracking
separately.

Usage:
    python -m cragb.finetune.filter_pairs --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from string import Template

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from cragb.bench.reference_answers import extract_citations
from cragb.eval.citation_validity import find_malformed_citations
from cragb.eval.cost_model import ModelPricing, UsageFn, cost_usd, load_pricing_config
from cragb.eval.judge import build_judge_prompt, parse_judge_response
from cragb.finetune.schema import TrainingExample, load_training_examples_jsonl, write_training_examples_jsonl
from cragb.generate.api_clients import CompletionResult, GroqClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

# Drop-reason funnel buckets. Checked in this order in `run_stage1` -- an example
# failing both a citation check and a format check is counted once, under
# "citation_invalid" (the semantically more serious failure), so the three buckets
# below plus "accepted" always partition the input exactly (no example is double
# counted, none is silently uncounted).
CITATION_INVALID = "citation_invalid"
FORMAT_INVALID = "format"
LOW_FAITHFULNESS = "low_faithfulness"

_BULLET_OR_HEADING_RE = re.compile(r"^\s*(#{1,6}\s|[-*•]\s|\d+[.)]\s)", re.MULTILINE)


def check_answer_format(answer: str) -> tuple[str, ...]:
    """Structural format issues in `answer`, per `grounded_qa_v1.md`'s rule 6 ("one short
    paragraph... no headings, no bullet points, no JSON").

    Citation-*bracket* format (`[doc_id]` vs some other bracket shape) is
    `cragb.eval.citation_validity.find_malformed_citations`'s job, not this function's --
    the two are complementary, both feeding `run_stage1`'s combined format check.

    Returns:
        A tuple of issue tags, empty if `answer` is well-formed. Possible tags:
        `"multiple_paragraphs"` (a blank line, i.e. a second paragraph),
        `"bullet_or_heading"` (a line starting with a markdown heading marker, a bullet
        marker, or a numbered-list marker), `"brace_or_json"` (a literal `{` or `}`,
        which a plain-prose grounded answer should never contain).
    """
    issues: list[str] = []
    if "\n\n" in answer.strip():
        issues.append("multiple_paragraphs")
    if _BULLET_OR_HEADING_RE.search(answer):
        issues.append("bullet_or_heading")
    if "{" in answer or "}" in answer:
        issues.append("brace_or_json")
    return tuple(issues)


@dataclass(frozen=True)
class Stage1Result:
    """Deterministic-stage outcome for one `TrainingExample`."""

    passed: bool
    drop_reason: str | None  # CITATION_INVALID, FORMAT_INVALID, or None if passed
    fabricated_citations: tuple[str, ...]
    citation_mismatch: bool  # extract_citations(answer) != example.cited_doc_ids
    malformed_citation_brackets: tuple[str, ...]
    format_issues: tuple[str, ...]


def run_stage1(example: TrainingExample) -> Stage1Result:
    """Re-verify `example`'s citations and answer format from scratch.

    Every check here is, in principle, already guaranteed by an earlier stage of the
    pipeline (T7.1's `TrainingExample.__post_init__`, T7.3's `parse_generated_pairs`,
    T7.4's construction methods) -- this function does not trust that and recomputes
    everything independently, exactly the "re-check what an earlier, separately-run task
    already should have gotten right" discipline `cragb.bench.assemble` established for
    CRAGB v1 itself.

    Args:
        example: the training example to check.

    Returns:
        A `Stage1Result`. `self_contradiction` (abstention with citations) and
        `ungrounded_answer` (non-abstention with none) are folded into
        `citation_invalid` alongside fabrication/mismatch, matching
        `ft_filter_v1.csv`'s single `n_dropped_citation_invalid` column -- they are not
        reported as separate reasons.
    """
    context_doc_ids = set(example.source_doc_ids)
    cited = example.cited_doc_ids
    fabricated = tuple(c for c in cited if c not in context_doc_ids)
    citation_mismatch = extract_citations(example.answer) != cited
    self_contradiction = example.is_abstention and bool(cited)
    ungrounded_answer = (not example.is_abstention) and not cited

    citation_invalid = bool(fabricated) or citation_mismatch or self_contradiction or ungrounded_answer

    malformed = find_malformed_citations(example.answer)
    format_issues = check_answer_format(example.answer)

    if citation_invalid:
        drop_reason = CITATION_INVALID
    elif malformed or format_issues:
        drop_reason = FORMAT_INVALID
    else:
        drop_reason = None

    return Stage1Result(
        passed=drop_reason is None,
        drop_reason=drop_reason,
        fabricated_citations=fabricated,
        citation_mismatch=citation_mismatch,
        malformed_citation_brackets=malformed,
        format_issues=format_issues,
    )


def run_stage2(
    example: TrainingExample,
    template: Template,
    usage_fn: UsageFn,
    faithfulness_threshold: int,
) -> tuple[bool, int, CompletionResult]:
    """Judge `example`'s faithfulness; `example.answer` stands in as its own reference
    (see module docstring for why that's safe for this one criterion).

    Args:
        example: a stage-1-passing, non-abstention example.
        template: `answer_judge_v1.md`, pre-loaded.
        usage_fn: `GroqClient.complete_with_usage` (or a test stub of the same shape).
        faithfulness_threshold: minimum faithfulness score (1-5) to pass.

    Returns:
        `(passed, faithfulness_score, completion)` -- `completion` is always returned
        (even on a fail) so the caller can fold it into judge-cost accounting regardless
        of the outcome; every stage-2 attempt costs a call whether or not it passes.

    Raises:
        ValueError: propagated from `parse_judge_response` on a malformed judge reply.
    """
    prompt = build_judge_prompt(
        question=example.question,
        context_text=example.context_text,
        candidate_answer=example.answer,
        reference_answer=example.answer,
        template=template,
    )
    completion = usage_fn([{"role": "user", "content": prompt}])
    score = parse_judge_response(completion.text)
    return score.faithfulness >= faithfulness_threshold, score.faithfulness, completion


@dataclass(frozen=True)
class FilterResult:
    """Per-example outcome of the full two-stage filter."""

    example: TrainingExample
    accepted: bool
    drop_reason: str | None  # CITATION_INVALID, FORMAT_INVALID, LOW_FAITHFULNESS, or None
    faithfulness_score: int | None  # None unless stage 2 was reached
    judge_completion: CompletionResult | None  # None unless stage 2 was reached


def filter_examples(
    examples: list[TrainingExample],
    template: Template,
    usage_fn: UsageFn,
    faithfulness_threshold: int,
    show_progress: bool = True,
) -> list[FilterResult]:
    """Run both filter stages over `examples`, in order, one `FilterResult` each.

    Args:
        examples: candidate training examples -- T7.3's positives and T7.4's
            abstentions, combined into one stream (both need stage 1; only positives
            reach stage 2).
        template: `answer_judge_v1.md`, pre-loaded.
        usage_fn: `GroqClient.complete_with_usage` (or a test stub).
        faithfulness_threshold: forwarded to `run_stage2`.
        show_progress: show a `tqdm` progress bar (disable in tests).

    Returns:
        One `FilterResult` per input example, same order.
    """
    results: list[FilterResult] = []
    for example in tqdm(examples, desc="Filtering training pairs", disable=not show_progress):
        stage1 = run_stage1(example)
        if not stage1.passed:
            results.append(
                FilterResult(
                    example=example,
                    accepted=False,
                    drop_reason=stage1.drop_reason,
                    faithfulness_score=None,
                    judge_completion=None,
                )
            )
            continue

        if example.is_abstention:
            results.append(
                FilterResult(
                    example=example, accepted=True, drop_reason=None,
                    faithfulness_score=None, judge_completion=None,
                )
            )
            continue

        passed, faithfulness, completion = run_stage2(example, template, usage_fn, faithfulness_threshold)
        results.append(
            FilterResult(
                example=example,
                accepted=passed,
                drop_reason=None if passed else LOW_FAITHFULNESS,
                faithfulness_score=faithfulness,
                judge_completion=completion,
            )
        )

    return results


# --------------------------------------------------------------------------
# Funnel + judge-cost reporting
# --------------------------------------------------------------------------


def _funnel_row(results: list[FilterResult], slice_name: str) -> dict:
    """One funnel row for `results` (already restricted to one slice)."""
    n_raw = len(results)
    n_dropped_citation_invalid = sum(1 for r in results if r.drop_reason == CITATION_INVALID)
    n_dropped_format = sum(1 for r in results if r.drop_reason == FORMAT_INVALID)
    n_dropped_low_faithfulness = sum(1 for r in results if r.drop_reason == LOW_FAITHFULNESS)
    n_accepted = sum(1 for r in results if r.accepted)

    assert n_raw == n_dropped_citation_invalid + n_dropped_format + n_dropped_low_faithfulness + n_accepted, (
        f"funnel arithmetic does not close for slice {slice_name!r}: "
        f"{n_raw} != {n_dropped_citation_invalid} + {n_dropped_format} + "
        f"{n_dropped_low_faithfulness} + {n_accepted}"
    )

    return {
        "slice": slice_name,
        "n_raw": n_raw,
        "n_dropped_citation_invalid": n_dropped_citation_invalid,
        "n_dropped_format": n_dropped_format,
        "n_dropped_low_faithfulness": n_dropped_low_faithfulness,
        "n_accepted": n_accepted,
    }


def _judge_cost_columns(
    results: list[FilterResult], judge_model: str, pricing: dict[str, ModelPricing]
) -> dict:
    """Judge-call cost summary, computed from every stage-2 `CompletionResult` in `results`
    (accepted or not -- every attempt costs a call).
    """
    calls = [r.judge_completion for r in results if r.judge_completion is not None]
    if not calls:
        return {
            "judge_n_calls": 0,
            "judge_n_cache_hits": 0,
            "judge_total_prompt_tokens": 0,
            "judge_total_completion_tokens": 0,
            "judge_total_usd": 0.0,
        }
    return {
        "judge_n_calls": len(calls),
        "judge_n_cache_hits": sum(1 for c in calls if c.cached),
        "judge_total_prompt_tokens": sum(c.prompt_tokens or 0 for c in calls),
        "judge_total_completion_tokens": sum(c.completion_tokens or 0 for c in calls),
        "judge_total_usd": sum(
            cost_usd(c.prompt_tokens or 0, c.completion_tokens or 0, judge_model, pricing) for c in calls
        ),
    }


def build_filter_funnel(
    results: list[FilterResult],
    judge_model: str,
    pricing: dict[str, ModelPricing],
) -> pd.DataFrame:
    """`ft_filter_v1.csv`'s exact shape: the overall funnel, sliced by category and by
    `is_abstention`, with judge-call cost columns on the `"overall"` row only.

    Args:
        results: from `filter_examples`.
        judge_model: the judge model id, for the cost lookup.
        pricing: `{model: ModelPricing}`, e.g. `cragb.eval.cost_model.load_pricing_config()`.

    Returns:
        One row per slice (`"overall"`, `"category:<name>"` for each category present,
        `"is_abstention:True"`/`"is_abstention:False"`), columns `slice, n_raw,
        n_dropped_citation_invalid, n_dropped_format, n_dropped_low_faithfulness,
        n_accepted` plus `judge_n_calls, judge_n_cache_hits, judge_total_prompt_tokens,
        judge_total_completion_tokens, judge_total_usd` (present only on the `"overall"`
        row -- `NaN` elsewhere, since judge cost is a whole-run total, not a
        per-slice-meaningful figure the way the funnel counts are).

    Raises:
        ValueError: if `results` is empty.
        AssertionError: if any slice's funnel arithmetic doesn't close (see `_funnel_row`).
    """
    if not results:
        raise ValueError("results must not be empty")

    overall_row = {**_funnel_row(results, "overall"), **_judge_cost_columns(results, judge_model, pricing)}
    rows = [overall_row]

    categories = sorted({r.example.category for r in results})
    for category in categories:
        subset = [r for r in results if r.example.category == category]
        rows.append(_funnel_row(subset, f"category:{category}"))

    for is_abstention in (False, True):
        subset = [r for r in results if r.example.is_abstention == is_abstention]
        rows.append(_funnel_row(subset, f"is_abstention:{is_abstention}"))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """Resolved, validated `configs/finetune.yaml`'s `filter.faithfulness_threshold`.

    The judge provider block (`filter.judge.provider`) is read directly from the raw
    config dict in `main()`, not wrapped here -- it's pure `GroqClient` construction
    plumbing, the same pattern `cragb.finetune.generate_pairs.main` already uses for its
    own provider block, not a knob this module's own logic branches on.
    """

    faithfulness_threshold: int

    def __post_init__(self) -> None:
        if not 1 <= self.faithfulness_threshold <= 5:
            raise ValueError(f"faithfulness_threshold must be in [1, 5], got {self.faithfulness_threshold}")


def load_filter_config(config_path: str = "configs/finetune.yaml") -> FilterConfig:
    cfg = load_config(config_path)
    return FilterConfig(faithfulness_threshold=int(cfg["filter"]["faithfulness_threshold"]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune.yaml", help="Path to fine-tuning config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    raw_cfg = load_config(args.config)
    config = load_filter_config(args.config)
    set_global_seed(raw_cfg["seed"])

    template = load_prompt_template(raw_cfg["paths"]["answer_judge_prompt_template"])
    positives = load_training_examples_jsonl(raw_cfg["paths"]["raw_pairs_out"])
    abstentions = load_training_examples_jsonl(raw_cfg["paths"]["abstentions_out"])
    examples = positives + abstentions

    judge_provider_cfg = raw_cfg["filter"]["judge"]["provider"]
    client = GroqClient(
        model=judge_provider_cfg["model"],
        api_base=judge_provider_cfg["api_base"],
        api_key_env=judge_provider_cfg["api_key_env"],
        temperature=judge_provider_cfg["temperature"],
        max_tokens=judge_provider_cfg["max_tokens"],
        reasoning_effort=judge_provider_cfg.get("reasoning_effort"),
        timeout_s=judge_provider_cfg["timeout_s"],
        max_retries=judge_provider_cfg["max_retries"],
        cache_dir=raw_cfg["paths"]["cache_dir"],
    )

    results = filter_examples(examples, template, client.complete_with_usage, config.faithfulness_threshold)

    accepted = [r.example for r in results if r.accepted]
    out_path = write_training_examples_jsonl(accepted, raw_cfg["paths"]["filtered_pairs_out"])

    funnel = build_filter_funnel(results, judge_model=judge_provider_cfg["model"], pricing=load_pricing_config())
    funnel_path = resolve_path(raw_cfg["paths"]["filter_report_out"])
    funnel_path.parent.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(funnel_path, index=False)

    overall = funnel.iloc[0]
    logger.info(
        "n_raw=%d -> dropped citation_invalid=%d, format=%d, low_faithfulness=%d -> accepted=%d",
        overall["n_raw"],
        overall["n_dropped_citation_invalid"],
        overall["n_dropped_format"],
        overall["n_dropped_low_faithfulness"],
        overall["n_accepted"],
    )
    logger.info(
        "Judge calls: %d (%d cache hits), $%.4f",
        overall["judge_n_calls"],
        overall["judge_n_cache_hits"],
        overall["judge_total_usd"],
    )
    logger.info("Wrote %d accepted examples to %s", len(accepted), out_path)
    logger.info("Wrote funnel to %s", funnel_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
