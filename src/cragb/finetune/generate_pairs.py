"""Teacher generation of grounded, cited QA pairs (T7.3; PLAN.md §3 E8, §10, M7.md T7.3).

Turns each of T7.2's sampled context groups into a handful of (question, grounded and
cited answer) pairs, using `openai/gpt-oss-120b` -- the same model RQ1's "large" arm
(`configs/grounded_qa_large.yaml`) already uses -- as the teacher. This is *distillation*
in PLAN.md §10's sense: a small local model will later be fine-tuned to imitate the large
model's grounding and citation discipline, not to gain new facts, so the teacher's job is
narrow -- write questions the shown excerpts genuinely support, and answer them the exact
way T4a.1's `grounded_qa_v1.md` already asks the *inference-time* model to.

**The generation prompt restates T4a.1's citation rules rather than referencing them**
(`prompts/finetune_gen_v1.md`) -- the teacher only ever sees this one prompt, so the rules
have to be self-contained here, not "see the other file."

**Citation fabrication is caught here, at parse time, not deferred to T7.5's filter.**
`parse_generated_pairs` extracts every `[doc_id]` citation from a generated answer (via
`cragb.bench.reference_answers.extract_citations`, the same citation parser T4a/T4b's
inference-time transcripts use) and drops any pair whose citations aren't a subset of the
doc ids actually shown in that context. This is a cheap, structural check -- it needs no
model call and no judge -- so there is no reason to let a pair with an invented review id
survive into the more expensive T7.5 pipeline stage.

**Resumability has two file, not one.** `raw_pairs_v1.jsonl` accumulates only *accepted*
`TrainingExample`s -- but a context group that the teacher answered with zero usable pairs
(every candidate dropped for a fabricated citation, say) writes *nothing* there, which
would make "has this context already been attempted" unrecoverable from that file alone
on a resumed run. `raw_pairs_v1_progress.jsonl` is the actual source of truth: one row per
*attempted* context group, written regardless of how many pairs it yielded, carrying the
call's token/latency/cached fields so `build_generation_cost_row` can report the true
cumulative cost across however many resumed sessions it took to build the full dataset --
not just whatever ran in the current process (PLAN.md §14.5's lesson about logging usage
from the first API-calling line of code, applied to a resumable batch this time).

Every function that would otherwise need a live API call takes an injected `client:
GroqClient` (mirroring `cragb.generate.draft_questions`/`cragb.generate.grounded_qa`'s
testability shape) or, for the pure parsing functions, no client at all -- only `main()`
constructs a real one.

Usage:
    python -m cragb.finetune.generate_pairs --config configs/finetune.yaml [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from cragb.bench.reference_answers import extract_citations
from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec, load_taxonomy
from cragb.eval.cost_model import ModelPricing, UsageFn, cost_usd, load_pricing_config
from cragb.finetune.sample_contexts import ContextGroup, load_contexts_jsonl
from cragb.finetune.schema import TrainingExample
from cragb.generate.api_clients import GroqClient
from cragb.generate.draft_questions import dedup_questions
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

PROMPT_VERSION = "finetune_gen_v1"

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_CODE_FENCE_LANG_RE = re.compile(r"^json\s*", re.IGNORECASE)


# --------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------


def render_generation_prompt(
    context_group: ContextGroup,
    category: TaxonomyCategory,
    n: int,
    template: Template,
) -> str:
    """Render `finetune_gen_v1.md` for one context group.

    Args:
        context_group: T7.2's sampled context (supplies `$context_block`).
        category: the taxonomy category to draft questions in -- supplies
            `$category_name`/`$category_description`. Must match `context_group.category`
            (guarded below): the caller looks this up from the same `TaxonomySpec`
            `context_group.category` was assigned from, so a mismatch would only ever
            indicate a caller bug, not legitimate input variation.
        n: number of questions to ask the teacher for.
        template: `finetune_gen_v1.md`, loaded via
            `cragb.generate.grounded_qa.load_prompt_template`.

    Returns:
        The rendered prompt text.

    Raises:
        ValueError: if `category.name != context_group.category`.
    """
    if category.name != context_group.category:
        raise ValueError(
            f"category {category.name!r} does not match context_group.category "
            f"{context_group.category!r} for group {context_group.group_id!r}"
        )
    return template.substitute(
        category_name=category.name,
        category_description=category.description,
        context_block=context_group.context_text,
        n_questions=n,
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_raw_items(raw: str) -> list[dict[str, str]]:
    """Strict JSON-array extraction: `raw` -> `[{"question": str, "answer": str}, ...]`.

    Tolerant of an accidental markdown code fence around the JSON (models sometimes add
    one despite being told not to -- the same allowance
    `cragb.generate.draft_questions.parse_llm_questions` and
    `cragb.eval.judge.parse_judge_response` make for the same reason). Anything else that
    fails to parse -- including a genuinely truncated array with no closing `]` at all, or
    one cut off mid-object -- raises with the raw response attached, rather than silently
    returning a partial list: a parseable-looking but cut-off response is a distinct
    failure shape from an empty one (PLAN.md §14.6's `gemini-3.6-flash` lesson), and both
    must be visible, not swallowed.

    Args:
        raw: the teacher model's raw completion text.

    Returns:
        Parsed items, in the model's original order. An empty list (`[]`) is a
        well-formed result -- the model deciding this context supports zero questions --
        not a parse failure, and is returned as-is rather than raised on.

    Raises:
        ValueError: if no valid JSON array can be extracted from `raw`, or if any element
            is not an object carrying both `"question"` and `"answer"` string keys.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = _CODE_FENCE_LANG_RE.sub("", text)

    match = _JSON_ARRAY_RE.search(text)
    candidate_text = match.group(0) if match else text

    try:
        items = json.loads(candidate_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse a JSON array from the teacher's response: {e}\n"
            f"--- raw response ---\n{raw}"
        ) from e

    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array from the teacher, got {type(items).__name__}: {items!r}")

    parsed: list[dict[str, str]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(f"Malformed item {i} in teacher response: {item!r}")
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
        if not question or not answer:
            raise ValueError(f"Item {i} has an empty question or answer: {item!r}")
        parsed.append({"question": question, "answer": answer})
    return parsed


def parse_generated_pairs(
    raw: str,
    context_group: ContextGroup,
    *,
    provenance: dict,
) -> list[TrainingExample]:
    """Parse `raw` into validated `TrainingExample`s, dropping (not raising on) bad items.

    Two independent per-item checks, each a silent drop (logged, not raised) rather than
    an all-or-nothing failure for the whole response -- one bad item in an otherwise good
    batch of 3 should not throw away the other 2:

    - **Citation fabrication.** `extract_citations(answer)` must be a subset of
      `context_group.doc_ids` -- a citation to a doc id never shown in this context is
      fabricated evidence, and cheap to catch here rather than waiting for T7.5's judge.
    - **Structural invariants** (`cragb.finetune.schema.TrainingExample.__post_init__`) --
      e.g. an answer that happens to contain the literal `ABSTENTION_TEXT` phrase (this
      prompt never asks for one; if the teacher writes one anyway, `TrainingExample`
      itself raises `ValueError` on construction since `is_abstention=False` here always).

    A genuinely malformed *response* (not parseable as a JSON array at all, or missing the
    `question`/`answer` keys) is a different failure shape and is **not** caught here --
    `parse_raw_items` raises for that, and it is the caller's job to decide what "the whole
    response failed" means for resumability (see `generate_all`).

    Args:
        raw: the teacher's raw completion text.
        context_group: the context this response was generated for -- supplies the doc-id
            allowlist for citation validation and every `TrainingExample` field that isn't
            in `raw` itself (`category`, `source_doc_ids`, `source_parent_asins`,
            `context_text`).
        provenance: base provenance dict (e.g. `{"method": "teacher_generation",
            "teacher_model": ..., "prompt_version": ...}`) merged with
            `context_group_id`/`raw_item_index` per example -- the former is what
            `generate_all`'s resumability skip-check and `build_generation_cost_row`'s
            per-context accounting both key on.

    Returns:
        Accepted `TrainingExample`s, in `raw`'s original item order (fewer than
        `len(parse_raw_items(raw))` iff at least one item was dropped).

    Raises:
        ValueError: propagated from `parse_raw_items` on a malformed/truncated response.
    """
    items = parse_raw_items(raw)
    examples: list[TrainingExample] = []

    for i, item in enumerate(items):
        cited = extract_citations(item["answer"])
        fabricated = set(cited) - set(context_group.doc_ids)
        if fabricated:
            logger.warning(
                "Context %s item %d: dropping -- cites unknown doc id(s) %s not in this context.",
                context_group.group_id,
                i,
                sorted(fabricated),
            )
            continue

        try:
            example = TrainingExample(
                example_id=f"{context_group.group_id}_{i:02d}",
                category=context_group.category,
                source_doc_ids=context_group.doc_ids,
                source_parent_asins=(context_group.parent_asin,),
                question=item["question"],
                context_text=context_group.context_text,
                answer=item["answer"],
                cited_doc_ids=cited,
                is_abstention=False,
                provenance={
                    **provenance,
                    "context_group_id": context_group.group_id,
                    "raw_item_index": i,
                },
            )
        except ValueError as e:
            logger.warning("Context %s item %d: dropping -- %s", context_group.group_id, i, e)
            continue

        examples.append(example)

    return examples


# --------------------------------------------------------------------------
# Resumable batch generation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationReport:
    """Summary of one `generate_all` invocation, for the printed progress line + logs."""

    n_contexts_total: int
    n_contexts_done: int  # cumulative, across every resumed session, after this run
    n_contexts_processed_this_run: int
    n_contexts_skipped_already_done: int
    n_new_calls: int
    n_cache_hits: int
    n_rate_limited: int
    n_errors: int
    n_examples_generated_raw: int
    n_examples_dropped_invalid: int
    n_examples_dropped_near_duplicate: int
    n_examples_accepted: int


def load_done_context_group_ids(progress_path: str | Path) -> set[str]:
    """`context_group_id`s already attempted, per `raw_pairs_v1_progress.jsonl`.

    Returns an empty set if `progress_path` doesn't exist yet (a first, non-resumed run).
    """
    path = resolve_path(progress_path)
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["context_group_id"])
    return done


def _load_progress_rows(progress_path: str | Path) -> list[dict]:
    path = resolve_path(progress_path)
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_progress_row(
    progress_path: str | Path,
    context_group_id: str,
    *,
    n_accepted: int,
    cached: bool,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_s: float | None,
) -> None:
    """Append one attempted-context row to the resumability/cost source-of-truth log."""
    path = resolve_path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "context_group_id": context_group_id,
        "n_accepted": n_accepted,
        "cached": cached,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_s": latency_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def append_training_examples_jsonl(examples: list[TrainingExample], out_path: str | Path) -> Path:
    """Append `examples` to `raw_pairs_v1.jsonl` (does not truncate -- this is a resumable log)."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False))
            f.write("\n")
    return resolved


def generate_all(
    contexts: list[ContextGroup],
    *,
    taxonomy: TaxonomySpec,
    template: Template,
    usage_fn: UsageFn,
    questions_per_context: int,
    dedup_threshold: float,
    raw_pairs_path: str | Path,
    progress_path: str | Path,
    limit: int | None = None,
    show_progress: bool = True,
) -> GenerationReport:
    """Generate training pairs for every not-yet-done context in `contexts`.

    Resumable: contexts already recorded in `progress_path` are skipped without an API
    call. Interrupting a run and restarting it produces the same `raw_pairs_path`/
    `progress_path` contents as an uninterrupted run over the same `contexts`, because
    each context is fully processed (call -> parse -> append both files) before the next
    one starts, and a cached teacher call returns byte-identical text on resume (same
    request payload -> same `DiskCache` key), so re-parsing it yields the same accepted
    examples in the same order.

    Args:
        contexts: context groups to (attempt to) process, in order (e.g.
            `cragb.finetune.sample_contexts.load_contexts_jsonl`'s output).
        taxonomy: supplies each context's `TaxonomyCategory` (for the prompt's
            `$category_description`) by name.
        template: `finetune_gen_v1.md`, pre-loaded.
        usage_fn: `GroqClient.complete_with_usage` (or a plain test stub of the same
            shape, `Callable[[messages], CompletionResult]` --
            `cragb.eval.cost_model.UsageFn`), already bound to a client configured for the
            teacher model.
        questions_per_context: `n` passed to `render_generation_prompt`.
        dedup_threshold: `cragb.generate.draft_questions.dedup_questions`'s
            `SequenceMatcher` ratio threshold, applied per context group (near-duplicate
            questions are only checked *within* one context's own batch, not globally --
            two different products can legitimately prompt a similar-sounding question).
        raw_pairs_path: `TrainingExample`s are appended here.
        progress_path: one row appended here per *attempted* context (see module
            docstring for why this, not `raw_pairs_path`, is the resumability source).
        limit: process at most this many **not-yet-done** contexts this invocation (an
            already-done context, encountered while scanning `contexts`, does not count
            against this limit). `None` (default) processes every not-yet-done context.
        show_progress: show a `tqdm` progress bar (disable in tests).

    Returns:
        A `GenerationReport` covering this invocation only (`n_contexts_done` is the one
        cumulative field -- everything else describes just this run).

    Raises:
        cragb.generate.api_clients.MissingAPIKeyError: propagated immediately (not
            caught per-context) -- a missing key fails identically for every context, so
            there is nothing to gain from skip-and-continue.
    """
    category_by_name = {c.name: c for c in taxonomy.categories}
    already_done = load_done_context_group_ids(progress_path)

    n_contexts_processed_this_run = 0
    n_contexts_skipped_already_done = 0
    n_new_calls = 0
    n_cache_hits = 0
    n_rate_limited = 0
    n_errors = 0
    n_examples_generated_raw = 0
    n_examples_dropped_near_duplicate = 0
    n_examples_accepted = 0

    iterator = tqdm(contexts, desc="Generating training pairs", disable=not show_progress)
    for context_group in iterator:
        if context_group.group_id in already_done:
            n_contexts_skipped_already_done += 1
            continue
        if limit is not None and n_contexts_processed_this_run >= limit:
            break

        category = category_by_name[context_group.category]
        prompt = render_generation_prompt(context_group, category, questions_per_context, template)

        try:
            result = usage_fn([{"role": "user", "content": prompt}])
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                n_rate_limited += 1
                logger.warning(
                    "Context %s: rate-limited (429) after retries exhausted; will retry on resume.",
                    context_group.group_id,
                )
            else:
                n_errors += 1
                logger.warning(
                    "Context %s: HTTP error (status=%s); will retry on resume.",
                    context_group.group_id,
                    status,
                )
            continue

        if result.cached:
            n_cache_hits += 1
        else:
            n_new_calls += 1

        provenance = {
            "method": "teacher_generation",
            "teacher_model": result.model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            n_raw = len(parse_raw_items(result.text))
            candidates = parse_generated_pairs(result.text, context_group, provenance=provenance)
        except ValueError:
            n_errors += 1
            logger.warning(
                "Context %s: could not parse teacher response; will retry on resume.",
                context_group.group_id,
                exc_info=True,
            )
            continue

        deduped, n_near_dup = dedup_questions(candidates, threshold=dedup_threshold)

        append_training_examples_jsonl(deduped, raw_pairs_path)
        append_progress_row(
            progress_path,
            context_group.group_id,
            n_accepted=len(deduped),
            cached=result.cached,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=result.latency_s,
        )
        already_done.add(context_group.group_id)

        n_contexts_processed_this_run += 1
        n_examples_generated_raw += n_raw
        n_examples_dropped_near_duplicate += n_near_dup
        n_examples_accepted += len(deduped)

    n_examples_dropped_invalid = n_examples_generated_raw - n_examples_dropped_near_duplicate - n_examples_accepted

    return GenerationReport(
        n_contexts_total=len(contexts),
        n_contexts_done=len(already_done),
        n_contexts_processed_this_run=n_contexts_processed_this_run,
        n_contexts_skipped_already_done=n_contexts_skipped_already_done,
        n_new_calls=n_new_calls,
        n_cache_hits=n_cache_hits,
        n_rate_limited=n_rate_limited,
        n_errors=n_errors,
        n_examples_generated_raw=n_examples_generated_raw,
        n_examples_dropped_invalid=n_examples_dropped_invalid,
        n_examples_dropped_near_duplicate=n_examples_dropped_near_duplicate,
        n_examples_accepted=n_examples_accepted,
    )


# --------------------------------------------------------------------------
# Cost accounting
# --------------------------------------------------------------------------


def build_generation_cost_row(
    progress_path: str | Path,
    model: str,
    pricing: dict[str, ModelPricing],
) -> pd.DataFrame:
    """One-row cost summary from the *entire* progress log (every resumed session).

    Reading `progress_path` rather than accumulating totals in-process is what makes this
    correct across resumed runs: a call made in an earlier, already-exited invocation
    contributes its tokens/latency here exactly as one made in the current invocation
    does, since both wrote the same row shape to the same durable log.

    Args:
        progress_path: `raw_pairs_v1_progress.jsonl` (or a test double of the same shape).
        model: the teacher model id, for the `pricing` lookup and the output row.
        pricing: `{model: ModelPricing}`, e.g. `cragb.eval.cost_model.load_pricing_config()`.

    Returns:
        A one-row DataFrame: `model, n_calls, n_cache_hits, total_prompt_tokens,
        total_completion_tokens, mean_prompt_tokens, mean_completion_tokens,
        total_wall_clock_s, total_usd, n_calls_missing_usage`. `total_wall_clock_s` sums
        only non-cached calls' `latency_s` (a cache hit's near-zero replay time doesn't
        describe how long the underlying generation actually took, mirroring T5.5's
        `bypass_cache` reasoning for why a hit's latency is never reported as real).
        `n_calls_missing_usage` counts calls whose token counts came back `None` (e.g. the
        API omitted `usage`), which are treated as zero tokens in the totals rather than
        breaking the sum -- flagged via this count rather than silently mixed in
        unlabeled, mirroring `cost_model`'s `is_estimated` convention.

    Raises:
        ValueError: if the progress log is empty (nothing has been generated yet).
    """
    rows = _load_progress_rows(progress_path)
    if not rows:
        raise ValueError(
            f"No rows in {progress_path!r} -- run generation before building a cost table."
        )

    n_calls = len(rows)
    n_cache_hits = sum(1 for r in rows if r["cached"])
    n_missing_usage = sum(
        1 for r in rows if r["prompt_tokens"] is None or r["completion_tokens"] is None
    )
    total_prompt_tokens = sum(r["prompt_tokens"] or 0 for r in rows)
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in rows)
    total_wall_clock_s = sum(r["latency_s"] or 0.0 for r in rows if not r["cached"])
    total_usd = sum(
        cost_usd(r["prompt_tokens"] or 0, r["completion_tokens"] or 0, model, pricing) for r in rows
    )

    return pd.DataFrame(
        [
            {
                "model": model,
                "n_calls": n_calls,
                "n_cache_hits": n_cache_hits,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "mean_prompt_tokens": total_prompt_tokens / n_calls,
                "mean_completion_tokens": total_completion_tokens / n_calls,
                "total_wall_clock_s": total_wall_clock_s,
                "total_usd": total_usd,
                "n_calls_missing_usage": n_missing_usage,
            }
        ]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune.yaml", help="Path to fine-tuning config YAML.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many not-yet-done contexts this run "
        "(e.g. --limit 5 for a live smoke test before the full sweep).",
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

    raw_cfg = load_config(args.config)
    set_global_seed(raw_cfg["seed"])

    taxonomy = load_taxonomy(raw_cfg["paths"]["taxonomy_config"])
    template = load_prompt_template(raw_cfg["paths"]["finetune_gen_prompt_template"])
    contexts = load_contexts_jsonl(raw_cfg["paths"]["contexts_out"])

    gen_cfg = raw_cfg["generation"]
    provider_cfg = gen_cfg["provider"]
    client = GroqClient(
        model=provider_cfg["model"],
        api_base=provider_cfg["api_base"],
        api_key_env=provider_cfg["api_key_env"],
        temperature=provider_cfg["temperature"],
        max_tokens=provider_cfg["max_tokens"],
        timeout_s=provider_cfg["timeout_s"],
        max_retries=provider_cfg["max_retries"],
        cache_dir=raw_cfg["paths"]["cache_dir"],
    )

    report = generate_all(
        contexts,
        taxonomy=taxonomy,
        template=template,
        usage_fn=client.complete_with_usage,
        questions_per_context=gen_cfg["questions_per_context"],
        dedup_threshold=gen_cfg["near_duplicate_threshold"],
        raw_pairs_path=raw_cfg["paths"]["raw_pairs_out"],
        progress_path=raw_cfg["paths"]["raw_pairs_progress_out"],
        limit=args.limit,
    )

    cost_df = build_generation_cost_row(
        raw_cfg["paths"]["raw_pairs_progress_out"],
        model=provider_cfg["model"],
        pricing=load_pricing_config(),
    )
    cost_path = resolve_path(raw_cfg["paths"]["generation_cost_out"])
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_df.to_csv(cost_path, index=False)

    logger.info(
        "Contexts done %d/%d (this run: processed %d, skipped %d already-done)",
        report.n_contexts_done,
        report.n_contexts_total,
        report.n_contexts_processed_this_run,
        report.n_contexts_skipped_already_done,
    )
    logger.info(
        "New calls %d, cache hits %d, rate-limited (429) %d, other errors %d",
        report.n_new_calls,
        report.n_cache_hits,
        report.n_rate_limited,
        report.n_errors,
    )
    logger.info(
        "Examples this run: %d generated -> %d dropped invalid, %d dropped near-duplicate -> %d accepted",
        report.n_examples_generated_raw,
        report.n_examples_dropped_invalid,
        report.n_examples_dropped_near_duplicate,
        report.n_examples_accepted,
    )
    logger.info("Wrote cost summary to %s", cost_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
