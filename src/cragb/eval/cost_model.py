"""Pricing config + token accounting → $/query per arm (T5.4; PLAN.md §3 E6, §8 G4, M5.md T5.4).

Two independent halves, deliberately kept separate:

- **Pure cost arithmetic** (`ModelPricing`, `load_pricing_config`, `cost_usd`,
  `arm_cost_table`): turns token counts into dollars. No network, no cache,
  fully unit-testable against hand-computed numbers.
- **Token recovery** (`build_messages_for_row`, `reissue_transcripts_for_tokens`):
  the 180 transcripts T4b.2 generated predate T5.2's usage telemetry
  (`cragb.generate.api_clients.GroqClient.complete_with_usage`), so their
  token counts don't exist anywhere on disk. Recovering them means
  re-issuing the *exact same request* each transcript already made and
  reading whatever the cache (or, on a genuine miss, a fresh call) reports
  back. Getting "exact same request" right is the whole difficulty here:
  the request is `[{"role": "user", "content": <rendered prompt>}]`, and
  the rendered prompt must be byte-identical to what T4a.3/T4b.1 originally
  sent, or this becomes a cache **miss** and a live (quota-spending, and
  wrong-latency-timed) API call instead of a free cache hit.

  The one thing that makes byte-identical reconstruction possible without
  re-running retrieval: `answer_gen_rag_{small,large}_v1.jsonl` already
  persist `context_text` — the exact `$context_block` value substituted
  into T4a.1's prompt template originally. `render_prompt` (both
  `cragb.generate.grounded_qa`'s and `cragb.generate.closed_book_qa`'s) is
  a pure function of `(question, context_text)`, so re-rendering from the
  saved transcript reproduces the same prompt without touching a
  retriever, an index, or the GPU. This module dispatches on the presence
  of `context_text` in a transcript row rather than an `arm` string, since
  that is the actual thing determining which `render_prompt` applies.

  Following this project's established testability shape (T4a.3/T4b.1's
  injected `chat_fn`), `reissue_transcripts_for_tokens` takes a
  `usage_fn: Callable[[messages], CompletionResult]` rather than a
  concrete `GroqClient` — tests inject a fake, `main()` passes
  `GroqClient.complete_with_usage`.

`is_estimated` (per-row and, in `arm_cost_table`, per-arm) marks token
counts recovered via a `len(text) / 4` chars-per-token fallback rather
than the API's real `usage` block — used only when a row's usage is
genuinely unrecoverable (a cache entry from before T5.2, with no
sidecar, whose original response text is still on disk). Measured and
estimated numbers are never silently mixed into one unlabeled column.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable

import numpy as np
import pandas as pd

from cragb.eval.bootstrap import bootstrap_ci
from cragb.generate.api_clients import CompletionResult, GroqClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4.0


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """One `configs/pricing.yaml` model entry."""

    input_usd_per_1m: float
    output_usd_per_1m: float
    snapshot_date: str
    source_url: str

    def __post_init__(self) -> None:
        if self.input_usd_per_1m < 0:
            raise ValueError(f"input_usd_per_1m must be >= 0, got {self.input_usd_per_1m}")
        if self.output_usd_per_1m < 0:
            raise ValueError(f"output_usd_per_1m must be >= 0, got {self.output_usd_per_1m}")


def load_pricing_config(path: str | Path = "configs/pricing.yaml") -> dict[str, ModelPricing]:
    """Load `configs/pricing.yaml` (or an equivalent file) into `{model: ModelPricing}`.

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if a model entry is missing a required field.
        ValueError: if a rate is negative (`ModelPricing.__post_init__`).
    """
    raw = load_config(path)
    return {
        model: ModelPricing(
            input_usd_per_1m=fields["input_usd_per_1m"],
            output_usd_per_1m=fields["output_usd_per_1m"],
            snapshot_date=fields["snapshot_date"],
            source_url=fields["source_url"],
        )
        for model, fields in raw["models"].items()
    }


def cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing: dict[str, ModelPricing],
) -> float:
    """USD cost of one call: `prompt_tokens` in + `completion_tokens` out, at `model`'s rate.

    Args:
        prompt_tokens: input token count.
        completion_tokens: output token count.
        model: a key in `pricing`.
        pricing: `{model: ModelPricing}`, as returned by `load_pricing_config`.

    Returns:
        Cost in USD.

    Raises:
        KeyError: if `model` has no entry in `pricing`.
    """
    if model not in pricing:
        raise KeyError(f"No pricing entry for model {model!r}; add it to configs/pricing.yaml.")
    rate = pricing[model]
    return (prompt_tokens / 1e6) * rate.input_usd_per_1m + (completion_tokens / 1e6) * rate.output_usd_per_1m


def arm_cost_table(
    call_rows: pd.DataFrame,
    pricing: dict[str, ModelPricing],
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-arm cost summary (`results/tables/answer_cost_v1.csv`'s exact shape).

    Args:
        call_rows: one row per (arm, question) with columns `arm`, `model`,
            `prompt_tokens`, `completion_tokens`, `is_estimated` — as
            produced by `reissue_transcripts_for_tokens`, concatenated
            across arms.
        pricing: `{model: ModelPricing}`.
        n_boot, alpha, rng: forwarded to `cragb.eval.bootstrap.bootstrap_ci`
            for `usd_per_query_ci_{lo,hi}`. Pass a seeded `rng` (e.g. via
            `cragb.utils.seeds.set_global_seed`) for a reproducible CI.

    Returns:
        One row per arm: `arm, model, n_questions, mean_prompt_tokens,
        mean_completion_tokens, mean_usd_per_query, usd_per_query_ci_lo,
        usd_per_query_ci_hi, total_usd, is_estimated`. `total_usd ==
        mean_usd_per_query * n_questions` by construction (both are
        derived from the same per-question cost list). `is_estimated` is
        `True` for an arm iff *any* of its rows used the chars-per-token
        fallback, so a partially-estimated arm is never silently reported
        as fully measured.

    Raises:
        ValueError: if `call_rows` is empty.
        KeyError: propagated from `cost_usd` if a row's model has no
            pricing entry.
    """
    if call_rows.empty:
        raise ValueError("call_rows must be non-empty")

    rows: list[dict] = []
    for arm, group in call_rows.groupby("arm", sort=False):
        models = group["model"].unique()
        if len(models) != 1:
            raise ValueError(
                f"arm {arm!r} has multiple distinct models {sorted(models)}; "
                "RQ0/RQ1 require one model per arm."
            )
        model = models[0]

        costs = [
            cost_usd(pt, ct, model, pricing)
            for pt, ct in zip(group["prompt_tokens"], group["completion_tokens"])
        ]
        ci_lo, ci_hi = bootstrap_ci(costs, n_boot=n_boot, alpha=alpha, rng=rng)

        rows.append(
            {
                "arm": arm,
                "model": model,
                "n_questions": len(group),
                "mean_prompt_tokens": float(group["prompt_tokens"].mean()),
                "mean_completion_tokens": float(group["completion_tokens"].mean()),
                "mean_usd_per_query": float(np.mean(costs)),
                "usd_per_query_ci_lo": ci_lo,
                "usd_per_query_ci_hi": ci_hi,
                "total_usd": float(np.sum(costs)),
                "is_estimated": bool(group["is_estimated"].any()),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Token recovery (re-issuing cached calls through complete_with_usage)
# --------------------------------------------------------------------------

UsageFn = Callable[[list[dict[str, str]]], CompletionResult]


def _estimate_tokens(text: str) -> int:
    """`len(text) / CHARS_PER_TOKEN`, rounded, floored at 1 (a real call is never 0 tokens)."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def build_messages_for_row(row: dict, template: Template) -> list[dict[str, str]]:
    """Reconstruct the exact chat `messages` a saved transcript row originally sent.

    Dispatches on the presence of `context_text` — the RAG arms
    (`answer_gen_rag_{small,large}_v1.jsonl`) have it, the closed-book arm
    (`answer_gen_closed_book_v1.jsonl`) doesn't — rather than on an `arm`
    label, since that key is what actually determines which `render_prompt`
    applies.

    Args:
        row: one parsed JSON line from an `answer_gen_*_v1.jsonl` file.
        template: the arm's prompt template (`load_prompt_template`).

    Returns:
        `[{"role": "user", "content": <rendered prompt>}]`, byte-identical
        to what the original generation call sent, given the same template.
    """
    if "context_text" in row:
        from cragb.generate.context_builder import ContextBlock
        from cragb.generate.grounded_qa import render_prompt as render_grounded_prompt

        context = ContextBlock(text=row["context_text"], doc_ids=(), photo_flags={})
        prompt = render_grounded_prompt(template, row["question"], context)
    else:
        from cragb.generate.closed_book_qa import render_prompt as render_closed_book_prompt

        prompt = render_closed_book_prompt(template, row["question"])
    return [{"role": "user", "content": prompt}]


def reissue_transcripts_for_tokens(
    arm: str,
    rows: list[dict],
    template: Template,
    usage_fn: UsageFn,
) -> pd.DataFrame:
    """Recover per-question token counts for one arm's transcripts.

    Re-sends each row's reconstructed prompt through `usage_fn`. For the
    180 transcripts this project has generated so far, every one of these
    is a disk-cache hit (same model/temperature/max_tokens/prompt as the
    original call) — so this costs nothing and issues no network traffic.
    A row whose usage comes back `None` (a cache entry from before T5.2)
    falls back to `_estimate_tokens` on the reconstructed prompt and the
    returned completion text, flagged via `is_estimated`.

    Args:
        arm: label carried through to the output (`"closed_book"`,
            `"rag_small"`, `"rag_large"`).
        rows: parsed JSON lines from that arm's `answer_gen_*_v1.jsonl`.
        template: that arm's prompt template.
        usage_fn: `GroqClient.complete_with_usage` in production, or a
            fake for tests.

    Returns:
        `[arm, question_id, model, prompt_tokens, completion_tokens,
        is_estimated]`, one row per input row, in input order.
    """
    records: list[dict] = []
    for row in rows:
        messages = build_messages_for_row(row, template)
        result = usage_fn(messages)

        if result.prompt_tokens is not None and result.completion_tokens is not None:
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            is_estimated = False
        else:
            prompt_tokens = _estimate_tokens(messages[0]["content"])
            completion_tokens = _estimate_tokens(result.text)
            is_estimated = True

        records.append(
            {
                "arm": arm,
                "question_id": row["question_id"],
                "model": result.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "is_estimated": is_estimated,
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# (arm label, generation config, transcripts file) for the three RQ0/RQ1 arms T4b.2 built.
_ARMS: tuple[tuple[str, str, str], ...] = (
    ("closed_book", "configs/closed_book_qa.yaml", "results/tables/answer_gen_closed_book_v1.jsonl"),
    ("rag_small", "configs/grounded_qa.yaml", "results/tables/answer_gen_rag_small_v1.jsonl"),
    ("rag_large", "configs/grounded_qa_large.yaml", "results/tables/answer_gen_rag_large_v1.jsonl"),
)


def _load_transcript_rows(path: str | Path) -> list[dict]:
    lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _client_for_arm_config(config_path: str) -> tuple[GroqClient, Template]:
    cfg = load_config(config_path)
    provider = cfg["provider"]
    client = GroqClient(
        model=provider["model"],
        api_base=provider["api_base"],
        api_key_env=provider["api_key_env"],
        temperature=provider["temperature"],
        max_tokens=provider["max_tokens"],
        reasoning_effort=provider.get("reasoning_effort"),
        timeout_s=provider["timeout_s"],
        max_retries=provider["max_retries"],
        cache_dir=cfg["paths"]["cache_dir"],
    )
    template = load_prompt_template(cfg["paths"]["prompt_template"])
    return client, template


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recover token counts for the RQ0/RQ1 answer-generation arms "
        "(re-issuing cached calls through complete_with_usage) and write the per-arm "
        "$/query cost table (T5.4; PLAN.md §3 E6, §8 G4)."
    )
    parser.add_argument("--pricing-config", default="configs/pricing.yaml")
    parser.add_argument("--out", default="results/tables/answer_cost_v1.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pricing = load_pricing_config(args.pricing_config)

    call_rows_per_arm = []
    for arm, config_path, transcripts_path in _ARMS:
        client, template = _client_for_arm_config(config_path)
        rows = _load_transcript_rows(transcripts_path)
        call_rows = reissue_transcripts_for_tokens(arm, rows, template, client.complete_with_usage)
        n_estimated = int(call_rows["is_estimated"].sum())
        logger.info(
            "%s: recovered tokens for %d questions (%d estimated, %d measured)",
            arm,
            len(call_rows),
            n_estimated,
            len(call_rows) - n_estimated,
        )
        call_rows_per_arm.append(call_rows)

    call_rows_all = pd.concat(call_rows_per_arm, ignore_index=True)

    from cragb.utils.seeds import set_global_seed

    seed_state = set_global_seed(args.seed)
    table = arm_cost_table(call_rows_all, pricing, rng=seed_state.numpy_rng)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("wrote cost table (%d rows) to %s", len(table), out_path)


if __name__ == "__main__":
    main()
