"""RQ4 pilot run: photo-evidence win-rate, bootstrap CI, significance vs the
0.5 null (T6.5; PLAN.md §3 E7, §8 G3, §2 RQ4, M6.md T6.5).

Orchestrates the pieces T6.1-T6.4 already built into the pilot's actual
result: load T6.3's usable pairs (`mm_pairs_v1.jsonl`) -> judge each pair in
both photo orders (T6.4's `judge_pair`, order-swap bias control included) ->
write one verdict row per pair -> aggregate into a win-rate table with a
bootstrap 95% CI and a one-sided significance test against the 0.5 null ->
write a cost/latency summary for the run.

**The significance test is one-sided, not two-sided.** RQ4/H4
(PLAN.md §2) is a directional claim -- "the surfaced photo is judged
relevant well above a random-photo baseline" -- not merely "the win rate
differs from 0.5". `scipy.stats.binomtest(..., alternative="greater")`
tests exactly that direction. This matters concretely at the pilot's small
sample size: an all-tie run (win_rate=0.0) must report as *not*
significant, and a two-sided test would instead flag it as a significant
deviation from 0.5 in the *wrong* direction (evidence the photo loses, not
wins) -- `alternative="greater"` gives the correct p ~= 1.0 for that case.

**Win-rate convention (ties count in the denominator, not the numerator).**
`win_rate = n_surfaced_win / n_pairs`, where `n_pairs = n_surfaced_win +
n_control_win + n_tie` -- the single most common way a pairwise win-rate
gets quietly inflated is dropping ties from the denominator entirely.
Every count (`n_surfaced_win`, `n_control_win`, `n_tie`, `n_pairs`) is
written alongside the rate so this is checkable from the CSV alone.

**Cost accounting must add Gemini's hidden thinking tokens to the output
count.** Google bills `thoughtsTokenCount` at the same per-token rate as
visible completion tokens (confirmed live, `configs/vision_judge.yaml`);
`summarize_cost` passes `completion_tokens + thinking_tokens` to
`cragb.eval.cost_model.cost_usd`, not `completion_tokens` alone, or every
number in `mm_cost_v1.csv` would understate the real bill.

**Crash resumability comes from the disk cache, not from code in this
module.** Every judge call goes through `GeminiClient.complete_with_usage`,
which caches on the full request payload (T6.2). If a live run dies partway
through (a malformed response, a network blip), re-running this module from
scratch re-issues every pair's two calls, but every one already answered is
a cache hit -- free, instant, and zero quota spent -- so only the failure
point and beyond actually hits the network again. `run_pilot` therefore
does not catch or skip per-pair failures: a bad response aborts the run
loudly (so it's investigated, not silently missing from the win-rate), and
a re-run picks up where it left off. This is also why `n_pairs` in
`mm_winrate_v1.csv` always equals the row count of `mm_verdicts_v1.jsonl`
and T6.3's `usable_pairs` funnel stage -- nothing here can silently drop a
pair.

Usage:
    python -m cragb.eval.run_multimodal_pilot --config configs/vision_judge.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import binomtest

from cragb.eval.bootstrap import bootstrap_ci
from cragb.eval.cost_model import ModelPricing, cost_usd, load_pricing_config
from cragb.generate.api_clients import CompletionResult
from cragb.generate.gemini_client import GeminiClient
from cragb.generate.grounded_qa import load_prompt_template
from cragb.multimodal.photo_store import PhotoStore
from cragb.multimodal.vision_judge import PairVerdict, judge_pair
from cragb.utils.io import load_config, resolve_path
from cragb.utils.timing import Timer

logger = logging.getLogger(__name__)

VisionUsageFn = Callable[[list[dict[str, Any]]], CompletionResult]

DEFAULT_SEED = 42
DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------
# Loading T6.3's pairs
# --------------------------------------------------------------------------


def load_pairs(path: str | Path) -> pd.DataFrame:
    """Load T6.3's `mm_pairs_v1.jsonl` -- the pair set actually judged.

    Returns:
        A DataFrame with columns `question_id, type, question,
        surfaced_photo_id, surfaced_doc_id, control_photo_id,
        control_doc_id` (T6.3's `write_pairs_jsonl` output shape; every row
        is already usable -- `drop_reason` was dropped before writing).
    """
    rows: list[dict[str, Any]] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Usage recording (adapts GeminiClient.complete_with_usage to judge_pair's
# plain text-in/text-out chat_fn, without touching T6.4's interface)
# --------------------------------------------------------------------------


@dataclass
class UsageRecorder:
    """Wraps a `usage_fn` into `judge_pair`'s plain `chat_fn` shape, recording
    every `CompletionResult` along the way for `summarize_cost` -- `judge_pair`
    itself only ever sees completion text, so T6.4's interface is untouched."""

    usage_fn: VisionUsageFn
    calls: list[CompletionResult] = field(default_factory=list)

    def __call__(self, parts: list[dict[str, Any]]) -> str:
        result = self.usage_fn(parts)
        self.calls.append(result)
        return result.text


# --------------------------------------------------------------------------
# Running the pilot
# --------------------------------------------------------------------------


def _verdict_row(question_id: str, type_: str, result: PairVerdict) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "type": type_,
        "outcome": result.outcome,
        "order_agreement": result.order_agreement,
        "winner_surfaced_as_a": result.verdict_surfaced_as_a.winner,
        "confidence_surfaced_as_a": result.verdict_surfaced_as_a.confidence,
        "rationale_surfaced_as_a": result.verdict_surfaced_as_a.rationale,
        "winner_surfaced_as_b": result.verdict_surfaced_as_b.winner,
        "confidence_surfaced_as_b": result.verdict_surfaced_as_b.confidence,
        "rationale_surfaced_as_b": result.verdict_surfaced_as_b.rationale,
    }


def run_pilot(
    pairs: pd.DataFrame,
    store: PhotoStore,
    template: Template,
    chat_fn: Callable[[list[dict[str, Any]]], str],
) -> pd.DataFrame:
    """Judge every pair in `pairs`, one row per pair, in `pairs`' order.

    Args:
        pairs: `load_pairs`'s output (T6.3's usable pairs).
        store: a `PhotoStore` used to resolve photo ids to image bytes.
        template: the loaded vision-judge prompt template.
        chat_fn: stands in for `GeminiClient.complete` (or a `UsageRecorder`
            wrapping `complete_with_usage`) -- injected so this function
            needs no network/API key to test.

    Returns:
        A DataFrame: `question_id, type, outcome, order_agreement,
        winner_surfaced_as_{a,b}, confidence_surfaced_as_{a,b},
        rationale_surfaced_as_{a,b}` -- `mm_verdicts_v1.jsonl`'s exact shape,
        one row per input row, same order. See module docstring for why a
        failed judge call aborts rather than silently skipping a row.
    """
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(pairs.itertuples(index=False), start=1):
        logger.info("judging %d/%d: %s", i, len(pairs), row.question_id)
        result = judge_pair(row.question, row.surfaced_photo_id, row.control_photo_id, store, template, chat_fn)
        rows.append(_verdict_row(row.question_id, row.type, result))
    return pd.DataFrame(rows)


def write_verdicts_jsonl(verdicts: pd.DataFrame, path: str | Path) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in verdicts.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# --------------------------------------------------------------------------
# Win-rate summary
# --------------------------------------------------------------------------


def _winrate_row(
    label: str,
    group: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator | None,
) -> dict[str, Any]:
    n_pairs = len(group)
    is_win = (group["outcome"] == "surfaced_win").astype(float).to_numpy()
    n_surfaced_win = int(is_win.sum())
    n_control_win = int((group["outcome"] == "control_win").sum())
    n_tie = int((group["outcome"] == "tie").sum())

    win_rate = float(is_win.mean())
    ci_lo, ci_hi = bootstrap_ci(is_win.tolist(), n_boot=n_boot, alpha=alpha, rng=rng)
    # One-sided: RQ4/H4 is a directional claim ("better than chance"), not
    # merely "different from chance" -- see module docstring.
    p_value = float(binomtest(n_surfaced_win, n_pairs, p=0.5, alternative="greater").pvalue)

    return {
        "group": label,
        "n_pairs": n_pairs,
        "n_surfaced_win": n_surfaced_win,
        "n_control_win": n_control_win,
        "n_tie": n_tie,
        "win_rate": win_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value_vs_0.5_greater": p_value,
        "tie_rate": n_tie / n_pairs,
        "order_agreement_rate": float(group["order_agreement"].mean()),
    }


def summarize_winrate(
    verdicts: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """The RQ4 headline table: pooled win-rate + a per-question-type breakdown.

    Args:
        verdicts: `run_pilot`'s output (or `mm_verdicts_v1.jsonl` reloaded).
            Must be non-empty.
        n_boot, alpha: forwarded to `cragb.eval.bootstrap.bootstrap_ci`.
        rng: seeded `numpy.random.Generator` for a reproducible CI.

    Returns:
        One row labelled `"overall"` (the pooled win-rate T6.5/RQ4 reports
        as the headline), then one row per `type` present in `verdicts`
        (H4 predicts photos help most on `colour_appearance`/`defects`
        (PLAN.md §2); per-type counts are small at this pilot's scale, so
        their CIs are wide -- report them, don't hide them). Columns:
        `group, n_pairs, n_surfaced_win, n_control_win, n_tie, win_rate,
        ci_lo, ci_hi, p_value_vs_0.5_greater, tie_rate,
        order_agreement_rate`.

    Raises:
        ValueError: if `verdicts` is empty.
    """
    if verdicts.empty:
        raise ValueError("verdicts must be non-empty")

    rows = [_winrate_row("overall", verdicts, n_boot=n_boot, alpha=alpha, rng=rng)]
    for type_, group in sorted(verdicts.groupby("type", sort=False), key=lambda kv: kv[0]):
        rows.append(_winrate_row(type_, group, n_boot=n_boot, alpha=alpha, rng=rng))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cost / latency summary
# --------------------------------------------------------------------------


def summarize_cost(
    calls: list[CompletionResult],
    model: str,
    pricing: dict[str, ModelPricing],
    wall_clock_s: float,
) -> pd.DataFrame:
    """One-row $/latency summary for every judge API call made this run.

    Args:
        calls: every `CompletionResult` from the run (2 per pair --
            `UsageRecorder.calls` after `run_pilot`).
        model: the judge model id, for the `pricing` lookup.
        pricing: `{model: ModelPricing}` (`cragb.eval.cost_model.load_pricing_config`).
        wall_clock_s: total measured wall-clock time for the run (a
            cache-bypassed or first-time live run -- see M5.md T5.5's same
            caution: a cached call reports ~0ms and would make this number
            fiction if the run were mostly cache hits).

    Returns:
        A single-row DataFrame: `model, n_calls, n_pairs,
        total_prompt_tokens, total_completion_tokens,
        total_thinking_tokens, mean_usd_per_call, total_usd, wall_clock_s,
        calls_per_second`. Cost is computed on
        `completion_tokens + thinking_tokens` per call -- see module
        docstring.

    Raises:
        ValueError: if `calls` is empty.
        KeyError: propagated from `cost_usd` if `model` has no pricing entry.
    """
    if not calls:
        raise ValueError("calls must be non-empty")

    costs = [
        cost_usd(c.prompt_tokens or 0, (c.completion_tokens or 0) + (c.thinking_tokens or 0), model, pricing)
        for c in calls
    ]
    return pd.DataFrame(
        [
            {
                "model": model,
                "n_calls": len(calls),
                "n_pairs": len(calls) // 2,
                "total_prompt_tokens": int(sum(c.prompt_tokens or 0 for c in calls)),
                "total_completion_tokens": int(sum(c.completion_tokens or 0 for c in calls)),
                "total_thinking_tokens": int(sum(c.thinking_tokens or 0 for c in calls)),
                "mean_usd_per_call": float(np.mean(costs)),
                "total_usd": float(np.sum(costs)),
                "wall_clock_s": wall_clock_s,
                "calls_per_second": (len(calls) / wall_clock_s) if wall_clock_s > 0 else float("nan"),
            }
        ]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vision_judge.yaml", help="Path to vision-judge config YAML.")
    parser.add_argument("--pairs-in", default="results/tables/mm_pairs_v1.jsonl")
    parser.add_argument("--pricing-config", default="configs/pricing.yaml")
    parser.add_argument("--verdicts-out", default="results/tables/mm_verdicts_v1.jsonl")
    parser.add_argument("--winrate-out", default="results/tables/mm_winrate_v1.csv")
    parser.add_argument("--cost-out", default="results/tables/mm_cost_v1.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
    pricing = load_pricing_config(args.pricing_config)

    pairs = load_pairs(args.pairs_in)
    logger.info("loaded %d usable pairs from %s", len(pairs), args.pairs_in)
    logger.info("about to spend %d live judge calls (2 per pair) before any result is final", 2 * len(pairs))

    recorder = UsageRecorder(usage_fn=client.complete_with_usage)
    with Timer() as t:
        verdicts = run_pilot(pairs, store, template, recorder)
    assert t.elapsed_s is not None

    write_verdicts_jsonl(verdicts, args.verdicts_out)
    logger.info("wrote %d verdict rows to %s", len(verdicts), args.verdicts_out)

    rng = np.random.default_rng(args.seed)
    winrate = summarize_winrate(verdicts, rng=rng)
    winrate_path = resolve_path(args.winrate_out)
    winrate_path.parent.mkdir(parents=True, exist_ok=True)
    winrate.to_csv(winrate_path, index=False)
    logger.info("wrote win-rate table (%d rows) to %s", len(winrate), args.winrate_out)

    cost = summarize_cost(recorder.calls, provider_cfg["model"], pricing, t.elapsed_s)
    cost_path = resolve_path(args.cost_out)
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost.to_csv(cost_path, index=False)
    logger.info("wrote cost table to %s", args.cost_out)

    overall = winrate.iloc[0]
    logger.info(
        "win-rate %.2f [%.2f, %.2f], n=%d, p(greater)=%.4f, ties=%.2f, order-agreement=%.2f",
        overall["win_rate"],
        overall["ci_lo"],
        overall["ci_hi"],
        overall["n_pairs"],
        overall["p_value_vs_0.5_greater"],
        overall["tie_rate"],
        overall["order_agreement_rate"],
    )
    if overall["ci_lo"] <= 0.5:
        logger.warning(
            "95%% CI includes 0.5 -- RQ4 is underpowered at this sample size (n=%d); "
            "report this as a null result with the T6.3 coverage funnel as the "
            "explanation, do not go fishing in the per-type breakdown for a "
            "subgroup that clears significance.",
            overall["n_pairs"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
