"""Joined quality-vs-cost table + Pareto scatter (T5.6; PLAN.md §3 E6, §7 figure #4,
§11 upgrade #5, M5.md T5.6).

The decision artefact M5 exists to produce: one table where each answer-generation
configuration (closed-book, RAG-small, RAG-large) carries both its judge-scored quality
(M4b) and its measured cost (T5.4's $/query, T5.5's end-to-end latency), plus the scatter
that makes the trade-off visible.

**Why the answer-generation arms and the BM25/dense retrievers are two separate sections
of one file, not one Pareto frontier.** They aren't comparable on the same axes: an
answer-generation arm has a judge-scored quality (1-5 scale) and a real $/query (a paid
API call); a retriever has Recall@k (0-1 scale) and a per-query search latency, but no
API cost at all (BM25/dense both run locally). Forcing both onto one `quality_mean`/
`usd_per_query` frontier would either drop the retrievers' real quality axis (Recall@k)
or fabricate a $/query for a free local computation. Instead `join_quality_cost` (for the
three generation arms, against $/query) and `retrieval_quality_cost_rows` (for BM25/dense,
against per-query latency) each compute their own Pareto frontier on their own natural
axes, and `main()` concatenates both into `results/tables/cost_quality_v1.csv` under a
`component` column ("answer_generation" / "retrieval") — so the retrieval configs are
still in the report (PLAN.md §7's cost table asks for both retrieval and generation cost),
just never silently pooled into a comparison that wouldn't mean anything.

**Why the Pareto figure (`quality_vs_cost_pareto_v1.png`) covers only the generation
arms.** PLAN.md §7 figure #4 and §11 upgrade #5 are both specifically about "is the big
model worth it" — a generation-arm question. The expected, and reported-honestly, result:
RQ1's 20B-vs-120B correctness gap (`results/tables/rq1_answer_quality_v1.csv`: 3.67 vs
3.77, Wilcoxon p=0.34) is not statistically distinguishable, while RAG-large costs
roughly double RAG-small per query (`results/tables/answer_cost_v1.csv`). The figure must
show that honestly — overlapping CIs on the y-axis at a large x-axis cost jump — not
imply a win the statistics don't support.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_METRIC = "correctness"
DEFAULT_K = 5


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------


def pareto_frontier(df: pd.DataFrame, quality_col: str, cost_col: str) -> pd.Series:
    """Mark rows on the Pareto frontier: maximize `quality_col`, minimize `cost_col`.

    A row is on the frontier iff no *other* row weakly dominates it (equal or
    better on both axes, and strictly better on at least one) — the standard
    non-dominated-set definition. Exact ties (identical quality and cost) are
    all kept on the frontier together, since neither dominates the other.

    Args:
        df: rows to evaluate. `O(n^2)` — fine at this project's scale
            (a handful of configurations per call), not intended for
            large `df`.
        quality_col: column where higher is better.
        cost_col: column where lower is better.

    Returns:
        A boolean Series aligned to `df`'s index (`True` = on the frontier).

    Raises:
        ValueError: if `df` is empty.
    """
    if df.empty:
        raise ValueError("df must be non-empty")

    quality = df[quality_col].to_numpy()
    cost = df[cost_col].to_numpy()
    n = len(df)
    on_frontier = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weakly_better = quality[j] >= quality[i] and cost[j] <= cost[i]
            strictly_better = quality[j] > quality[i] or cost[j] < cost[i]
            if weakly_better and strictly_better:
                on_frontier[i] = False
                break

    return pd.Series(on_frontier, index=df.index, name="on_frontier")


# --------------------------------------------------------------------------
# Answer-generation arms: quality + $/query + latency
# --------------------------------------------------------------------------


def prepare_quality_df(rq0_table: pd.DataFrame, rq1_table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per arm's `metric` score, deduplicated across the RQ0/RQ1 tables.

    `closed_book` only appears in `rq0_table`; `rag_large` only in
    `rq1_table`; `rag_small` appears in *both* (RQ0 compares it against
    closed-book, RQ1 against rag-large), with slightly different bootstrap
    CIs each time (independent resamples, same underlying per-question
    scores). `rq1_table`'s version is kept — it's the table that directly
    contextualizes rag_small against rag_large, which is what this
    milestone's headline comparison (§7 figure #4) is about.

    Args:
        rq0_table: `results/tables/rq0_answer_quality_v1.csv`'s shape
            (`metric`, `arm`, `n`, `mean`, `ci_lo`, `ci_hi`, `wilcoxon_p`).
        rq1_table: `results/tables/rq1_answer_quality_v1.csv`'s shape (same columns).
        metric: which row of the long-format `metric` column to keep
            (e.g. `"correctness"`).

    Returns:
        `[arm, n, mean, ci_lo, ci_hi]`, one row per arm.

    Raises:
        ValueError: if `metric` has no rows in either table.
    """
    rq0_metric = rq0_table[rq0_table["metric"] == metric]
    rq1_metric = rq1_table[rq1_table["metric"] == metric]
    if rq0_metric.empty and rq1_metric.empty:
        raise ValueError(f"metric {metric!r} not found in rq0_table or rq1_table")

    combined = pd.concat([rq0_metric, rq1_metric], ignore_index=True)
    deduped = combined.drop_duplicates(subset="arm", keep="last")
    return deduped[["arm", "n", "mean", "ci_lo", "ci_hi"]].reset_index(drop=True)


def join_quality_cost(
    quality_df: pd.DataFrame, cost_df: pd.DataFrame, latency_df: pd.DataFrame
) -> pd.DataFrame:
    """Join per-arm quality, $/query, and end-to-end latency into one row per arm.

    Args:
        quality_df: one row per arm — `arm`, `n`, `mean`, `ci_lo`, `ci_hi`
            (e.g. `prepare_quality_df`'s output, already filtered to one metric).
        cost_df: `results/tables/answer_cost_v1.csv`'s shape (must have
            `arm`, `model`, `mean_usd_per_query`, `usd_per_query_ci_lo`,
            `usd_per_query_ci_hi`).
        latency_df: `results/tables/e2e_latency_v1.csv`'s shape (must have
            `arm`, `e2e_ms_p50`, `e2e_ms_p95`).

    Returns:
        One row per arm in `quality_df`: `arm, model, n, quality_mean,
        quality_ci_lo, quality_ci_hi, usd_per_query, usd_per_query_ci_lo,
        usd_per_query_ci_hi, e2e_ms_p50, e2e_ms_p95`.

    Raises:
        ValueError: if any arm in `quality_df` is missing from `cost_df` or
            `latency_df` (a silent inner-join row loss), or if `cost_df`/
            `latency_df` carry duplicate arms (which would silently fan out
            the join into more rows than `quality_df` has).
    """
    missing_cost = set(quality_df["arm"]) - set(cost_df["arm"])
    if missing_cost:
        raise ValueError(f"arm(s) {sorted(missing_cost)} in quality_df missing from cost_df")
    missing_latency = set(quality_df["arm"]) - set(latency_df["arm"])
    if missing_latency:
        raise ValueError(f"arm(s) {sorted(missing_latency)} in quality_df missing from latency_df")
    if cost_df["arm"].duplicated().any():
        raise ValueError("cost_df has duplicate arms; join would fan out")
    if latency_df["arm"].duplicated().any():
        raise ValueError("latency_df has duplicate arms; join would fan out")

    quality = quality_df.rename(
        columns={"mean": "quality_mean", "ci_lo": "quality_ci_lo", "ci_hi": "quality_ci_hi"}
    )
    merged = quality.merge(
        cost_df[["arm", "model", "mean_usd_per_query", "usd_per_query_ci_lo", "usd_per_query_ci_hi"]],
        on="arm",
        how="left",
    ).merge(
        latency_df[["arm", "e2e_ms_p50", "e2e_ms_p95"]],
        on="arm",
        how="left",
    )
    merged = merged.rename(columns={"mean_usd_per_query": "usd_per_query"})

    if len(merged) != len(quality_df):
        raise ValueError("join changed row count; expected exactly one row per arm in quality_df")

    return merged[
        [
            "arm",
            "model",
            "n",
            "quality_mean",
            "quality_ci_lo",
            "quality_ci_hi",
            "usd_per_query",
            "usd_per_query_ci_lo",
            "usd_per_query_ci_hi",
            "e2e_ms_p50",
            "e2e_ms_p95",
        ]
    ]


# --------------------------------------------------------------------------
# Retrieval companion rows: Recall@k + search latency
# --------------------------------------------------------------------------


def retrieval_quality_cost_rows(
    retrieval_eval_df: pd.DataFrame, retrieval_cost_latency_df: pd.DataFrame, k: int
) -> pd.DataFrame:
    """BM25/dense companion rows for `cost_quality_v1.csv`'s `retrieval` component.

    Retrieval has no per-query $ cost (BM25/dense both run locally, no API
    call involved) and no judge-scored "quality" — its natural quality axis
    is Recall@k (already bootstrap-CI'd by T3.6's `retrieval_eval_v1.csv`)
    and its natural cost axis is per-query search latency (T5.3's
    `retrieval_cost_latency_v1.csv`), not $/query. See module docstring for
    why these rows get their own Pareto frontier rather than joining the
    generation arms' frontier.

    Args:
        retrieval_eval_df: `results/tables/retrieval_eval_v1.csv`'s shape.
        retrieval_cost_latency_df: `results/tables/retrieval_cost_latency_v1.csv`'s shape.
        k: which `k` row to read from `retrieval_eval_df` — T3.6/T5.3's shared headline k.

    Returns:
        One row per retriever: `retriever, n, quality_mean, quality_ci_lo,
        quality_ci_hi, latency_ms_p50, on_frontier` (Recall@k maximized,
        latency minimized).

    Raises:
        ValueError: if `k` has no rows in `retrieval_eval_df`, a retriever
            at that `k` is missing from `retrieval_cost_latency_df`, or
            either input has duplicate retrievers at the relevant rows.
    """
    eval_at_k = retrieval_eval_df[retrieval_eval_df["k"] == k]
    if eval_at_k.empty:
        raise ValueError(f"no rows in retrieval_eval_df at k={k}")
    if eval_at_k["retriever"].duplicated().any():
        raise ValueError(f"retrieval_eval_df has duplicate retriever rows at k={k}")
    if retrieval_cost_latency_df["retriever"].duplicated().any():
        raise ValueError("retrieval_cost_latency_df has duplicate retriever rows")

    missing = set(eval_at_k["retriever"]) - set(retrieval_cost_latency_df["retriever"])
    if missing:
        raise ValueError(f"retriever(s) {sorted(missing)} missing from retrieval_cost_latency_df")

    merged = eval_at_k.merge(
        retrieval_cost_latency_df[["retriever", "latency_p50_ms"]], on="retriever", how="left"
    )
    result = pd.DataFrame(
        {
            "retriever": merged["retriever"],
            "n": merged["n_questions"],
            "quality_mean": merged["recall_mean"],
            "quality_ci_lo": merged["recall_ci_lo"],
            "quality_ci_hi": merged["recall_ci_hi"],
            "latency_ms_p50": merged["latency_p50_ms"],
        }
    ).reset_index(drop=True)
    result["on_frontier"] = pareto_frontier(result, "quality_mean", "latency_ms_p50")
    return result


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def plot_quality_vs_cost(joined: pd.DataFrame, out_path: str | Path, metric: str) -> None:
    """Quality-vs-cost Pareto scatter for the answer-generation arms (report figure #4).

    x = $/query (log scale — the arms differ by roughly an order of
    magnitude), y = `quality_mean` with error bars from its bootstrap CI,
    frontier points connected. Deliberately does *not* smooth over
    overlapping CIs — see module docstring.

    Args:
        joined: `join_quality_cost`'s output, with an added `on_frontier`
            boolean column (`pareto_frontier(joined, "quality_mean", "usd_per_query")`).
        out_path: where to save the figure (parent directories created as needed).
        metric: the judge metric plotted on the y-axis (for the axis label/title only).

    Raises:
        ValueError: if `joined` is empty.
    """
    if joined.empty:
        raise ValueError("joined must be non-empty")

    import matplotlib.pyplot as plt  # lazy: plotting is optional for callers that only need the table

    ordered = joined.sort_values("usd_per_query")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        ordered["usd_per_query"],
        ordered["quality_mean"],
        yerr=[
            ordered["quality_mean"] - ordered["quality_ci_lo"],
            ordered["quality_ci_hi"] - ordered["quality_mean"],
        ],
        xerr=[
            ordered["usd_per_query"] - ordered["usd_per_query_ci_lo"],
            ordered["usd_per_query_ci_hi"] - ordered["usd_per_query"],
        ],
        fmt="none",
        ecolor="0.6",
        capsize=3,
        zorder=1,
    )
    colors = {True: "tab:blue", False: "0.5"}
    for on_frontier, group in ordered.groupby("on_frontier"):
        ax.scatter(
            group["usd_per_query"],
            group["quality_mean"],
            s=70,
            zorder=2,
            color=colors[bool(on_frontier)],
            label="Pareto frontier" if on_frontier else "dominated",
        )
    frontier = ordered[ordered["on_frontier"]]
    if len(frontier) > 1:
        ax.plot(frontier["usd_per_query"], frontier["quality_mean"], "--", color="tab:blue", alpha=0.5, zorder=1)

    for _, row in ordered.iterrows():
        ax.annotate(
            row["arm"], (row["usd_per_query"], row["quality_mean"]), textcoords="offset points", xytext=(6, 6)
        )

    ax.set_xscale("log")
    ax.set_xlabel("USD / query (log scale)")
    ax.set_ylabel(f"Judge {metric} (1-5, mean ± 95% bootstrap CI)")
    ax.set_title("Quality vs. cost: closed-book / RAG-small / RAG-large (CRAGB v1)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    resolved_out_path = resolve_path(out_path)
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Join per-arm answer-quality, $/query, and end-to-end latency into "
        "one cost/quality table, add BM25/dense retrieval companion rows, and render the "
        "quality-vs-cost Pareto scatter (T5.6; PLAN.md §3 E6, §7 figure #4)."
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="judge metric to plot (default: correctness)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="retrieval k to read (default: 5)")
    parser.add_argument("--rq0", default="results/tables/rq0_answer_quality_v1.csv")
    parser.add_argument("--rq1", default="results/tables/rq1_answer_quality_v1.csv")
    parser.add_argument("--answer-cost", default="results/tables/answer_cost_v1.csv")
    parser.add_argument("--e2e-latency", default="results/tables/e2e_latency_v1.csv")
    parser.add_argument("--retrieval-eval", default="results/tables/retrieval_eval_v1.csv")
    parser.add_argument("--retrieval-cost-latency", default="results/tables/retrieval_cost_latency_v1.csv")
    parser.add_argument("--out", default="results/tables/cost_quality_v1.csv")
    parser.add_argument("--figure-out", default="reports/figures/quality_vs_cost_pareto_v1.png")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rq0 = pd.read_csv(resolve_path(args.rq0))
    rq1 = pd.read_csv(resolve_path(args.rq1))
    quality_df = prepare_quality_df(rq0, rq1, args.metric)

    cost_df = pd.read_csv(resolve_path(args.answer_cost))
    latency_df = pd.read_csv(resolve_path(args.e2e_latency))
    joined = join_quality_cost(quality_df, cost_df, latency_df)
    joined["on_frontier"] = pareto_frontier(joined, "quality_mean", "usd_per_query")
    joined.insert(0, "component", "answer_generation")
    joined.insert(1, "config", joined["arm"])
    logger.info(
        "answer_generation: %d rows, frontier=%s", len(joined), joined.loc[joined["on_frontier"], "arm"].tolist()
    )

    retrieval_eval = pd.read_csv(resolve_path(args.retrieval_eval))
    retrieval_cost_latency = pd.read_csv(resolve_path(args.retrieval_cost_latency))
    retrieval_rows = retrieval_quality_cost_rows(retrieval_eval, retrieval_cost_latency, args.k)
    retrieval_rows.insert(0, "component", "retrieval")
    retrieval_rows.insert(1, "config", retrieval_rows["retriever"])
    logger.info(
        "retrieval (k=%d): %d rows, frontier=%s",
        args.k,
        len(retrieval_rows),
        retrieval_rows.loc[retrieval_rows["on_frontier"], "retriever"].tolist(),
    )

    combined = pd.concat([joined, retrieval_rows], ignore_index=True)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("wrote %d rows to %s", len(combined), out_path)

    plot_quality_vs_cost(joined, args.figure_out, args.metric)
    logger.info("wrote figure to %s", resolve_path(args.figure_out))


if __name__ == "__main__":
    main()
