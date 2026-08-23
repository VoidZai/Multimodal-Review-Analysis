"""RQ0 (RAG vs closed-book) + RQ1 (small vs large) answer-quality tables (T4b.7;
PLAN.md §3 E5, §8, M4b.md T4b.7).

The last piece of E5: turn T4b.3's embedding-similarity metric and T4b.5's judge scores
into the two headline comparison tables PLAN.md §2 defines --

- **RQ0** ("does grounding help?"): `closed_book` vs `rag_small`.
- **RQ1** ("does model scale help, holding retrieval fixed?"): `rag_small` vs
  `rag_large`.

-- by reusing `cragb.eval.bootstrap` (`bootstrap_ci`, `paired_significance`)
**unchanged**, exactly the pattern T3.6 already validated for RQ2's Recall/Hit/nDCG/MRR
comparison. Pairing for the Wilcoxon test is by `question_id`, for the same reason
`bootstrap.py`'s own docstring gives: the same 60 CRAGB questions are scored under every
arm, so a paired test uses statistical information a two-sample test would throw away.

**Output shape is tidy/long, not wide** (one row per `(metric, arm)`, `METRICS` = 5
values, deliberately including `similarity` alongside the four judge criteria as just
another metric with the same row shape): a wide table (one row per metric, arm_a's and
arm_b's mean/CI as separate column groups) packs more into fewer rows but is harder to
filter, plot, or feed into a dashboard later (PLAN.md §11 point 1: "one results schema").
The `wilcoxon_p` column is identical on both of a metric's two rows -- it characterizes
the `arm_a`-vs-`arm_b` pair, not a single arm, but repeating it keeps every row
self-contained rather than requiring a join to recover which comparison it belongs to.

**Windows/venv note (PLAN.md §14.1):** computing similarity needs
`cragb.eval.metrics_answer.load_model`, which needs `sentence-transformers`/`torch` --
run `main()` via the short-path venv:

    C:\\venv\\cragb\\Scripts\\python.exe -m cragb.eval.run_answer_quality_eval

Everything else in this module (`build_combined_scores`, `build_comparison_table`,
`validate_comparison_table`) is generic over any `EmbeddingModel`-shaped object (same as
`metrics_answer` itself), so it imports and is fully unit-testable in the main
environment with no venv needed -- only `main()`'s call to `load_model()` requires it.

Usage:
    C:\\venv\\cragb\\Scripts\\python.exe -m cragb.eval.run_answer_quality_eval
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cragb.bench.reference_answers import ReferenceAnswer, load_reference_answers
from cragb.eval.bootstrap import bootstrap_ci, paired_significance
from cragb.eval.metrics_answer import EmbeddingModel, load_model
from cragb.eval.metrics_answer import score_arm as score_similarity
from cragb.eval.run_answer_generation import ARM_DEFAULT_OUT
from cragb.eval.run_grounded_qa_pilot import write_csv
from cragb.eval.run_judge_eval import load_arm_transcripts
from cragb.utils.io import resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

METRICS: tuple[str, ...] = ("similarity", "correctness", "faithfulness", "completeness", "conciseness")

RQ0_ARMS: tuple[str, str] = ("closed_book", "rag_small")
RQ1_ARMS: tuple[str, str] = ("rag_small", "rag_large")

DEFAULT_JUDGE_SCORES_IN = "results/tables/judge_scores_v1.csv"
DEFAULT_REFERENCES_IN = "benchmark/reference_answers_v1.jsonl"
DEFAULT_RQ0_OUT = "results/tables/rq0_answer_quality_v1.csv"
DEFAULT_RQ1_OUT = "results/tables/rq1_answer_quality_v1.csv"


def build_combined_scores(
    arms: tuple[str, ...],
    references: dict[str, ReferenceAnswer],
    judge_scores: pd.DataFrame,
    model: EmbeddingModel,
) -> pd.DataFrame:
    """Merge T4b.3's similarity scores and T4b.5's judge scores into one table.

    Args:
        arms: which arms to load transcripts for and score similarity over -- compute
            this once for the union of every arm a caller needs (e.g. all three, if
            building both the RQ0 and RQ1 tables) rather than once per comparison, so
            an arm shared by both (`rag_small`) is never embedded twice.
        references: from `cragb.bench.reference_answers.load_reference_answers`.
        judge_scores: T4b.5's full table (`results/tables/judge_scores_v1.csv`).
        model: a loaded embedding model (`cragb.eval.metrics_answer.load_model`), or
            any `EmbeddingModel`-shaped stand-in.

    Returns:
        One row per `(arm, question_id)` present in every requested arm: `arm`,
        `question_id`, `similarity`, `correctness`, `faithfulness`, `completeness`,
        `conciseness`.

    Raises:
        ValueError: if the similarity scores and the judge scores don't merge
            one-to-one for `arms` (e.g. a question scored by one but not the other) --
            an inner join would silently drop the mismatched rows instead of failing,
            quietly biasing every downstream mean toward whichever questions happened
            to survive.
    """
    similarity_frames = []
    for arm in arms:
        transcripts = load_arm_transcripts(arm, ARM_DEFAULT_OUT[arm])
        sim = score_similarity(transcripts, references, model)
        sim.insert(0, "arm", arm)
        similarity_frames.append(sim)
    similarity_scores = pd.concat(similarity_frames, ignore_index=True)

    arm_judge_scores = judge_scores[judge_scores["arm"].isin(arms)]
    merged = similarity_scores.merge(arm_judge_scores, on=["arm", "question_id"], how="inner")

    if len(merged) != len(similarity_scores) or len(merged) != len(arm_judge_scores):
        raise ValueError(
            f"Similarity ({len(similarity_scores)} row(s)) and judge "
            f"({len(arm_judge_scores)} row(s)) scores did not merge one-to-one for "
            f"arms {arms} -- got {len(merged)} matched row(s). Check both were "
            "computed over the same question set."
        )

    return merged[["arm", "question_id", *METRICS]]


def build_comparison_table(
    combined_scores: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    n_boot: int = 10000,
    seed: int | None = None,
) -> pd.DataFrame:
    """The RQ0/RQ1 headline table: per-metric, per-arm mean + bootstrap CI, plus a
    paired Wilcoxon p-value between `arm_a` and `arm_b`.

    Args:
        combined_scores: from `build_combined_scores`, covering at least `arm_a` and
            `arm_b`.
        arm_a, arm_b: the two arms to compare, e.g. RQ0's `("closed_book", "rag_small")`.
        n_boot: bootstrap resamples per `cragb.eval.bootstrap.bootstrap_ci`.
        seed: seeds the bootstrap resampling for reproducible CIs; `None` (default)
            draws a fresh, non-reproducible sample each call, matching `bootstrap_ci`'s
            own default.

    Returns:
        A tidy long table, one row per `(metric, arm)`: `metric`, `arm`, `n`, `mean`,
        `ci_lo`, `ci_hi`, `wilcoxon_p` (module docstring: identical on both of a
        metric's two rows). Pairing for the Wilcoxon test is by `question_id`.

    Raises:
        ValueError: if `arm_a` and `arm_b` do not cover exactly the same set of
            `question_id`s in `combined_scores` -- the Wilcoxon test requires an
            aligned pairing, and proceeding on a mismatched one would compute a
            meaningless p-value rather than failing.
    """
    a = combined_scores[combined_scores["arm"] == arm_a].set_index("question_id").sort_index()
    b = combined_scores[combined_scores["arm"] == arm_b].set_index("question_id").sort_index()
    if set(a.index) != set(b.index):
        raise ValueError(
            f"{arm_a!r} ({len(a)} question(s)) and {arm_b!r} ({len(b)} question(s)) "
            "do not cover the same question_id set -- cannot pair them for the "
            "Wilcoxon test."
        )
    b = b.loc[a.index]  # explicit re-alignment to a's order, belt-and-braces on the pairing

    rng = np.random.default_rng(seed) if seed is not None else None

    rows = []
    for metric in METRICS:
        p_value = paired_significance(a[metric].tolist(), b[metric].tolist())
        for arm_name, series in ((arm_a, a[metric]), (arm_b, b[metric])):
            lo, hi = bootstrap_ci(series.tolist(), n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "metric": metric,
                    "arm": arm_name,
                    "n": len(series),
                    "mean": float(series.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def validate_comparison_table(table: pd.DataFrame) -> None:
    """Fail loudly if any CI or p-value in `table` is structurally invalid.

    Args:
        table: from `build_comparison_table`.

    Raises:
        ValueError: if any row's `mean` falls outside its own `[ci_lo, ci_hi]` (a
            percentile bootstrap CI must contain the sample mean by construction --
            never possible unless something upstream is broken), or if any
            `wilcoxon_p` is outside `[0, 1]`.
    """
    bad_ci = table[(table["ci_lo"] > table["mean"]) | (table["mean"] > table["ci_hi"])]
    if not bad_ci.empty:
        raise ValueError(f"mean falls outside its own bootstrap CI: {bad_ci.to_dict('records')}")

    bad_p = table[(table["wilcoxon_p"] < 0) | (table["wilcoxon_p"] > 1)]
    if not bad_p.empty:
        raise ValueError(f"wilcoxon_p out of [0, 1]: {bad_p.to_dict('records')}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-scores-in", default=DEFAULT_JUDGE_SCORES_IN)
    parser.add_argument("--references-in", default=DEFAULT_REFERENCES_IN)
    parser.add_argument("--rq0-out", default=DEFAULT_RQ0_OUT)
    parser.add_argument("--rq1-out", default=DEFAULT_RQ1_OUT)
    parser.add_argument("--seed", type=int, default=42, help="Seeds bootstrap resampling for reproducible CIs.")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples per metric/arm.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    set_global_seed(args.seed)

    references = load_reference_answers(args.references_in)
    judge_scores = pd.read_csv(resolve_path(args.judge_scores_in))
    model = load_model()

    all_arms = tuple(sorted(set(RQ0_ARMS) | set(RQ1_ARMS)))
    combined = build_combined_scores(all_arms, references, judge_scores, model)

    rq0 = build_comparison_table(combined, *RQ0_ARMS, n_boot=args.n_boot, seed=args.seed)
    validate_comparison_table(rq0)
    rq0_path = write_csv(rq0, args.rq0_out)
    logger.info("RQ0 (%s vs %s): wrote %d row(s) to %s", *RQ0_ARMS, len(rq0), rq0_path)

    rq1 = build_comparison_table(combined, *RQ1_ARMS, n_boot=args.n_boot, seed=args.seed)
    validate_comparison_table(rq1)
    rq1_path = write_csv(rq1, args.rq1_out)
    logger.info("RQ1 (%s vs %s): wrote %d row(s) to %s", *RQ1_ARMS, len(rq1), rq1_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
