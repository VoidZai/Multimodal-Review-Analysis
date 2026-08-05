"""Bootstrap confidence intervals + paired significance testing (T3.3; PLAN.md §8, §3 E3/E5).

PLAN.md §8's cross-cutting rigor requirement is that every headline
number in this project — Recall@k, nDCG@k, judge scores, cost — is
reported with a **bootstrap 95% CI**, and every RQ2/RQ0/RQ1 comparison
between two configurations (e.g. BM25 vs dense, T3.6/T3.7) is backed by
a **paired significance test over the shared question set**, since the
same 60 CRAGB questions are scored under both configurations — that
pairing is statistical information a two-sample test would throw away.

This module provides both primitives, generically over any list of
per-question scores; it has no knowledge of retrieval, judges, or cost
specifically, so `T3.6`+ (retrieval), the future answer-quality eval
(E5), and the cost/latency harness (E6) can all reuse it unchanged.

Reproducibility: both functions here draw randomness (`bootstrap_ci`
resamples; `paired_significance` does not, but is included alongside for
a single import site). Following `cragb.utils.seeds`' convention of
passing an explicit `numpy.random.Generator` rather than seeding global
state, `bootstrap_ci` takes an optional `rng` — pass
`np.random.default_rng(seed)` (e.g. via `cragb.utils.seeds.set_global_seed`)
for a fully reproducible CI; omit it for a fresh, non-reproducible draw
each call.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    scores: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `scores`.

    Resamples `scores` with replacement `n_boot` times, computes the mean
    of each resample, and returns the `[alpha/2, 1 - alpha/2]` percentiles
    of that distribution of means — the standard percentile-bootstrap CI
    (PLAN.md §8's "paired bootstrap 95% CIs" at the single-configuration
    level; see `paired_significance` for comparing two configurations).

    Args:
        scores: per-question (or per-item) scores to bootstrap, e.g. one
            Recall@5 value per CRAGB question for a given retriever.
            Must be non-empty.
        n_boot: number of bootstrap resamples. Higher is more precise but
            slower; 10000 is enough for a stable 95% CI at this project's
            scale (≤60 questions per config).
        alpha: significance level; `alpha=0.05` gives a 95% CI.
        rng: a seeded `numpy.random.Generator` for reproducible output,
            or `None` (default) to draw a fresh, non-reproducible one.

    Returns:
        `(lo, hi)` — the CI bounds. If every score in `scores` is
        identical (e.g. a single-element input, or a constant metric),
        `lo == hi == that value`: every resample mean is necessarily that
        same value, so the CI correctly collapses to a point rather than
        reporting spurious uncertainty.

    Raises:
        ValueError: if `scores` is empty, `n_boot` is not positive, or
            `alpha` is not in `(0, 1)`.
    """
    if not scores:
        raise ValueError("scores must be non-empty.")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    generator = rng if rng is not None else np.random.default_rng()
    values = np.asarray(scores, dtype=float)
    n = len(values)

    resample_indices = generator.integers(0, n, size=(n_boot, n))
    resample_means = values[resample_indices].mean(axis=1)

    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_significance(scores_a: list[float], scores_b: list[float]) -> float:
    """Wilcoxon signed-rank test p-value for paired `scores_a` vs `scores_b`.

    The paired unit is a question: `scores_a[i]` and `scores_b[i]` must
    be the same question's score under two different configurations
    (e.g. BM25's and dense's Recall@5 on CRAGB question `i`) — exactly
    what PLAN.md §8 specifies ("paired bootstrap / Wilcoxon signed-rank
    across the shared question set — questions are the paired unit").
    Wilcoxon signed-rank, not a paired t-test, because retrieval/judge
    scores are bounded and often non-normally distributed (e.g. Recall@k
    piles up at 0.0 and 1.0), which the Wilcoxon test does not assume
    away.

    Args:
        scores_a: per-question scores under configuration A.
        scores_b: per-question scores under configuration B, same length
            and question order as `scores_a`.

    Returns:
        The two-sided p-value. If `scores_a` and `scores_b` are
        (numerically) identical for every question, there is no evidence
        of a difference and `1.0` is returned directly — `scipy`'s
        Wilcoxon test raises `ValueError` when every paired difference is
        exactly zero (it has no ranks to compute), which is a real
        possibility here (e.g. two retrievers tie on every question at
        `k=1`) and not something callers should have to special-case.

    Raises:
        ValueError: if `scores_a` and `scores_b` have different lengths,
            or either is empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must be the same length (paired by "
            f"question), got {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) == 0:
        raise ValueError("scores_a/scores_b must be non-empty.")

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if np.array_equal(a, b):
        return 1.0

    _statistic, p_value = stats.wilcoxon(a, b)
    return float(p_value)
