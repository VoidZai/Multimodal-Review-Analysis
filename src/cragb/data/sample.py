"""Stratified sampling with a fixed seed (T1.7; PLAN.md §3 E0, Risk C).

Risk C in PLAN.md is specific: plain random sampling over a corpus that is
63.65% five-star would starve 1-2 star ("defect/fit") and image-bearing
rows, so retrieval questions about problems or visual evidence would have
little to no supporting evidence in the index — a failure that would look
like "the retriever is bad" when actually "the corpus never had the
answer." This module forces minimum representation for those strata
instead of leaving it to chance.

Design:
- Two independent stratification dimensions are read from
  `configs/data.yaml` (`sampling.strata.rating`, `sampling.strata.has_image`),
  each a set of target *marginal* proportions (must sum to ~1.0).
- A joint target count per (rating_bucket, has_image) cell is computed by
  **assuming independence** between the two dimensions (joint weight =
  product of marginals) and allocating `target_size` across cells with
  the largest-remainder method, so integer allocations sum to exactly
  `target_size` rather than drifting from rounding.
- Sampling within each cell is done via a single seeded
  `numpy.random.Generator`, so the whole draw is reproducible end to end.
- If a cell doesn't have enough rows to meet its target (a real
  possibility for the 1-2-star + has_image intersection, the rarest
  combination), the shortfall is **capped and reported**, never silently
  padded or ignored — `SamplingReport.shortfall_total` makes an
  under-filled corpus visible instead of a quiet surprise at EDA time.

Sub-category is deliberately *not* forced to target weights here: PLAN.md
asks for its coverage to be checked (T1.10's EDA figure), not pinned to a
quota, since there's no principled target distribution for it the way
"defects need 1-2 star reviews" gives one for rating. Forcing arbitrary
sub-category quotas would be precision without justification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Amazon Reviews 2023 ratings are floats in {1.0, 2.0, 3.0, 4.0, 5.0}.
_RATING_TO_BUCKET: dict[float, str] = {
    1.0: "1-2",
    2.0: "1-2",
    3.0: "3",
    4.0: "4-5",
    5.0: "4-5",
}


def bucket_rating(rating: object) -> str | None:
    """Map a raw star rating to its stratification bucket label.

    Args:
        rating: a review's `rating` field. Expected to be a float in
            {1.0..5.0}; anything else (NaN, out-of-range, wrong type)
            returns `None` rather than raising, since a handful of dirty
            rating values shouldn't crash a sampling pass over hundreds of
            thousands of rows.

    Returns:
        One of `"1-2"`, `"3"`, `"4-5"`, or `None` if unmapped.
    """
    try:
        return _RATING_TO_BUCKET.get(float(rating))
    except (TypeError, ValueError):
        return None


def add_rating_bucket(df: pd.DataFrame, rating_col: str = "rating", out_col: str = "rating_bucket") -> pd.DataFrame:
    """Attach a `rating_bucket` column computed via `bucket_rating`.

    Args:
        df: reviews frame containing `rating_col`.
        rating_col: source ratings column.
        out_col: name for the new bucket-label column.

    Returns:
        A new DataFrame (input not mutated) with `out_col` added.
    """
    out = df.copy()
    out[out_col] = out[rating_col].map(bucket_rating)
    return out


def _validate_marginal_weights(name: str, weights: dict[Any, float]) -> None:
    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"strata weights for {name!r} must sum to ~1.0, got {total!r}: {weights!r}")


def compute_joint_targets(
    strata_weights: dict[str, dict[Any, float]],
    target_size: int,
) -> dict[tuple[Any, ...], int]:
    """Allocate `target_size` rows across the cartesian product of strata.

    Joint weight per cell is the product of each dimension's marginal
    weight (an explicit independence assumption — see module docstring).
    Integer allocation uses the largest-remainder method (Hamilton's
    method) so allocations sum to exactly `target_size` instead of
    drifting from naive per-cell rounding.

    Args:
        strata_weights: mapping of dimension name -> {bucket_label:
            marginal_proportion}. Each dimension's weights must sum to
            ~1.0.
        target_size: total rows to allocate across all joint cells.

    Returns:
        Mapping of joint key (a tuple of one bucket label per dimension,
        in the order `strata_weights` was given) to an integer target
        row count. Values sum to exactly `target_size`.

    Raises:
        ValueError: if any dimension's weights don't sum to ~1.0.
    """
    for name, weights in strata_weights.items():
        _validate_marginal_weights(name, weights)

    dim_names = list(strata_weights.keys())
    cells: list[tuple[tuple[Any, ...], float]] = [((), 1.0)]
    for name in dim_names:
        next_cells = []
        for prefix, weight_so_far in cells:
            for label, weight in strata_weights[name].items():
                next_cells.append((prefix + (label,), weight_so_far * weight))
        cells = next_cells

    # Largest-remainder rounding: floor each allocation, then hand out the
    # leftover units (target_size - sum of floors) to the cells with the
    # largest fractional remainder, so the total matches exactly.
    raw = [(key, weight * target_size) for key, weight in cells]
    floors = {key: int(value) for key, value in raw}
    remainder = target_size - sum(floors.values())
    remainders_sorted = sorted(raw, key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
    targets = dict(floors)
    for key, _ in remainders_sorted[:remainder]:
        targets[key] += 1

    assert sum(targets.values()) == target_size
    return targets


@dataclass(frozen=True)
class StratumOutcome:
    target: int
    available: int
    sampled: int


@dataclass(frozen=True)
class SamplingReport:
    """Per-stratum and overall outcome of a `stratified_sample` call."""

    target_size: int
    actual_size: int
    per_stratum: dict[tuple[Any, ...], StratumOutcome] = field(default_factory=dict)

    @property
    def shortfall_total(self) -> int:
        return self.target_size - self.actual_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_size": self.target_size,
            "actual_size": self.actual_size,
            "shortfall_total": self.shortfall_total,
            "per_stratum": {
                "|".join(map(str, key)): vars(outcome) for key, outcome in self.per_stratum.items()
            },
        }


def stratified_sample(
    df: pd.DataFrame,
    strata_weights: dict[str, dict[Any, float]],
    target_size: int,
    seed: int,
    strata_cols: tuple[str, ...] = ("rating_bucket", "has_image"),
) -> tuple[pd.DataFrame, SamplingReport]:
    """Draw a reproducible stratified sample of `target_size` rows.

    Args:
        df: input frame; must already contain every column in
            `strata_cols` (e.g. run `add_rating_bucket` and
            `cragb.data.features.add_review_features` first).
        strata_weights: as in `compute_joint_targets`; keys must match
            `strata_cols` in the same order.
        target_size: desired total sample size.
        seed: RNG seed — the same `df`, weights, and seed always produce
            the same sample.
        strata_cols: columns in `df` to stratify on, matching the
            dimension order of `strata_weights`.

    Returns:
        `(sampled_df, report)`. `sampled_df` has at most `target_size`
        rows (fewer only if one or more strata are underpopulated — see
        `report.shortfall_total`), in shuffled (non-block) order. Row
        `index` values are preserved from `df` so callers can trace a
        sampled row back to its source.

    Raises:
        ValueError: if `strata_weights` keys don't match `strata_cols`.
    """
    if tuple(strata_weights.keys()) != strata_cols:
        raise ValueError(
            f"strata_weights dimensions {tuple(strata_weights.keys())!r} must match "
            f"strata_cols {strata_cols!r} (same keys, same order)"
        )

    rng = np.random.default_rng(seed)
    targets = compute_joint_targets(strata_weights, target_size)
    grouped = df.groupby(list(strata_cols), observed=True, dropna=False)

    sampled_frames: list[pd.DataFrame] = []
    per_stratum: dict[tuple[Any, ...], StratumOutcome] = {}

    for key, target in targets.items():
        try:
            group = grouped.get_group(key if len(strata_cols) > 1 else key[0])
        except KeyError:
            group = df.iloc[0:0]

        available = len(group)
        n = min(target, available)
        if n < target:
            logger.warning(
                "stratified_sample: stratum %s wanted %d rows but only %d available "
                "(shortfall %d) — corpus will be smaller than target_size unless "
                "re-run with a larger source pool.",
                key,
                target,
                available,
                target - n,
            )
        if n > 0:
            chosen_idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
            sampled_frames.append(df.loc[chosen_idx])

        per_stratum[key] = StratumOutcome(target=target, available=available, sampled=n)

    sampled = pd.concat(sampled_frames) if sampled_frames else df.iloc[0:0]
    # Shuffle final row order so strata aren't laid out in contiguous
    # blocks (which would bias anything downstream that reads sequentially,
    # e.g. a train/val split taken from the head of the file).
    shuffled_order = rng.permutation(len(sampled))
    sampled = sampled.iloc[shuffled_order]

    report = SamplingReport(target_size=target_size, actual_size=len(sampled), per_stratum=per_stratum)
    if report.shortfall_total:
        logger.info(
            "stratified_sample: delivered %d/%d requested rows (shortfall %d)",
            report.actual_size,
            report.target_size,
            report.shortfall_total,
        )
    return sampled, report
