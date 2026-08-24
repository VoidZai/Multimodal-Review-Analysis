"""Human spot-check of the vision-evidence judge: worksheet + Cohen's kappa
(T6.6; PLAN.md §1.4 risk E, §8 G3, M6.md T6.6).

T6.4/T6.5's vision judge is an AI judge, and PLAN.md's own risk register
(§1.4 risk E) is explicit that no judge should be trusted before it's
validated -- the text judge got this treatment in T4b.6
(`cragb.eval.judge_validation`); this module gives the vision judge the
same treatment at pilot scale. RQ4's headline win-rate is only as
trustworthy as the procedure producing it.

Mirrors `judge_validation.py`'s two-step worksheet workflow exactly, for
the same reasons that module gives:

    python -m cragb.multimodal.vision_spotcheck export
    # ... a human looks at both photos in every row and writes A/B/tie ...
    python -m cragb.multimodal.vision_spotcheck score

**The worksheet hides the judge's own verdict.** Every row shows only the
question and the two photo file paths (labelled Photo A = the pipeline's
surfaced photo, Photo B = the random control -- consistently, never
shuffled per row) -- never `mm_verdicts_v1.jsonl`'s `outcome` for that
pair. Comparing against `outcome` (not one raw sub-call) is what makes
this spot-check validate the actual number RQ4's win-rate is built from.

**Stratified by outcome, not by question type.** With only 25 pairs total
and three outcome categories split roughly 12/3/10 (T6.5's live run),
requesting an *equal* share per category the way T4b.6 does per arm would
demand more `control_win` rows than exist at all. `_apportion_quotas` uses
the same largest-remainder (Hamilton) method
`cragb.eval.run_cost_latency.stratified_question_sample` already
established for a different grouping, generalized here to whatever
grouping key is given -- proportional to each group's size rather than
equal, so a quota can never exceed that group's own pool (guaranteed for
`n <= total`, not merely hoped for).

**No separate key file**, same reasoning as `judge_validation.py`:
`score_worksheet` re-derives the identical sample from
`mm_verdicts_v1.jsonl` + `mm_pairs_v1.jsonl` + the same seed, and
cross-checks the re-derived row ids against what the worksheet actually
contains before trusting it.

**Agreement uses unweighted, not quadratic-weighted, Cohen's kappa.**
T4b.6's rubric scores are ordinal (1-5; "off by one" is a real, graded
notion of disagreement, so quadratic weighting is appropriate).
`A`/`B`/`tie` is nominal -- there's no meaningful sense in which "tie" is
"between" A and B -- so plain (unweighted) kappa is the correct measure
here, not an oversight relative to T4b.6.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from cragb.multimodal.photo_store import PhotoStore
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

_VALID_VERDICTS: tuple[str, ...] = ("A", "B", "tie")

# outcome (mm_verdicts_v1.jsonl) -> A/B/tie in this worksheet's fixed
# Photo A = surfaced, Photo B = control convention.
_OUTCOME_TO_VERDICT: dict[str, str] = {"surfaced_win": "A", "control_win": "B", "tie": "tie"}


@dataclass(frozen=True)
class SpotcheckRow:
    """One worksheet row: full content plus the judge's own (hidden) verdict."""

    row_id: str
    question_id: str
    type: str
    question: str
    photo_a_path: str  # surfaced
    photo_b_path: str  # control
    judge_verdict: str  # "A" | "B" | "tie"


def _load_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _apportion_quotas(n: int, counts_by_key: dict[str, int]) -> dict[str, int]:
    """Largest-remainder (Hamilton) proportional apportionment of `n` seats.

    Generalizes `cragb.eval.run_cost_latency.stratified_question_sample`'s
    per-type apportionment to any grouping key: `floor(n * count_k / total)`
    per group, with the `n - sum(quotas)` leftover seats given to the
    groups with the largest fractional remainder (ties broken by key name,
    for full determinism independent of dict ordering).

    Args:
        n: total seats to allocate.
        counts_by_key: each group's pool size.

    Returns:
        `{key: quota}`, `sum(quotas) == n`. Proportional allocation
        guarantees `quota <= counts_by_key[key]` for every key whenever
        `n <= sum(counts_by_key.values())` -- a quota can never exceed its
        own group's pool.

    Raises:
        ValueError: if `n` exceeds the total pool across all groups.
    """
    total = sum(counts_by_key.values())
    if n > total:
        raise ValueError(f"n={n} exceeds the available pool of {total}")
    exact = {k: n * c / total for k, c in counts_by_key.items()}
    quotas = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(quotas.values())
    remainder_order = sorted(counts_by_key, key=lambda k: (-(exact[k] - quotas[k]), k))
    for k in remainder_order[:remainder]:
        quotas[k] += 1
    return quotas


def build_sample(
    verdicts: pd.DataFrame,
    pairs: pd.DataFrame,
    n: int,
    seed: int,
    *,
    photos_dir: str | Path = "data/photos",
) -> list[SpotcheckRow]:
    """Seeded sample of judged pairs, stratified by `outcome`, joined to full row content.

    Args:
        verdicts: T6.5's `mm_verdicts_v1.jsonl` loaded (`_load_jsonl`) -- has
            `question_id`, `type`, `outcome`.
        pairs: T6.3's `mm_pairs_v1.jsonl` loaded -- has `question_id`,
            `question`, `surfaced_photo_id`, `control_photo_id`.
        n: total rows to sample.
        seed: seeds both the per-outcome draw and the shuffle -- deterministic,
            so `score_worksheet` can reconstruct this exact sample later.
        photos_dir: where cached photo files live (`PhotoStore.photo_path`).

    Returns:
        `n` `SpotcheckRow`s in the worksheet's (shuffled, verdict-blind) row order.

    Raises:
        ValueError: `verdicts` is empty, `n` exceeds the pool size, or a
            verdict row has no matching pair.
        FileNotFoundError: a sampled pair's surfaced or control photo isn't
            cached (should not happen for pairs that passed T6.3's
            fetchable-bytes stage, but this is not re-verified here).
    """
    if verdicts.empty:
        raise ValueError("Cannot sample from an empty verdicts table.")

    merged = verdicts.merge(
        pairs[["question_id", "question", "surfaced_photo_id", "control_photo_id"]],
        on="question_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(verdicts):
        raise ValueError(
            f"{len(verdicts) - len(merged)} verdict row(s) have no matching pair in "
            "mm_pairs_v1.jsonl."
        )

    counts = merged["outcome"].value_counts().to_dict()
    quotas = _apportion_quotas(n, counts)

    rng = np.random.default_rng(seed)
    picked_indices: list[int] = []
    for outcome in sorted(quotas):  # stable order for determinism, independent of dict/groupby ordering
        quota = quotas[outcome]
        if quota == 0:
            continue
        group = merged[merged["outcome"] == outcome]
        chosen = rng.choice(group.index.to_numpy(), size=quota, replace=False)
        picked_indices.extend(int(i) for i in chosen)

    rng.shuffle(picked_indices)  # outcome mix must not be inferable from row order

    store = PhotoStore(photos_dir=photos_dir)
    rows: list[SpotcheckRow] = []
    for i, idx in enumerate(picked_indices):
        record = merged.loc[idx]
        surfaced_path = store.photo_path(record["surfaced_photo_id"])
        control_path = store.photo_path(record["control_photo_id"])
        if surfaced_path is None or control_path is None:
            raise FileNotFoundError(
                f"question_id={record['question_id']!r}: photo not cached "
                f"(surfaced={surfaced_path}, control={control_path})."
            )
        rows.append(
            SpotcheckRow(
                row_id=f"R{i + 1:02d}",
                question_id=str(record["question_id"]),
                type=str(record["type"]),
                question=str(record["question"]),
                photo_a_path=str(surfaced_path),
                photo_b_path=str(control_path),
                judge_verdict=_OUTCOME_TO_VERDICT[record["outcome"]],
            )
        )
    return rows


def render_worksheet(rows: list[SpotcheckRow]) -> str:
    """Render the human-facing worksheet: one photo-order-fixed, verdict-blind block per row.

    Deliberately omits the judge's `outcome`/rationale/confidence -- see
    module docstring. The blank `my_verdict:`/`notes:` lines are what
    `parse_worksheet` looks for when reading a filled-in copy back.
    """
    lines = [
        "# CRAGB Vision-Judge Spot-Check Worksheet (v1)",
        "",
        f"{len(rows)} judged photo pairs, sampled proportionally across the judge's own "
        "outcome categories (surfaced-win / control-win / tie) and shuffled so that mix "
        "can't be guessed from row order. For each row: open both photo files, decide "
        "which one is better **evidence for the question** (not which looks nicer or is "
        "better lit), and write `A`, `B`, or `tie` on the my_verdict line. The judge's "
        "own verdict is never shown here so your answer isn't anchored to it -- score "
        "every row, an incomplete worksheet will not parse.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row.row_id}")
        lines.append("")
        lines.append(f"**Question:** {row.question}")
        lines.append("")
        lines.append(f"**Photo A:** `{row.photo_a_path}`")
        lines.append(f"**Photo B:** `{row.photo_b_path}`")
        lines.append("")
        lines.append("**Your verdict:**")
        lines.append("- my_verdict: ")
        lines.append("- notes: ")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_worksheet(text: str) -> dict[str, dict[str, str]]:
    """Parse a filled-in worksheet into `{row_id: {"my_verdict": ..., "notes": ...}}`.

    Args:
        text: the worksheet file's full text, after a human has filled in
            every `my_verdict:` blank `render_worksheet` left.

    Returns:
        One entry per `## R##` block found. `notes` is `""` if the line was
        left blank or omitted -- notes are optional context, not a scored
        field, so an empty note never blocks parsing.

    Raises:
        ValueError: a row has no `my_verdict:` line at all, the value is
            blank, or it isn't one of `"A"`/`"B"`/`"tie"` (case-insensitive)
            -- refuses to run on an incomplete or malformed worksheet
            rather than silently dropping or defaulting a missing verdict.
    """
    blocks = re.split(r"(?m)^## (R\d+)\s*$", text)
    # re.split with a capturing group interleaves: [preamble, id1, block1, id2, block2, ...]
    result: dict[str, dict[str, str]] = {}
    for row_id, block in zip(blocks[1::2], blocks[2::2]):
        # [ \t]*, not \s*, before the capture group: \s* would happily consume the
        # trailing newline of a BLANK value and keep eating into the *next* line
        # (e.g. "- notes:"), since \n is whitespace too -- a real bug caught by
        # test_unfilled_verdict_raises capturing "- notes:" as the "verdict" instead
        # of raising the intended blank-value error. [ \t]* only ever eats same-line
        # whitespace, so an empty value stays empty.
        verdict_match = re.search(r"(?im)^- my_verdict:[ \t]*(.*)$", block)
        if verdict_match is None:
            raise ValueError(f"{row_id}: no 'my_verdict:' line found in the worksheet.")
        raw_verdict = verdict_match.group(1).strip()
        if not raw_verdict:
            raise ValueError(
                f"{row_id}: my_verdict is blank -- every row must be scored before running this."
            )
        normalized = raw_verdict.upper() if raw_verdict.upper() in ("A", "B") else raw_verdict.lower()
        if normalized not in _VALID_VERDICTS:
            raise ValueError(f"{row_id}: my_verdict={raw_verdict!r} is not one of {_VALID_VERDICTS}.")

        notes_match = re.search(r"(?im)^- notes:[ \t]*(.*)$", block)
        notes = notes_match.group(1).strip() if notes_match else ""

        result[row_id] = {"my_verdict": normalized, "notes": notes}
    return result


def compute_agreement(paired: pd.DataFrame) -> pd.DataFrame:
    """Raw agreement % and (unweighted) Cohen's kappa between judge and human verdicts.

    Args:
        paired: one row per scored item, with `judge_verdict` and
            `human_verdict` columns (each `"A"`/`"B"`/`"tie"`).

    Returns:
        A single-row DataFrame: `n`, `n_agree`, `raw_agreement`,
        `cohens_kappa` -- unweighted, since `A`/`B`/`tie` is nominal, not
        ordinal (see module docstring).

    Raises:
        ValueError: if `paired` is empty.
    """
    if paired.empty:
        raise ValueError("Cannot compute agreement over an empty paired verdict table.")

    judge_vals = paired["judge_verdict"].to_numpy()
    human_vals = paired["human_verdict"].to_numpy()
    n = len(paired)
    n_agree = int(np.sum(judge_vals == human_vals))
    kappa = cohen_kappa_score(judge_vals, human_vals, labels=list(_VALID_VERDICTS))

    return pd.DataFrame(
        [
            {
                "n": n,
                "n_agree": n_agree,
                "raw_agreement": n_agree / n,
                "cohens_kappa": float(kappa),
            }
        ]
    )


def export_worksheet(
    verdicts_path: str | Path,
    pairs_path: str | Path,
    n: int,
    seed: int,
    out_path: str | Path,
    *,
    photos_dir: str | Path = "data/photos",
) -> tuple[Path, list[SpotcheckRow]]:
    """Sample, render, and write the human spot-check worksheet."""
    verdicts = _load_jsonl(verdicts_path)
    pairs = _load_jsonl(pairs_path)
    rows = build_sample(verdicts, pairs, n, seed, photos_dir=photos_dir)

    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_worksheet(rows), encoding="utf-8")
    return resolved, rows


def build_paired_scores(
    worksheet_path: str | Path,
    verdicts_path: str | Path,
    pairs_path: str | Path,
    n: int,
    seed: int,
    *,
    photos_dir: str | Path = "data/photos",
) -> pd.DataFrame:
    """Re-derive the sample and pair it against a filled-in worksheet's human verdicts.

    Re-derives the sample `export_worksheet` produced rather than reading a
    persisted key file -- see module docstring.

    Args:
        worksheet_path: a filled-in worksheet (`export_worksheet`'s output).
        verdicts_path, pairs_path, n, seed, photos_dir: must match what
            `export_worksheet` was called with, to reconstruct the same sample.

    Returns:
        One row per sampled item: `row_id`, `question_id`, `type`,
        `judge_verdict`, `human_verdict`, `notes`.

    Raises:
        ValueError: everything `parse_worksheet`/`build_sample` can raise,
            plus a mismatch error if the worksheet's row ids don't exactly
            match the re-derived sample's.
    """
    verdicts = _load_jsonl(verdicts_path)
    pairs = _load_jsonl(pairs_path)
    rows = build_sample(verdicts, pairs, n, seed, photos_dir=photos_dir)

    worksheet_text = resolve_path(worksheet_path).read_text(encoding="utf-8")
    human_verdicts = parse_worksheet(worksheet_text)

    expected_row_ids = {row.row_id for row in rows}
    got_row_ids = set(human_verdicts)
    if expected_row_ids != got_row_ids:
        missing = sorted(expected_row_ids - got_row_ids)
        extra = sorted(got_row_ids - expected_row_ids)
        raise ValueError(
            f"Worksheet rows do not match the re-derived sample for seed={seed} "
            f"(missing={missing}, extra={extra}). If mm_verdicts_v1.jsonl or "
            "mm_pairs_v1.jsonl changed since this worksheet was exported, re-run "
            "'export' and re-score the new worksheet from scratch."
        )

    records = []
    for row in rows:
        human = human_verdicts[row.row_id]
        records.append(
            {
                "row_id": row.row_id,
                "question_id": row.question_id,
                "type": row.type,
                "judge_verdict": row.judge_verdict,
                "human_verdict": human["my_verdict"],
                "notes": human["notes"],
            }
        )
    return pd.DataFrame(records)


def score_worksheet(
    worksheet_path: str | Path,
    verdicts_path: str | Path,
    pairs_path: str | Path,
    n: int,
    seed: int,
    *,
    photos_dir: str | Path = "data/photos",
) -> pd.DataFrame:
    """Read a filled-in worksheet back and compute judge-vs-human agreement.

    A thin wrapper: `build_paired_scores` does the re-derivation and
    pairing, `compute_agreement` turns that into the summary this returns.
    """
    paired = build_paired_scores(worksheet_path, verdicts_path, pairs_path, n, seed, photos_dir=photos_dir)
    return compute_agreement(paired)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write `df` to `out_path`, creating parent directories as needed."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved, index=False)
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/vision_spotcheck.yaml", help="Path to vision-spotcheck config YAML."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("command", choices=["export", "score"], help="export the worksheet, or score a filled-in one")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    seed = cfg["seed"]
    n = cfg["sampling"]["n"]
    paths = cfg["paths"]

    if args.command == "export":
        out_path, rows = export_worksheet(
            paths["verdicts_in"],
            paths["pairs_in"],
            n,
            seed,
            paths["worksheet_out"],
            photos_dir=paths["photos_dir"],
        )
        outcome_counts = pd.Series([row.judge_verdict for row in rows]).value_counts().to_dict()
        logger.info("Wrote %d-row worksheet to %s", len(rows), out_path)
        logger.info("Sampled judge-verdict mix: %s", outcome_counts)
        return 0

    # args.command == "score"
    summary = score_worksheet(
        paths["worksheet_out"],
        paths["verdicts_in"],
        paths["pairs_in"],
        n,
        seed,
        photos_dir=paths["photos_dir"],
    )
    out_path = write_csv(summary, paths["spotcheck_out"])
    logger.info("Wrote spot-check agreement table to %s", out_path)
    row = summary.iloc[0]
    if row["cohens_kappa"] >= 0.4:
        logger.info(
            "kappa=%.2f (n=%d) -- usable: the judge's win-rate can be reported as validated.",
            row["cohens_kappa"], int(row["n"]),
        )
    else:
        logger.warning(
            "kappa=%.2f (n=%d) is below the 0.4 usability bar -- report this as a "
            "reliability limitation on RQ4, do not omit it.",
            row["cohens_kappa"], int(row["n"]),
        )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
