"""Judge validation: human-scoring worksheet + Cohen's kappa (T4b.6; PLAN.md §1.4
risk E, §8, §11, M4b.md T4b.6).

T4b.4's judge is a rubric AI-judge, and PLAN.md's own risk register (§1.4 risk E) is
explicit that an un-validated one shouldn't be trusted at scale: "LLM judges have
position/verbosity/self-preference bias." This module is the validation step T4b.7's
RQ0/RQ1 tables lean on before treating judge scores as ground truth: sample a subset of
T4b.5's 180 judge-scored answers, have a human independently score the same rubric, and
measure agreement.

Two-step workflow, mirroring `cragb.bench.label_relevance`'s `export`/`validate` split
for the same reason (T2.7's own worksheet-based human-labeling precedent):

    python -m cragb.eval.judge_validation export   # writes the blank-scores worksheet
    # ... a human fills in every row's four score blanks by hand ...
    python -m cragb.eval.judge_validation score     # reads it back, computes agreement

**The worksheet hides everything the judge itself was never allowed to see, and one
thing more.** Every row shows only the question, the context the answerer had (or
`judge.NO_CONTEXT_MARKER` if none), the candidate answer, and the reference answer --
never the arm, never the judge's own score. Rows are drawn from every arm in roughly
equal numbers (PLAN.md's instinct against sampling only "easy" cases: a plain random
draw within each arm, not filtered by score, keeps every arm represented and every row's
inclusion independent of how well the judge happened to score it) and then the *combined*
sample is shuffled before anonymous row ids (R01, R02, ...) are assigned -- grouping by
arm would let a human infer arm identity from neighbouring rows alone (e.g. "these three
in a row all say 'not enough information', this block must be the no-context arm"), the
same leak PLAN.md §9 already guards the judge itself against.

**No separate key file.** Sampling is a pure, seeded function of `judge_scores_v1.csv`
and `references_in` (`build_sample`), so `score_worksheet` reconstructs the identical
sample from the same config rather than reading back a persisted row-id map. To catch
the one real risk that creates -- `judge_scores_v1.csv` (or the references) changing
between `export` and `score` -- `score_worksheet` cross-checks that the re-derived
sample's row ids exactly match what the worksheet actually contains, and raises with a
concrete "re-export and re-score" instruction rather than silently misaligning rows.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from cragb.bench.reference_answers import ReferenceAnswer, load_reference_answers
from cragb.eval.judge import NO_CONTEXT_MARKER
from cragb.eval.run_answer_generation import ARM_DEFAULT_OUT
from cragb.eval.run_grounded_qa_pilot import write_csv
from cragb.eval.run_judge_eval import context_text_for, load_arm_transcripts
from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

_CRITERIA: tuple[str, ...] = ("correctness", "faithfulness", "completeness", "conciseness")
_RATING_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SampledRow:
    """One worksheet row: full content plus the judge's own (hidden-from-worksheet) score."""

    row_id: str
    arm: str
    question_id: str
    question: str
    context_text: str | None
    candidate_answer: str
    reference_answer: str
    judge_scores: dict[str, int]


def build_sample(
    judge_scores: pd.DataFrame,
    references: dict[str, ReferenceAnswer],
    sample_size: int,
    seed: int,
) -> list[SampledRow]:
    """Seeded stratified sample of `judge_scores`, joined back to full row content.

    Args:
        judge_scores: T4b.5's full score table (one row per `(arm, question_id)`).
        references: from `cragb.bench.reference_answers.load_reference_answers`.
        sample_size: total rows to sample across every arm combined.
        seed: seeds both the per-arm draw and the shuffle -- deterministic, so
            `score_worksheet` can reconstruct this exact sample later from the same
            inputs (module docstring: no separate persisted key file).

    Returns:
        `sample_size` `SampledRow`s in the worksheet's (shuffled, arm-blind) row order.

    Raises:
        ValueError: if `sample_size` requests more rows from some arm than that arm
            has, or if `judge_scores` is empty.
    """
    if judge_scores.empty:
        raise ValueError("Cannot sample from an empty judge_scores table.")

    arms = sorted(judge_scores["arm"].unique())
    n_arms = len(arms)
    base, remainder = divmod(sample_size, n_arms)
    rng = np.random.default_rng(seed)

    picked_indices: list[int] = []
    for i, arm in enumerate(arms):
        arm_df = judge_scores[judge_scores["arm"] == arm]
        n_for_arm = base + (1 if i < remainder else 0)
        if n_for_arm > len(arm_df):
            raise ValueError(
                f"Requested {n_for_arm} sample(s) from arm {arm!r} but only "
                f"{len(arm_df)} row(s) available."
            )
        chosen = rng.choice(arm_df.index.to_numpy(), size=n_for_arm, replace=False)
        picked_indices.extend(int(i) for i in chosen)

    rng.shuffle(picked_indices)  # arms must not cluster in worksheet order -- see module docstring

    transcripts_cache: dict[str, dict[str, object]] = {}
    rows: list[SampledRow] = []
    for i, idx in enumerate(picked_indices):
        record = judge_scores.loc[idx]
        arm, qid = str(record["arm"]), str(record["question_id"])
        if arm not in transcripts_cache:
            transcripts_cache[arm] = {
                t.question_id: t for t in load_arm_transcripts(arm, ARM_DEFAULT_OUT[arm])
            }
        transcript = transcripts_cache[arm][qid]
        rows.append(
            SampledRow(
                row_id=f"R{i + 1:02d}",
                arm=arm,
                question_id=qid,
                question=transcript.question,
                context_text=context_text_for(arm, transcript),
                candidate_answer=transcript.answer_text,
                reference_answer=references[qid].answer,
                judge_scores={c: int(record[c]) for c in _CRITERIA},
            )
        )
    return rows


def render_worksheet(rows: list[SampledRow]) -> str:
    """Render the human-facing worksheet: one anonymized, arm-blind block per row.

    Deliberately omits `arm`, `question_id`, and the judge's own scores -- see module
    docstring for why. The blank score lines (`- correctness: `, one per criterion,
    nothing after the colon) are what `parse_worksheet` looks for when reading a
    filled-in copy back.
    """
    lines = [
        "# CRAGB Judge Validation Worksheet (v1)",
        "",
        f"{len(rows)} answers, sampled across all three answer-generation arms and "
        "shuffled so arm identity can't be guessed from row order or from anything "
        "adjacent. For each row below, read the question, the context the answerer had "
        "(if any), the candidate answer, and the reference answer, then write your own "
        "1-5 score on the blank line after each criterion's colon. Score every row and "
        "every criterion -- an incomplete worksheet will not parse.",
        "",
        "**Criteria (1 = very poor, 5 = excellent):**",
        "- **correctness** -- does the candidate's substance agree with the reference?",
        "- **faithfulness** -- is every claim traceable to the context shown, or -- if "
        "there is none -- is the candidate honest about not knowing?",
        "- **completeness** -- does it cover the same ground as the reference?",
        "- **conciseness** -- as short as possible while still answering fully.",
        "",
        "If the reference states there isn't enough information and the candidate "
        "says the same in substance, that's a correct, faithful, complete, concise "
        "answer (5 on every criterion) -- the same rule the judge itself was given.",
        "",
    ]
    for row in rows:
        context_display = row.context_text if row.context_text is not None else NO_CONTEXT_MARKER
        lines.append(f"## {row.row_id}")
        lines.append("")
        lines.append(f"**Question:** {row.question}")
        lines.append("")
        lines.append(f"**Context shown to the answerer:** {context_display}")
        lines.append("")
        lines.append(f"**Candidate answer:** {row.candidate_answer}")
        lines.append("")
        lines.append(f"**Reference answer:** {row.reference_answer}")
        lines.append("")
        lines.append("**Your scores (integer 1-5 each):**")
        for criterion in _CRITERIA:
            lines.append(f"- {criterion}: ")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_worksheet(text: str) -> dict[str, dict[str, int]]:
    """Parse a filled-in worksheet into `{row_id: {criterion: score}}`.

    Args:
        text: the worksheet file's full text, after a human has filled in every score
            blank `render_worksheet` left.

    Returns:
        One entry per `## R##` block found, each mapping every criterion in
        `_CRITERIA` to its integer score.

    Raises:
        ValueError: if any row is missing a `<criterion>:` line entirely, if any
            score is left blank, or if a filled-in value is not an integer in
            `[1, 5]` -- refuses to run on an incomplete or malformed worksheet rather
            than silently dropping or defaulting a missing score.
    """
    blocks = re.split(r"(?m)^## (R\d+)\s*$", text)
    # re.split with a capturing group interleaves: [preamble, id1, block1, id2, block2, ...]
    scores: dict[str, dict[str, int]] = {}
    for row_id, block in zip(blocks[1::2], blocks[2::2]):
        row_scores: dict[str, int] = {}
        for criterion in _CRITERIA:
            match = re.search(rf"(?im)^- {criterion}:\s*(\S*)\s*$", block)
            if match is None:
                raise ValueError(f"{row_id}: no {criterion!r} line found in the worksheet.")
            raw_value = match.group(1).strip()
            if not raw_value:
                raise ValueError(
                    f"{row_id}: {criterion!r} is blank -- every row must be scored before running this."
                )
            try:
                value = int(raw_value)
            except ValueError:
                raise ValueError(f"{row_id}: {criterion!r} value {raw_value!r} is not an integer.") from None
            if value not in _RATING_LABELS:
                raise ValueError(f"{row_id}: {criterion!r}={value} is not an integer in [1, 5].")
            row_scores[criterion] = value
        scores[row_id] = row_scores
    return scores


def compute_agreement(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-criterion Cohen's kappa (quadratic-weighted) and Spearman correlation.

    Args:
        paired: one row per scored item, with integer `judge_<criterion>` and
            `human_<criterion>` columns for every criterion in `_CRITERIA` (see
            `score_worksheet` for how this is built from a real sample + worksheet).

    Returns:
        One row per criterion: `criterion`, `n`, `cohens_kappa` (quadratic-weighted,
        appropriate for this 1-5 ordinal rubric per PLAN.md §3 E5), `spearman_r`,
        `pct_within_one_point` (share of pairs differing by at most 1 point -- a
        plain-language sanity check alongside kappa, per M4b.md's own "How to verify"
        instruction).

    Raises:
        ValueError: if `paired` is empty.
    """
    if paired.empty:
        raise ValueError("Cannot compute agreement over an empty paired score table.")

    rows = []
    for criterion in _CRITERIA:
        judge_vals = paired[f"judge_{criterion}"].to_numpy()
        human_vals = paired[f"human_{criterion}"].to_numpy()
        kappa = cohen_kappa_score(judge_vals, human_vals, labels=list(_RATING_LABELS), weights="quadratic")
        corr, _p_value = spearmanr(judge_vals, human_vals)
        within_one = float(np.mean(np.abs(judge_vals - human_vals) <= 1))
        rows.append(
            {
                "criterion": criterion,
                "n": len(paired),
                "cohens_kappa": float(kappa),
                "spearman_r": float(corr),
                "pct_within_one_point": within_one,
            }
        )
    return pd.DataFrame(rows)


def export_worksheet(
    judge_scores_path: str | Path,
    references_path: str | Path,
    sample_size: int,
    seed: int,
    out_path: str | Path,
) -> tuple[Path, list[SampledRow]]:
    """Sample, render, and write the human-scoring worksheet."""
    judge_scores = pd.read_csv(resolve_path(judge_scores_path))
    references = load_reference_answers(references_path)
    rows = build_sample(judge_scores, references, sample_size, seed)

    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_worksheet(rows), encoding="utf-8")
    return resolved, rows


def build_paired_scores(
    worksheet_path: str | Path,
    judge_scores_path: str | Path,
    references_path: str | Path,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Re-derive the sample and pair it against a filled-in worksheet's human scores.

    Re-derives the sample `export_worksheet` produced (same
    `judge_scores_path`/`references_path`/`sample_size`/`seed`) rather than reading a
    persisted key file -- see module docstring. Split out of `score_worksheet` (T4b.6)
    as its own function for T4b.8: the judge-vs-human agreement *figure* needs these
    raw per-row paired values, not just `compute_agreement`'s aggregated summary.

    Args:
        worksheet_path: a filled-in worksheet (`export_worksheet`'s output, hand-scored).
        judge_scores_path, references_path, sample_size, seed: must match what
            `export_worksheet` was called with, to reconstruct the same sample.

    Returns:
        One row per sampled item: `row_id`, `arm`, `question_id`, and
        `judge_<criterion>`/`human_<criterion>` (integers) for every criterion in
        `_CRITERIA`.

    Raises:
        ValueError: everything `parse_worksheet` can raise (incomplete/malformed
            worksheet), plus a mismatch error if the worksheet's row ids don't exactly
            match the re-derived sample's -- the concrete symptom of
            `judge_scores_v1.csv` or the references having changed since this
            worksheet was exported.
    """
    judge_scores = pd.read_csv(resolve_path(judge_scores_path))
    references = load_reference_answers(references_path)
    rows = build_sample(judge_scores, references, sample_size, seed)

    worksheet_text = resolve_path(worksheet_path).read_text(encoding="utf-8")
    human_scores = parse_worksheet(worksheet_text)

    expected_row_ids = {row.row_id for row in rows}
    got_row_ids = set(human_scores)
    if expected_row_ids != got_row_ids:
        missing = sorted(expected_row_ids - got_row_ids)
        extra = sorted(got_row_ids - expected_row_ids)
        raise ValueError(
            f"Worksheet rows do not match the re-derived sample for seed={seed} "
            f"(missing={missing}, extra={extra}). If judge_scores_v1.csv or the "
            "reference answers changed since this worksheet was exported, re-run "
            "'export' and re-score the new worksheet from scratch."
        )

    records = []
    for row in rows:
        human = human_scores[row.row_id]
        record = {"row_id": row.row_id, "arm": row.arm, "question_id": row.question_id}
        for criterion in _CRITERIA:
            record[f"judge_{criterion}"] = row.judge_scores[criterion]
            record[f"human_{criterion}"] = human[criterion]
        records.append(record)

    return pd.DataFrame(records)


def score_worksheet(
    worksheet_path: str | Path,
    judge_scores_path: str | Path,
    references_path: str | Path,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Read a filled-in worksheet back and compute judge-vs-human agreement.

    A thin wrapper: `build_paired_scores` does the re-derivation and pairing,
    `compute_agreement` turns that into the per-criterion summary this returns.

    Raises:
        Everything `build_paired_scores` and `compute_agreement` can raise.
    """
    paired = build_paired_scores(worksheet_path, judge_scores_path, references_path, sample_size, seed)
    return compute_agreement(paired)


def plot_judge_human_agreement(
    paired: pd.DataFrame, agreement: pd.DataFrame, out_path: str | Path, seed: int = 0
) -> Path:
    """Judge score vs human score, one scatter panel per criterion, kappa annotated.

    PLAN.md §7 figure: "judge-vs-human agreement plot." A 2x2 grid, one panel per
    `_CRITERIA` entry, each an (x=judge, y=human) scatter with the `y=x` "perfect
    agreement" line and that criterion's Cohen's kappa/n in the title. Scores are
    integers 1-5, so a small deterministic jitter is added to both axes purely for
    display -- otherwise many points would land exactly on top of each other and the
    density of disagreement would be invisible.

    Uses the non-interactive `Agg` backend explicitly, set locally to this function
    rather than relying on whatever backend matplotlib happens to auto-select --
    that auto-selection has been observed to resolve to `TkAgg` on this project's own
    dev machine, which fails outright on a broken Tk install (an unrelated, pre-existing
    environment issue this function does not depend on being fixed).

    Args:
        paired: from `build_paired_scores` -- `judge_<criterion>`/`human_<criterion>`
            columns for every criterion in `_CRITERIA`.
        agreement: from `compute_agreement(paired)` -- supplies the annotated kappa/n
            per criterion, so the figure can never silently disagree with
            `judge_validation_v1.csv`'s own numbers by recomputing them differently.
        out_path: where to save the PNG.
        seed: seeds the display jitter, so the figure is reproducible byte-for-byte
            across re-runs rather than shifting points randomly each time.

    Returns:
        The resolved path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    for ax, criterion in zip(axes.flat, _CRITERIA):
        judge_vals = paired[f"judge_{criterion}"].to_numpy(dtype=float)
        human_vals = paired[f"human_{criterion}"].to_numpy(dtype=float)
        jitter_x = rng.uniform(-0.15, 0.15, size=len(judge_vals))
        jitter_y = rng.uniform(-0.15, 0.15, size=len(human_vals))

        ax.plot([0.5, 5.5], [0.5, 5.5], linestyle="--", color="gray", linewidth=1, label="perfect agreement")
        ax.scatter(judge_vals + jitter_x, human_vals + jitter_y, alpha=0.6, s=28, edgecolors="none")

        row = agreement.loc[agreement["criterion"] == criterion].iloc[0]
        ax.set_title(f"{criterion}  (κ={row['cohens_kappa']:.2f}, n={int(row['n'])})")
        ax.set_xlabel("judge score")
        ax.set_ylabel("human score")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Judge vs human agreement (T4b.6 validation worksheet, jittered for display)")
    fig.tight_layout()

    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved, dpi=150)
    plt.close(fig)
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/judge_validation.yaml", help="Path to judge-validation config YAML."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "command", choices=["export", "score"], help="export the worksheet, or score a filled-in one"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    seed = cfg["seed"]
    sample_size = cfg["sampling"]["sample_size"]

    if args.command == "export":
        out_path, rows = export_worksheet(
            cfg["paths"]["judge_scores_in"],
            cfg["paths"]["references_in"],
            sample_size,
            seed,
            cfg["paths"]["worksheet_out"],
        )
        arm_counts = pd.Series([row.arm for row in rows]).value_counts().to_dict()
        logger.info("Wrote %d-row worksheet to %s", len(rows), out_path)
        logger.info("Per-arm sample counts: %s", arm_counts)
        return 0

    # args.command == "score"
    summary = score_worksheet(
        cfg["paths"]["worksheet_out"],
        cfg["paths"]["judge_scores_in"],
        cfg["paths"]["references_in"],
        sample_size,
        seed,
    )
    out_path = write_csv(summary, cfg["paths"]["validation_out"])
    logger.info("Wrote judge-validation table to %s", out_path)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
