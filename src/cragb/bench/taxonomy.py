"""CRAGB question taxonomy: definition, corpus-coverage check, datasheet.

T2.1 (M2.md; PLAN.md §3 E1). Before drafting a single CRAGB question
(T2.2), we fix the taxonomy the benchmark will be balanced against — 7
categories spanning fit/sizing, colour/appearance, fabric/quality,
durability, defects, occasion, and value — and check that `corpus_v1`
actually contains enough language in each category to support the
questions we're about to write. Skipping this check risks drafting
questions the corpus has no evidence for, an error that would otherwise
only surface much later, during CRAGB pooling (T2.6) — PLAN.md §3 E1
"Failure modes".

`fit_sizing`, `colour_appearance`, and `fabric_quality` reuse the keyword
lists T1.11 already curated and validated in
`cragb.data.vocab_check.ATTRIBUTE_KEYWORDS` (single source of truth for
those terms); `durability`, `defects`, `occasion`, and `value` are new
here. Counts/descriptions/thresholds are config-driven
(`configs/taxonomy.yaml`); keyword lists stay in code, matching the
precedent T1.11 set (small curated lists are easier to review as code,
with a comment on *why* each term is there, than as YAML string arrays).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cragb.data.vocab_check import ATTRIBUTE_KEYWORDS, attribute_vocab_coverage
from cragb.utils.io import load_config, resolve_path

# fit_sizing merges T1.11's "size" + "fit" lists (both drive the same
# question category here); colour_appearance and fabric_quality are
# renamed 1:1 from "colour"/"fabric". Union + dict.fromkeys de-duplicates
# while preserving first-seen order (e.g. "true to size" appears in both
# source lists).
_FIT_SIZING_TERMS = list(
    dict.fromkeys(ATTRIBUTE_KEYWORDS["size"] + ATTRIBUTE_KEYWORDS["fit"])
)

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "fit_sizing": _FIT_SIZING_TERMS,
    "colour_appearance": ATTRIBUTE_KEYWORDS["colour"],
    "fabric_quality": ATTRIBUTE_KEYWORDS["fabric"],
    "durability": [
        "durable", "durability", "lasted", "lasts", "long lasting",
        "long-lasting", "held up", "holding up", "falling apart",
        "fell apart", "wear and tear", "worn out", "still looks new",
        "torn", "tear", "tears", "ripped", "rip", "rips", "unraveled",
        "unravel", "pilling", "pills",
    ],
    "defects": [
        "defect", "defective", "flaw", "flawed", "damaged", "damage",
        "broken", "broke", "missing button", "missing piece",
        "loose thread", "loose threads", "hole", "holes", "stain",
        "stained", "faulty", "malfunction", "arrived damaged",
        "manufacturing defect",
    ],
    "occasion": [
        "wedding", "party", "work", "office", "casual", "formal",
        "everyday", "gym", "workout", "vacation", "holiday",
        "date night", "occasion", "event", "outdoor", "beach",
    ],
    "value": [
        "worth it", "worth the price", "worth the money", "value",
        "overpriced", "affordable", "budget", "price", "prices",
        "priced", "cost", "expensive", "cheap", "cheaply priced",
        "good deal", "bang for your buck",
    ],
}


@dataclass(frozen=True)
class TaxonomyCategory:
    """One CRAGB question category and its target composition.

    `negative_count` is the number of `target_count` questions in this
    category that are deliberately unanswerable from the corpus (the
    correct reference answer is "not enough information") — the fix for
    PLAN.md risk C (random sampling starves negative questions).
    """

    name: str
    description: str
    target_count: int
    negative_count: int

    def __post_init__(self) -> None:
        if self.target_count < 0 or self.negative_count < 0:
            raise ValueError(f"{self.name}: counts must be non-negative")
        if self.negative_count > self.target_count:
            raise ValueError(
                f"{self.name}: negative_count ({self.negative_count}) "
                f"exceeds target_count ({self.target_count})"
            )

    @property
    def keywords(self) -> list[str]:
        try:
            return CATEGORY_KEYWORDS[self.name]
        except KeyError as e:
            raise KeyError(
                f"No keyword list registered for category {self.name!r} "
                f"in CATEGORY_KEYWORDS"
            ) from e


@dataclass(frozen=True)
class TaxonomySpec:
    """The full taxonomy plus the validation thresholds it's checked against."""

    categories: tuple[TaxonomyCategory, ...]
    expected_total: int
    total_tolerance: int
    min_coverage_pct: float
    min_coverage_pct_hard_floor: float

    @property
    def total_target(self) -> int:
        return sum(c.target_count for c in self.categories)

    @property
    def total_negative(self) -> int:
        return sum(c.negative_count for c in self.categories)

    @property
    def keyword_lists(self) -> dict[str, list[str]]:
        return {c.name: c.keywords for c in self.categories}


def load_taxonomy(config_path: str | Path = "configs/taxonomy.yaml") -> TaxonomySpec:
    """Load and validate the taxonomy shape (counts, names) from YAML.

    Does not touch the corpus — that's `check_corpus_coverage`. Raises on
    a malformed config (unknown category referenced twice, negative
    counts, etc.) rather than silently continuing with a broken spec.
    """
    cfg = load_config(config_path)
    validation_cfg = cfg["validation"]

    categories = []
    seen_names: set[str] = set()
    for raw in cfg["categories"]:
        name = raw["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate category name in taxonomy config: {name!r}")
        seen_names.add(name)
        categories.append(
            TaxonomyCategory(
                name=name,
                description=raw["description"].strip(),
                target_count=int(raw["target_count"]),
                negative_count=int(raw["negative_count"]),
            )
        )

    return TaxonomySpec(
        categories=tuple(categories),
        expected_total=int(validation_cfg["expected_total"]),
        total_tolerance=int(validation_cfg["total_tolerance"]),
        min_coverage_pct=float(validation_cfg["min_coverage_pct"]),
        min_coverage_pct_hard_floor=float(validation_cfg["min_coverage_pct_hard_floor"]),
    )


def check_corpus_coverage(
    corpus: pd.DataFrame,
    spec: TaxonomySpec,
    text_col: str = "text",
) -> pd.DataFrame:
    """Measure what fraction of `corpus` mentions each category's keywords.

    Thin wrapper around T1.11's `attribute_vocab_coverage`, reusing its
    tested keyword-matching logic rather than re-implementing it —
    `spec.keyword_lists` is just this taxonomy's categories in the shape
    that function expects.
    """
    return attribute_vocab_coverage(
        corpus, text_col=text_col, keyword_lists=spec.keyword_lists
    )


@dataclass(frozen=True)
class TaxonomyValidation:
    """Result of checking a `TaxonomySpec` against itself and corpus coverage."""

    ok: bool
    total_target: int
    total_negative: int
    negative_share_pct: float
    under_covered: tuple[str, ...]  # soft flag: below min_coverage_pct
    zero_coverage: tuple[str, ...]  # hard fail: below min_coverage_pct_hard_floor
    no_negative_quota: tuple[str, ...]  # hard fail: category has 0 negatives
    messages: tuple[str, ...]


def validate_taxonomy(
    spec: TaxonomySpec, coverage: pd.DataFrame
) -> TaxonomyValidation:
    """Check the taxonomy sums to target and every category is corpus-supported.

    Hard failures (`ok=False`): total count outside tolerance of
    `expected_total`, any category with zero negative-question quota
    (M2.md T2.1 validation: "every category has ≥1 negative-style
    question"), or any category whose keyword coverage is below the hard
    floor (near-zero — the corpus effectively has no evidence for it).

    Soft flags (`ok` unaffected): coverage below `min_coverage_pct` but
    above the hard floor — worth a note in the datasheet, not a blocker.
    """
    messages: list[str] = []

    total_target = spec.total_target
    total_negative = spec.total_negative
    negative_share_pct = (
        round(100 * total_negative / total_target, 2) if total_target else 0.0
    )

    total_low = spec.expected_total - spec.total_tolerance
    total_high = spec.expected_total + spec.total_tolerance
    total_ok = total_low <= total_target <= total_high
    if not total_ok:
        messages.append(
            f"Total target_count {total_target} outside "
            f"[{total_low}, {total_high}] (expected {spec.expected_total} "
            f"+/- {spec.total_tolerance})"
        )

    no_negative_quota = tuple(
        c.name for c in spec.categories if c.negative_count == 0
    )
    if no_negative_quota:
        messages.append(
            f"Categories with no negative-question quota: {list(no_negative_quota)}"
        )

    under_covered: list[str] = []
    zero_coverage: list[str] = []
    for category in spec.categories:
        pct = float(coverage.loc[category.name, "pct_reviews_matched"])
        if pct < spec.min_coverage_pct_hard_floor:
            zero_coverage.append(category.name)
            messages.append(
                f"{category.name}: corpus coverage {pct:.3f}% is below the "
                f"hard floor {spec.min_coverage_pct_hard_floor}% — near-zero "
                f"evidence for this category"
            )
        elif pct < spec.min_coverage_pct:
            under_covered.append(category.name)
            messages.append(
                f"{category.name}: corpus coverage {pct:.2f}% is below the "
                f"{spec.min_coverage_pct}% flag threshold (not a hard failure)"
            )

    ok = total_ok and not no_negative_quota and not zero_coverage
    if ok and not messages:
        messages.append("Taxonomy passes all checks.")

    return TaxonomyValidation(
        ok=ok,
        total_target=total_target,
        total_negative=total_negative,
        negative_share_pct=negative_share_pct,
        under_covered=tuple(under_covered),
        zero_coverage=tuple(zero_coverage),
        no_negative_quota=no_negative_quota,
        messages=tuple(messages),
    )


def render_taxonomy_markdown(
    spec: TaxonomySpec,
    coverage: pd.DataFrame,
    validation: TaxonomyValidation,
) -> str:
    """Render the taxonomy + coverage + validation result as `benchmark/taxonomy.md`.

    This is the T2.1 artifact itself: the definition question-drafting
    (T2.2) works from, plus the evidence that each category is
    corpus-supported.
    """
    lines: list[str] = []
    lines.append("# CRAGB Question Taxonomy (v1)")
    lines.append("")
    lines.append(
        "Defines the 7 categories `cragb_v1` questions are drafted and "
        "balanced against (T2.1; PLAN.md §3 E1), and validates each "
        "category against `corpus_v1`'s attribute-term vocabulary "
        "(T1.11) before any question is written."
    )
    lines.append("")
    lines.append(
        f"**Total target:** {validation.total_target} questions "
        f"(expected {spec.expected_total} +/- {spec.total_tolerance}) · "
        f"**negative/unanswerable quota:** {validation.total_negative} "
        f"({validation.negative_share_pct}%)"
    )
    lines.append("")
    lines.append(
        "| Category | Description | Target | Negative | Corpus coverage | Top terms |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---|"
    )
    for category in spec.categories:
        row = coverage.loc[category.name]
        top_terms = ", ".join(f"{t} ({n})" for t, n in row["top_terms"])
        lines.append(
            f"| `{category.name}` | {category.description} | "
            f"{category.target_count} | {category.negative_count} | "
            f"{row['pct_reviews_matched']:.2f}% | {top_terms} |"
        )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"**Status:** {'PASS' if validation.ok else 'FAIL'}")
    lines.append("")
    for msg in validation.messages:
        lines.append(f"- {msg}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_taxonomy_markdown(
    spec: TaxonomySpec,
    coverage: pd.DataFrame,
    validation: TaxonomyValidation,
    out_path: str | Path = "benchmark/taxonomy.md",
) -> Path:
    """Render and write the taxonomy markdown, creating parent dirs as needed."""
    resolved = resolve_path(out_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        render_taxonomy_markdown(spec, coverage, validation), encoding="utf-8"
    )
    return resolved


def main(config_path: str | Path = "configs/taxonomy.yaml") -> int:
    """Build `benchmark/taxonomy.md` from config + `corpus_v1`, print a summary.

    Returns a process exit code (0 on pass, 1 on validation failure) so
    this can run as a pre-T2.2 gate in CI or a pre-commit check, not just
    a notebook cell.
    """
    cfg = load_config(config_path)
    spec = load_taxonomy(config_path)

    corpus_path = resolve_path(cfg["paths"]["corpus_in"])
    corpus = pd.read_parquet(corpus_path)

    coverage = check_corpus_coverage(corpus, spec)
    validation = validate_taxonomy(spec, coverage)
    out_path = write_taxonomy_markdown(
        spec, coverage, validation, out_path=cfg["paths"]["taxonomy_out"]
    )

    print(f"Wrote {out_path}")
    print(
        f"Total target: {validation.total_target} "
        f"(negative quota: {validation.total_negative}, "
        f"{validation.negative_share_pct}%)"
    )
    for msg in validation.messages:
        print(f"  - {msg}")
    print("PASS" if validation.ok else "FAIL")

    return 0 if validation.ok else 1


if __name__ == "__main__":
    sys.exit(main())
