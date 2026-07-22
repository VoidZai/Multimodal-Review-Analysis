"""Unit tests for cragb.bench.taxonomy.

Covers: TaxonomyCategory's own invariants (counts non-negative, negatives
don't exceed target), that the real `configs/taxonomy.yaml` loads into a
well-formed spec summing to ~60 with every category carrying a negative
quota (M2.md T2.1's own validation bar), that corpus-coverage checking
correctly flags a keyword-starved category, that `validate_taxonomy`
distinguishes hard failures from soft flags, and that the rendered
Markdown contains the sections the T2.1 artifact is required to have.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.bench.taxonomy import (
    CATEGORY_KEYWORDS,
    TaxonomyCategory,
    TaxonomySpec,
    check_corpus_coverage,
    load_taxonomy,
    render_taxonomy_markdown,
    validate_taxonomy,
)


def make_spec(**overrides) -> TaxonomySpec:
    categories = overrides.pop("categories", None) or (
        TaxonomyCategory("fit_sizing", "fit desc", 10, 2),
        TaxonomyCategory("colour_appearance", "colour desc", 10, 1),
    )
    defaults = dict(
        categories=categories,
        expected_total=20,
        total_tolerance=2,
        min_coverage_pct=1.0,
        min_coverage_pct_hard_floor=0.05,
    )
    defaults.update(overrides)
    return TaxonomySpec(**defaults)


class TestTaxonomyCategory:
    def test_valid_category_constructs(self):
        c = TaxonomyCategory("fit_sizing", "desc", target_count=10, negative_count=2)
        assert c.target_count == 10
        assert c.negative_count == 2

    def test_negative_count_exceeding_target_raises(self):
        with pytest.raises(ValueError, match="exceeds target_count"):
            TaxonomyCategory("fit_sizing", "desc", target_count=2, negative_count=3)

    @pytest.mark.parametrize("target,negative", [(-1, 0), (5, -1)])
    def test_negative_input_counts_raise(self, target, negative):
        with pytest.raises(ValueError, match="non-negative"):
            TaxonomyCategory("x", "desc", target_count=target, negative_count=negative)

    def test_keywords_resolve_from_registry(self):
        c = TaxonomyCategory("fit_sizing", "desc", target_count=1, negative_count=0)
        assert c.keywords == CATEGORY_KEYWORDS["fit_sizing"]

    def test_unknown_category_name_raises_on_keyword_lookup(self):
        c = TaxonomyCategory("not_a_real_category", "desc", target_count=1, negative_count=0)
        with pytest.raises(KeyError):
            _ = c.keywords


class TestLoadTaxonomy:
    def test_real_config_loads_and_sums_near_60(self):
        spec = load_taxonomy("configs/taxonomy.yaml")
        assert spec.total_target == 60
        low, high = spec.expected_total - spec.total_tolerance, spec.expected_total + spec.total_tolerance
        assert low <= spec.total_target <= high

    def test_every_category_has_a_negative_quota(self):
        spec = load_taxonomy("configs/taxonomy.yaml")
        assert all(c.negative_count >= 1 for c in spec.categories)

    def test_every_category_has_a_registered_keyword_list(self):
        spec = load_taxonomy("configs/taxonomy.yaml")
        for category in spec.categories:
            assert len(category.keywords) > 0

    def test_duplicate_category_name_raises(self, tmp_path, monkeypatch):
        bad_config = tmp_path / "bad_taxonomy.yaml"
        bad_config.write_text(
            """
seed: 42
paths:
  corpus_in: data/processed/corpus_v1.parquet
  taxonomy_out: benchmark/taxonomy.md
validation:
  expected_total: 2
  total_tolerance: 0
  min_coverage_pct: 1.0
  min_coverage_pct_hard_floor: 0.05
categories:
  - name: fit_sizing
    description: "a"
    target_count: 1
    negative_count: 0
  - name: fit_sizing
    description: "b"
    target_count: 1
    negative_count: 0
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate category"):
            load_taxonomy(bad_config)


class TestCheckCorpusCoverage:
    def test_matches_only_categories_with_present_terms(self):
        df = pd.DataFrame(
            {
                "text": [
                    "This runs small, size up.",
                    "The colour faded after one wash.",
                    "Completely unrelated text about nothing in particular.",
                ]
            }
        )
        spec = make_spec()
        coverage = check_corpus_coverage(df, spec, text_col="text")
        assert coverage.loc["fit_sizing", "n_reviews_matched"] == 1
        assert coverage.loc["colour_appearance", "n_reviews_matched"] == 1

    def test_returns_a_row_per_category(self):
        df = pd.DataFrame({"text": ["true to size"]})
        spec = make_spec()
        coverage = check_corpus_coverage(df, spec, text_col="text")
        assert set(coverage.index) == {c.name for c in spec.categories}


class TestValidateTaxonomy:
    def _coverage_for(self, spec: TaxonomySpec, pct_by_category: dict[str, float]) -> pd.DataFrame:
        rows = []
        for category in spec.categories:
            pct = pct_by_category.get(category.name, 5.0)
            rows.append(
                {
                    "category": category.name,
                    "n_reviews_matched": 1,
                    "pct_reviews_matched": pct,
                    "n_keyword_hits_total": 1,
                    "top_terms": [("term", 1)],
                }
            )
        return pd.DataFrame(rows).set_index("category")

    def test_well_formed_spec_passes(self):
        spec = make_spec()
        coverage = self._coverage_for(spec, {})
        result = validate_taxonomy(spec, coverage)
        assert result.ok
        assert result.zero_coverage == ()
        assert result.no_negative_quota == ()

    def test_total_outside_tolerance_fails(self):
        spec = make_spec(expected_total=100, total_tolerance=1)
        coverage = self._coverage_for(spec, {})
        result = validate_taxonomy(spec, coverage)
        assert not result.ok
        assert any("outside" in m for m in result.messages)

    def test_zero_negative_quota_category_fails(self):
        categories = (
            TaxonomyCategory("fit_sizing", "d", 10, 0),
            TaxonomyCategory("colour_appearance", "d", 10, 1),
        )
        spec = make_spec(categories=categories)
        coverage = self._coverage_for(spec, {})
        result = validate_taxonomy(spec, coverage)
        assert not result.ok
        assert "fit_sizing" in result.no_negative_quota

    def test_near_zero_coverage_is_a_hard_failure(self):
        spec = make_spec()
        coverage = self._coverage_for(spec, {"fit_sizing": 0.0})
        result = validate_taxonomy(spec, coverage)
        assert not result.ok
        assert "fit_sizing" in result.zero_coverage

    def test_below_flag_threshold_but_above_floor_is_a_soft_flag(self):
        spec = make_spec()
        coverage = self._coverage_for(spec, {"fit_sizing": 0.5})
        result = validate_taxonomy(spec, coverage)
        assert result.ok  # soft flag must not fail validation
        assert "fit_sizing" in result.under_covered
        assert "fit_sizing" not in result.zero_coverage

    def test_negative_share_pct_computed_correctly(self):
        spec = make_spec()  # totals: target 20, negatives 3
        coverage = self._coverage_for(spec, {})
        result = validate_taxonomy(spec, coverage)
        assert result.total_negative == 3
        assert result.negative_share_pct == pytest.approx(15.0)


class TestRenderTaxonomyMarkdown:
    def test_markdown_contains_required_sections(self):
        spec = make_spec()
        coverage = TestValidateTaxonomy()._coverage_for(spec, {})
        validation = validate_taxonomy(spec, coverage)
        md = render_taxonomy_markdown(spec, coverage, validation)

        assert "# CRAGB Question Taxonomy" in md
        assert "## Validation" in md
        assert "fit_sizing" in md
        assert "colour_appearance" in md
        assert "PASS" in md or "FAIL" in md

    def test_markdown_reflects_fail_status(self):
        categories = (TaxonomyCategory("fit_sizing", "d", 10, 0),)
        spec = make_spec(categories=categories, expected_total=10, total_tolerance=0)
        coverage = TestValidateTaxonomy()._coverage_for(spec, {})
        validation = validate_taxonomy(spec, coverage)
        md = render_taxonomy_markdown(spec, coverage, validation)
        assert "FAIL" in md
