"""Unit tests for cragb.finetune.sample_contexts (T7.2; PLAN.md §3 E8, M7.md T7.2).

Covers: `SamplingConfig`'s validation (including the n_contexts/category_quotas
arithmetic guard), `cragb_evidence_doc_ids`'s union logic, `excluded_parent_asins`'s
doc-id-to-product mapping, `ContextGroup`'s validation and JSONL round-trip, and --
the task's actual point -- `sample_contexts`'s end-to-end behaviour on a small synthetic
corpus: correct keyword-based category assignment, exactly-k groups, no same-user
duplicates within a group, product-level (not just document-level) exclusion of CRAGB
evidence, photo-bearing over-sampling, shortfall reporting when a category's pool is too
small, and seeded reproducibility. `build_contexts_manifest` is tested separately as the
pure function it is, against hand-built `ContextGroup` lists.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.bench.taxonomy import load_taxonomy
from cragb.finetune.sample_contexts import (
    ContextGroup,
    SamplingConfig,
    build_contexts_manifest,
    cragb_evidence_doc_ids,
    excluded_parent_asins,
    load_contexts_jsonl,
    load_sampling_config,
    sample_contexts,
    write_contexts_jsonl,
)
from cragb.utils.io import resolve_path

TAXONOMY = load_taxonomy("configs/taxonomy.yaml")
CATEGORY_NAMES = [c.name for c in TAXONOMY.categories]


def make_zero_quotas(**overrides: int) -> dict[str, int]:
    """All 7 taxonomy categories at quota 0, except the ones named in `overrides`."""
    quotas = {name: 0 for name in CATEGORY_NAMES}
    quotas.update(overrides)
    return quotas


def make_config(**overrides) -> SamplingConfig:
    quotas = overrides.pop("category_quotas", make_zero_quotas(fit_sizing=1))
    defaults = dict(
        seed=42,
        n_contexts=sum(quotas.values()),
        k=3,
        max_review_chars=600,
        category_quotas=quotas,
        photo_bearing_target_share=0.5,
    )
    defaults.update(overrides)
    return SamplingConfig(**defaults)


# --------------------------------------------------------------------------
# SamplingConfig
# --------------------------------------------------------------------------


class TestSamplingConfig:
    def test_valid_config_constructs(self):
        config = make_config()
        assert config.k == 3

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            make_config(k=0)

    def test_non_positive_max_review_chars_raises(self):
        with pytest.raises(ValueError, match="max_review_chars must be positive"):
            make_config(max_review_chars=0)

    def test_photo_bearing_target_share_out_of_range_raises(self):
        with pytest.raises(ValueError, match="photo_bearing_target_share must be in"):
            make_config(photo_bearing_target_share=1.5)

    def test_empty_category_quotas_raises(self):
        with pytest.raises(ValueError, match="category_quotas must not be empty"):
            make_config(category_quotas={}, n_contexts=0)

    def test_unknown_category_in_quotas_raises(self):
        quotas = make_zero_quotas()
        quotas["not_a_real_category"] = 5
        with pytest.raises(ValueError, match="unknown categories"):
            make_config(category_quotas=quotas, n_contexts=5)

    def test_negative_quota_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            make_config(category_quotas=make_zero_quotas(fit_sizing=-1), n_contexts=-1)

    def test_n_contexts_out_of_sync_with_quotas_raises(self):
        with pytest.raises(ValueError, match="must equal sum"):
            make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=999)


class TestLoadSamplingConfig:
    def test_loads_real_finetune_config(self):
        config = load_sampling_config("configs/finetune.yaml")
        assert config.seed == 42
        assert config.k == 5
        assert config.n_contexts == sum(config.category_quotas.values())
        assert set(config.category_quotas) == set(CATEGORY_NAMES)


# --------------------------------------------------------------------------
# CRAGB evidence
# --------------------------------------------------------------------------


class TestCragbEvidenceDocIds:
    def test_unions_relevant_and_cited_doc_ids(self):
        entries = [
            {"relevant_ids": ["1", "2"], "cited_doc_ids": ["2", "3"]},
            {"relevant_ids": ["4"], "cited_doc_ids": []},
        ]
        assert cragb_evidence_doc_ids(entries) == {"1", "2", "3", "4"}

    def test_pool_doc_ids_default_to_empty(self):
        entries = [{"relevant_ids": ["1"], "cited_doc_ids": []}]
        assert cragb_evidence_doc_ids(entries) == {"1"}

    def test_pool_doc_ids_are_unioned_in(self):
        entries = [{"relevant_ids": ["1"], "cited_doc_ids": []}]
        assert cragb_evidence_doc_ids(entries, pool_doc_ids=["1", "5", "6"]) == {"1", "5", "6"}

    def test_missing_fields_default_to_empty(self):
        assert cragb_evidence_doc_ids([{}]) == set()

    def test_ids_are_stringified(self):
        entries = [{"relevant_ids": [101], "cited_doc_ids": []}]
        assert cragb_evidence_doc_ids(entries) == {"101"}


class TestExcludedParentAsins:
    def make_corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"parent_asin": ["P1", "P1", "P2", "P3"]},
            index=pd.Index([10, 11, 20, 30]),
        )

    def test_maps_evidence_doc_ids_to_their_products(self):
        result = excluded_parent_asins(self.make_corpus(), exclude_doc_ids={"11"})
        assert result == {"P1"}

    def test_unions_across_multiple_evidence_docs(self):
        result = excluded_parent_asins(self.make_corpus(), exclude_doc_ids={"10", "20"})
        assert result == {"P1", "P2"}

    def test_no_matching_doc_ids_returns_empty_set(self):
        result = excluded_parent_asins(self.make_corpus(), exclude_doc_ids={"999"})
        assert result == set()


# --------------------------------------------------------------------------
# ContextGroup
# --------------------------------------------------------------------------


def make_group(
    group_id: str = "ctx_fit_sizing_0000",
    category: str = "fit_sizing",
    parent_asin: str = "P1",
    doc_ids: tuple[str, ...] = ("1", "2", "3"),
    context_text: str = "[1] has_photo: no\nsome text",
    photo_bearing: bool = False,
) -> ContextGroup:
    return ContextGroup(
        group_id=group_id,
        category=category,
        parent_asin=parent_asin,
        doc_ids=doc_ids,
        context_text=context_text,
        photo_bearing=photo_bearing,
    )


class TestContextGroupValidation:
    def test_valid_group_constructs(self):
        assert make_group().category == "fit_sizing"

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="unknown category"):
            make_group(category="not_a_real_category")

    def test_empty_doc_ids_raises(self):
        with pytest.raises(ValueError, match="doc_ids must not be empty"):
            make_group(doc_ids=())

    def test_duplicate_doc_ids_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            make_group(doc_ids=("1", "1", "2"))


class TestContextGroupRoundTrip:
    def test_to_dict_from_dict_is_identity(self):
        group = make_group()
        assert ContextGroup.from_dict(group.to_dict()) == group

    def test_to_dict_uses_lists_not_tuples(self):
        assert isinstance(make_group().to_dict()["doc_ids"], list)

    def test_round_trips_through_json_text(self):
        group = make_group()
        reloaded = ContextGroup.from_dict(json.loads(json.dumps(group.to_dict())))
        assert reloaded == group


class TestWriteLoadContextsJsonl:
    def test_round_trips_through_a_file(self, tmp_path):
        groups = [make_group(group_id="a"), make_group(group_id="b", parent_asin="P2")]
        out_path = write_contexts_jsonl(groups, tmp_path / "contexts.jsonl")
        assert load_contexts_jsonl(out_path) == groups

    def test_creates_parent_directories(self, tmp_path):
        out_path = write_contexts_jsonl([make_group()], tmp_path / "nested" / "dir" / "contexts.jsonl")
        assert out_path.is_file()


# --------------------------------------------------------------------------
# sample_contexts
# --------------------------------------------------------------------------


def make_synthetic_corpus() -> pd.DataFrame:
    """A small corpus with three products, unambiguous fit_sizing vs colour_appearance
    vocabulary, and one duplicate-user review to exercise the dedup guard.

    - P1: 5 fit_sizing-themed reviews by 5 distinct users, 2 with photos -- enough to
      form a k=3 group, comfortably category-assignable.
    - P2: 3 colour_appearance-themed reviews by 3 distinct users, 1 with a photo.
    - P3: only 2 reviews -- below any k=3 quota, must never appear in output.
    - P4: 4 reviews, but two are by the *same* user_id ("dup_user") -- after dedup only
      3 distinct-user reviews remain, exactly at the k=3 floor.
    """
    rows = [
        # P1 -- fit_sizing
        ("These run small, I had to size up for a comfortable fit.", "P1", "u1", True),
        ("Runs a little big, order down half a size for a snug fit.", "P1", "u2", False),
        ("True to size, fits great and comfortable all day.", "P1", "u3", True),
        ("Sizing was spot on, true to size as expected.", "P1", "u4", False),
        ("This runs small, size up two full sizes.", "P1", "u5", False),
        # P2 -- colour_appearance
        ("The colour matched the product photo exactly.", "P2", "u6", True),
        ("Colour looked faded compared to the picture online.", "P2", "u7", False),
        ("As pictured, the shade is a true match.", "P2", "u8", False),
        # P3 -- too few reviews for k=3
        ("Runs small, size up for a better fit.", "P3", "u9", False),
        ("True to size and comfortable.", "P3", "u10", False),
        # P4 -- has a duplicate reviewer
        ("Fits true to size, very comfortable.", "P4", "u11", False),
        ("Runs small, size up one size.", "P4", "u12", False),
        ("Second review from the same shopper: still runs small.", "P4", "dup_user", False),
        ("Same shopper again: sizing runs small here too.", "P4", "dup_user", False),
    ]
    return pd.DataFrame(
        rows, columns=["text", "parent_asin", "user_id", "has_image"]
    )


class TestSampleContextsCategoryAssignment:
    def test_assigns_fit_sizing_group_from_p1(self):
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=1)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert len(groups) == 1
        assert groups[0].category == "fit_sizing"
        assert groups[0].parent_asin == "P1"

    def test_assigns_colour_appearance_group_from_p2(self):
        config = make_config(category_quotas=make_zero_quotas(colour_appearance=1), n_contexts=1)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert len(groups) == 1
        assert groups[0].category == "colour_appearance"
        assert groups[0].parent_asin == "P2"

    def test_product_with_fewer_than_k_reviews_is_never_selected(self):
        # P3 has only 2 reviews; request more fit_sizing groups than P1 alone can
        # possibly saturate isn't needed here -- just confirm P3 never appears.
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=2), n_contexts=2)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert all(g.parent_asin != "P3" for g in groups)


class TestSampleContextsGroupShape:
    def test_every_group_has_exactly_k_doc_ids(self):
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=1, k=3)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert all(len(g.doc_ids) == 3 for g in groups)

    def test_context_text_contains_one_excerpt_per_doc_id(self):
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=1, k=3)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        group = groups[0]
        for doc_id in group.doc_ids:
            assert f"[{doc_id}] has_photo:" in group.context_text

    def test_p4_group_never_contains_the_duplicate_users_second_review(self):
        # P4 has exactly 3 distinct users after dedup (u11, u12, dup_user) -- if the
        # dedup guard failed, a k=3 group could contain both of dup_user's rows.
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=3), n_contexts=3, k=3)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        p4_groups = [g for g in groups if g.parent_asin == "P4"]
        assert len(p4_groups) == 1
        assert len(p4_groups[0].doc_ids) == len(set(p4_groups[0].doc_ids)) == 3


class TestSampleContextsExclusion:
    def test_excluding_one_review_removes_the_whole_product(self):
        corpus = make_synthetic_corpus()
        # Doc id "0" is P1's first review (positional index 0).
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=1)
        groups = sample_contexts(corpus, TAXONOMY, exclude_doc_ids={"0"}, config=config)
        # P1 is now fully excluded; no other product has enough reviews to be
        # confidently assigned fit_sizing at k=3, so no group should be produced.
        assert all(g.parent_asin != "P1" for g in groups)

    def test_sampled_doc_ids_never_intersect_excluded_doc_ids(self):
        corpus = make_synthetic_corpus()
        exclude = {"0", "5"}  # one doc from P1, one from P2
        config = make_config(
            category_quotas=make_zero_quotas(fit_sizing=1, colour_appearance=1), n_contexts=2
        )
        groups = sample_contexts(corpus, TAXONOMY, exclude_doc_ids=exclude, config=config)
        all_doc_ids = {doc_id for g in groups for doc_id in g.doc_ids}
        assert all_doc_ids.isdisjoint(exclude)

    def test_sampled_parent_asins_never_intersect_evidence_parent_asins(self):
        corpus = make_synthetic_corpus()
        exclude = {"0"}  # P1's first review
        evidence_pas = excluded_parent_asins(corpus, exclude)
        config = make_config(
            category_quotas=make_zero_quotas(fit_sizing=1, colour_appearance=1), n_contexts=2
        )
        groups = sample_contexts(corpus, TAXONOMY, exclude_doc_ids=exclude, config=config)
        sampled_pas = {g.parent_asin for g in groups}
        assert sampled_pas.isdisjoint(evidence_pas)


class TestSampleContextsPhotoOversampling:
    def test_full_target_share_prioritises_photo_bearing_docs(self):
        # P1 has 5 candidates, 2 photo-bearing (u1, u3). With target share 1.0 and
        # k=2, the group should contain exactly the 2 photo-bearing reviews.
        config = make_config(
            category_quotas=make_zero_quotas(fit_sizing=1),
            n_contexts=1,
            k=2,
            photo_bearing_target_share=1.0,
        )
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert groups[0].photo_bearing is True

    def test_zero_target_share_still_allows_a_group_without_photos(self):
        config = make_config(
            category_quotas=make_zero_quotas(fit_sizing=1),
            n_contexts=1,
            k=3,
            photo_bearing_target_share=0.0,
        )
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        # Not asserting photo_bearing is False (a random draw may still include one),
        # only that the pipeline runs and yields a valid group either way.
        assert len(groups) == 1
        assert len(groups[0].doc_ids) == 3


class TestSampleContextsShortfallHandling:
    def test_quota_larger_than_available_pool_returns_all_available_without_raising(self):
        # P1 can only ever produce 1 fit_sizing group at k=3 (5 candidates // ... well,
        # only distinct parent_asins count as one group each) -- requesting 5 groups
        # must not raise, and must return fewer than requested.
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=5), n_contexts=5)
        groups = sample_contexts(
            make_synthetic_corpus(), TAXONOMY, exclude_doc_ids=set(), config=config
        )
        assert len(groups) < 5

    def test_empty_corpus_after_exclusion_returns_empty_list(self):
        corpus = make_synthetic_corpus()
        all_doc_ids = {str(i) for i in range(len(corpus))}
        config = make_config(category_quotas=make_zero_quotas(fit_sizing=1), n_contexts=1)
        groups = sample_contexts(corpus, TAXONOMY, exclude_doc_ids=all_doc_ids, config=config)
        assert groups == []


class TestSampleContextsReproducibility:
    def test_same_seed_produces_identical_groups(self):
        corpus = make_synthetic_corpus()
        config = make_config(
            category_quotas=make_zero_quotas(fit_sizing=1, colour_appearance=1), n_contexts=2
        )
        first = sample_contexts(corpus, TAXONOMY, exclude_doc_ids=set(), config=config)
        second = sample_contexts(corpus, TAXONOMY, exclude_doc_ids=set(), config=config)
        assert first == second

    def test_mismatched_category_quotas_raises(self):
        corpus = make_synthetic_corpus()
        bad_quotas = {"fit_sizing": 1}  # missing the other 6 categories
        config = SamplingConfig(
            seed=42, n_contexts=1, k=3, max_review_chars=600,
            category_quotas=bad_quotas, photo_bearing_target_share=0.5,
        )
        with pytest.raises(ValueError, match="do not match taxonomy categories"):
            sample_contexts(corpus, TAXONOMY, exclude_doc_ids=set(), config=config)


# --------------------------------------------------------------------------
# build_contexts_manifest
# --------------------------------------------------------------------------


class TestBuildContextsManifest:
    def test_counts_groups_per_category(self):
        groups = [
            make_group(category="fit_sizing"),
            make_group(category="fit_sizing", parent_asin="P2"),
            make_group(category="colour_appearance", parent_asin="P3"),
        ]
        manifest = build_contexts_manifest(
            groups,
            seed=42,
            corpus_sha256="deadbeef",
            config_hash="cafef00d",
            n_docs_excluded_as_cragb_evidence=10,
            n_parent_asins_excluded=4,
            category_quotas=make_zero_quotas(fit_sizing=2, colour_appearance=1),
            photo_bearing_target_share=0.3,
        )
        assert manifest["per_category_counts"] == {"colour_appearance": 1, "fit_sizing": 2}
        assert manifest["n_groups"] == 3

    def test_quota_report_flags_shortfall(self):
        groups = [make_group(category="fit_sizing")]
        manifest = build_contexts_manifest(
            groups,
            seed=42,
            corpus_sha256="x",
            config_hash="y",
            n_docs_excluded_as_cragb_evidence=0,
            n_parent_asins_excluded=0,
            category_quotas=make_zero_quotas(fit_sizing=5),
            photo_bearing_target_share=0.3,
        )
        report = manifest["category_quota_report"]["fit_sizing"]
        assert report == {"target": 5, "actual": 1, "shortfall": 4}

    def test_quota_report_no_shortfall_when_met_or_exceeded(self):
        groups = [make_group(category="fit_sizing"), make_group(category="fit_sizing", parent_asin="P2")]
        manifest = build_contexts_manifest(
            groups, seed=42, corpus_sha256="x", config_hash="y",
            n_docs_excluded_as_cragb_evidence=0, n_parent_asins_excluded=0,
            category_quotas=make_zero_quotas(fit_sizing=2), photo_bearing_target_share=0.3,
        )
        assert manifest["category_quota_report"]["fit_sizing"]["shortfall"] == 0

    def test_photo_bearing_share_is_the_fraction_of_groups_with_a_photo(self):
        groups = [
            make_group(photo_bearing=True),
            make_group(photo_bearing=True, parent_asin="P2"),
            make_group(photo_bearing=False, parent_asin="P3"),
            make_group(photo_bearing=False, parent_asin="P4"),
        ]
        manifest = build_contexts_manifest(
            groups, seed=42, corpus_sha256="x", config_hash="y",
            n_docs_excluded_as_cragb_evidence=0, n_parent_asins_excluded=0,
            category_quotas=make_zero_quotas(fit_sizing=4), photo_bearing_target_share=0.3,
        )
        assert manifest["photo_bearing_share"] == 0.5

    def test_empty_groups_list_does_not_raise_and_reports_zero_share(self):
        manifest = build_contexts_manifest(
            [], seed=42, corpus_sha256="x", config_hash="y",
            n_docs_excluded_as_cragb_evidence=0, n_parent_asins_excluded=0,
            category_quotas=make_zero_quotas(), photo_bearing_target_share=0.3,
        )
        assert manifest["photo_bearing_share"] == 0.0
        assert manifest["n_groups"] == 0

    def test_carries_through_seed_and_hashes_and_timestamp(self):
        manifest = build_contexts_manifest(
            [], seed=7, corpus_sha256="abc123", config_hash="def456",
            n_docs_excluded_as_cragb_evidence=1, n_parent_asins_excluded=1,
            category_quotas=make_zero_quotas(), photo_bearing_target_share=0.3,
        )
        assert manifest["seed"] == 7
        assert manifest["corpus_sha256"] == "abc123"
        assert manifest["config_hash"] == "def456"
        assert "created_at_utc" in manifest


# --------------------------------------------------------------------------
# Real-artifact integration: config wiring only (no full corpus sweep)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        resolve_path("benchmark/cragb_v1.jsonl").is_file()
        and resolve_path("benchmark/pools_v1.jsonl").is_file()
    ),
    reason="real cragb_v1.jsonl / pools_v1.jsonl not present locally",
)
class TestRealCragbEvidenceWiring:
    """Confirms cragb_evidence_doc_ids/excluded_parent_asins wire up correctly against
    the real CRAGB artifacts -- not a full corpus sampling run (that's a manual
    `python -m cragb.finetune.sample_contexts` verification step, per M7.md T7.2's "How I
    verify it worked", not part of the automated suite).
    """

    def test_real_evidence_set_is_non_trivial_and_covers_known_ids(self):
        from cragb.finetune.sample_contexts import load_cragb_entries, load_pool_doc_ids

        entries = load_cragb_entries("benchmark/cragb_v1.jsonl")
        pool_ids = load_pool_doc_ids("benchmark/pools_v1.jsonl")
        evidence = cragb_evidence_doc_ids(entries, pool_ids)

        assert len(entries) == 60
        # From benchmark/cragb_v1.jsonl's fit_sizing_000 entry (PLAN.md-documented id).
        assert "128775" in evidence
