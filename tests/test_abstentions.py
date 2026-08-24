"""Unit tests for cragb.finetune.abstentions (T7.4; PLAN.md §3 E8, M7.md T7.4).

Covers: `AbstentionConfig`'s validation, `overlap_ratio`'s content-word overlap
computation, `_split_context_blocks`'s marker-based excerpt recovery (including the
embedded-blank-line risk a naive `"\\n\\n"` split would mishandle), each of the three
construction methods in isolation (via `build_abstentions` restricted to one method at a
time), and the orchestration-level guarantees: per-method quota split, total-target
sizing toward `config.target_share`, graceful shortfall handling, and the
cross-cutting invariant that no abstention's context contains any doc its source
positive's answer cited.
"""

from __future__ import annotations

import json

import pytest

from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.finetune.abstentions import (
    CATEGORICAL_ABSENCE_QUESTIONS,
    AbstentionConfig,
    _split_context_blocks,
    build_abstentions,
    load_abstention_config,
    overlap_ratio,
)
from cragb.finetune.sample_contexts import ContextGroup
from cragb.finetune.schema import VALID_CATEGORIES, TrainingExample


def make_context(
    group_id: str = "ctx_fit_sizing_0000",
    category: str = "fit_sizing",
    parent_asin: str = "P1",
    doc_ids: tuple[str, ...] = ("1", "2", "3"),
    context_text: str | None = None,
    photo_bearing: bool = False,
) -> ContextGroup:
    if context_text is None:
        blocks = [f"[{d}] has_photo: no\nReview text for doc {d} in {group_id}." for d in doc_ids]
        context_text = "\n\n".join(blocks)
    return ContextGroup(
        group_id=group_id,
        category=category,
        parent_asin=parent_asin,
        doc_ids=doc_ids,
        context_text=context_text,
        photo_bearing=photo_bearing,
    )


def make_positive(
    example_id: str = "ctx_fit_sizing_0000_00",
    category: str = "fit_sizing",
    context_group_id: str = "ctx_fit_sizing_0000",
    source_doc_ids: tuple[str, ...] = ("1", "2", "3"),
    parent_asin: str = "P1",
    question: str = "Does this run small?",
    context_text: str | None = None,
    answer: str = "Yes, it runs small [1]. It is true to size for others [2].",
    cited_doc_ids: tuple[str, ...] = ("1", "2"),
    provenance_extra: dict | None = None,
) -> TrainingExample:
    if context_text is None:
        blocks = [f"[{d}] has_photo: no\nReview text for doc {d} in {context_group_id}." for d in source_doc_ids]
        context_text = "\n\n".join(blocks)
    provenance = {"method": "teacher_generation", "context_group_id": context_group_id}
    if provenance_extra:
        provenance.update(provenance_extra)
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=source_doc_ids,
        source_parent_asins=(parent_asin,),
        question=question,
        context_text=context_text,
        answer=answer,
        cited_doc_ids=cited_doc_ids,
        is_abstention=False,
        provenance=provenance,
    )


def make_config(**overrides) -> AbstentionConfig:
    defaults = dict(
        seed=42,
        target_share=0.5,
        methods=("transplant", "categorical_absence", "evidence_stripped"),
        transplant_overlap_threshold=0.15,
        evidence_stripped_min_remaining_docs=1,
    )
    defaults.update(overrides)
    return AbstentionConfig(**defaults)


# --------------------------------------------------------------------------
# AbstentionConfig
# --------------------------------------------------------------------------


class TestAbstentionConfig:
    def test_valid_config_constructs(self):
        assert make_config().target_share == 0.5

    def test_target_share_of_one_raises(self):
        with pytest.raises(ValueError, match="target_share must be in"):
            make_config(target_share=1.0)

    def test_negative_target_share_raises(self):
        with pytest.raises(ValueError, match="target_share must be in"):
            make_config(target_share=-0.1)

    def test_empty_methods_raises(self):
        with pytest.raises(ValueError, match="methods must not be empty"):
            make_config(methods=())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown abstention method"):
            make_config(methods=("transplant", "not_a_real_method"))

    def test_duplicate_methods_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            make_config(methods=("transplant", "transplant"))

    def test_overlap_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="transplant_overlap_threshold must be in"):
            make_config(transplant_overlap_threshold=1.5)

    def test_negative_min_remaining_docs_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            make_config(evidence_stripped_min_remaining_docs=-1)


class TestLoadAbstentionConfig:
    def test_loads_real_finetune_config(self):
        config = load_abstention_config("configs/finetune.yaml")
        assert config.seed == 42
        assert set(config.methods) == {"transplant", "categorical_absence", "evidence_stripped"}
        assert 0.0 <= config.target_share < 1.0


# --------------------------------------------------------------------------
# overlap_ratio
# --------------------------------------------------------------------------


class TestOverlapRatio:
    def test_no_overlap_is_zero(self):
        assert overlap_ratio("Does this run small?", "Colour matched the photo exactly.") == 0.0

    def test_full_overlap_is_one(self):
        # overlap_ratio matches exact word forms (no stemming) -- "run" in the question
        # must appear as "run", not "runs", in the text for this to be a full match.
        assert overlap_ratio("Does this run small?", "It will run small for sure.") == 1.0

    def test_partial_overlap_is_a_fraction(self):
        # content words of "Does this run small?" -> {"run"->no (only "run" as substring
        # of "runs" via word-boundary won't match different form)... use exact words}
        ratio = overlap_ratio("Is the fabric soft and durable?", "The fabric feels soft.")
        assert 0.0 < ratio < 1.0

    def test_stopwords_are_ignored(self):
        # "Is this the one?" has no content words after stopword filtering.
        assert overlap_ratio("Is this the one?", "Completely unrelated text about shoes.") == 0.0

    def test_matching_is_word_boundary_not_substring(self):
        # "cat" must not match inside "category" or "concatenate".
        assert overlap_ratio("Where is the cat?", "This concatenates strings by category.") == 0.0

    def test_case_insensitive(self):
        assert overlap_ratio("Does this RUN small?", "it will RUN small for sure") == 1.0


# --------------------------------------------------------------------------
# _split_context_blocks
# --------------------------------------------------------------------------


class TestSplitContextBlocks:
    def test_recovers_each_block_in_order(self):
        context = make_context(doc_ids=("1", "2", "3"))
        blocks = _split_context_blocks(context.context_text, context.doc_ids)
        assert set(blocks) == {"1", "2", "3"}
        for doc_id, block in blocks.items():
            assert block.startswith(f"[{doc_id}] has_photo:")

    def test_survives_a_blank_line_embedded_in_one_snippet(self):
        # A naive "\n\n" split would misattribute text across this blank line.
        context_text = (
            "[1] has_photo: no\nFirst paragraph.\n\nSecond paragraph, same review.\n\n"
            "[2] has_photo: no\nA totally different review."
        )
        blocks = _split_context_blocks(context_text, ("1", "2"))
        assert blocks["1"] == "[1] has_photo: no\nFirst paragraph.\n\nSecond paragraph, same review."
        assert blocks["2"] == "[2] has_photo: no\nA totally different review."

    def test_raises_when_a_marker_is_missing(self):
        context = make_context(doc_ids=("1", "2"))
        with pytest.raises(ValueError, match="not found"):
            _split_context_blocks(context.context_text, ("1", "999"))

    def test_single_doc_block(self):
        context = make_context(doc_ids=("1",))
        blocks = _split_context_blocks(context.context_text, ("1",))
        assert blocks["1"] == context.context_text


# --------------------------------------------------------------------------
# transplant (via build_abstentions restricted to one method)
# --------------------------------------------------------------------------


class TestTransplant:
    def _config(self, **kw):
        return make_config(methods=("transplant",), **kw)

    def test_pairs_question_with_a_different_product_and_category(self):
        ctx_a = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1", doc_ids=("1", "2", "3"))
        ctx_b = make_context(
            group_id="ctx_colour_appearance_0000",
            category="colour_appearance",
            parent_asin="P2",
            doc_ids=("10", "11", "12"),
            context_text="[10] has_photo: no\nColour matched the photo exactly.\n\n"
            "[11] has_photo: no\nShade looked faded.\n\n[12] has_photo: yes\nTrue to the picture.",
        )
        positive = make_positive(context_group_id="ctx_fit_sizing_0000", question="Does this run small?")
        config = self._config(target_share=0.5, transplant_overlap_threshold=0.5)

        abstentions = build_abstentions([positive], [ctx_a, ctx_b], config)
        assert len(abstentions) == 1
        example = abstentions[0]
        assert example.provenance["method"] == "transplant"
        assert example.context_text == ctx_b.context_text
        assert example.category == positive.category  # tagged by the question's own category
        assert example.source_parent_asins == ("P2",)
        assert example.question == "Does this run small?"

    def test_overlapping_pairing_is_rejected(self):
        ctx_a = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1", doc_ids=("1",))
        # ctx_b is a different product/category but its text is deliberately full of the
        # same content words as the question -- a real answerability leak if transplanted.
        ctx_b = make_context(
            group_id="ctx_colour_appearance_0000",
            category="colour_appearance",
            parent_asin="P2",
            doc_ids=("10",),
            context_text="[10] has_photo: no\nThis definitely runs small, everyone should size up.",
        )
        positive = make_positive(context_group_id="ctx_fit_sizing_0000", question="Does this run small?")
        config = self._config(target_share=0.5, transplant_overlap_threshold=0.15)

        abstentions = build_abstentions([positive], [ctx_a, ctx_b], config)
        assert abstentions == []

    def test_positive_missing_context_group_id_is_skipped(self):
        ctx_a = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1")
        ctx_b = make_context(group_id="ctx_colour_appearance_0000", category="colour_appearance", parent_asin="P2")
        positive = make_positive(context_group_id="ctx_fit_sizing_0000")
        object.__setattr__(positive, "provenance", {"method": "teacher_generation"})  # no context_group_id

        config = self._config(target_share=0.5)
        assert build_abstentions([positive], [ctx_a, ctx_b], config) == []

    def test_no_eligible_target_context_yields_nothing(self):
        # Only one context exists at all -- there is no *different* product/category to
        # transplant into.
        ctx_a = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1")
        positive = make_positive(context_group_id="ctx_fit_sizing_0000")
        config = self._config(target_share=0.5)
        assert build_abstentions([positive], [ctx_a], config) == []

    def test_at_most_one_transplant_per_source_positive(self):
        ctx_a = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1")
        ctx_b = make_context(group_id="ctx_colour_appearance_0000", category="colour_appearance", parent_asin="P2")
        ctx_c = make_context(group_id="ctx_defects_0000", category="defects", parent_asin="P3")
        positive = make_positive(context_group_id="ctx_fit_sizing_0000")
        # target_share high enough to *want* more than 1, but there's only 1 positive.
        config = self._config(target_share=0.9)
        abstentions = build_abstentions([positive], [ctx_a, ctx_b, ctx_c], config)
        assert len(abstentions) == 1


# --------------------------------------------------------------------------
# categorical_absence
# --------------------------------------------------------------------------


class TestCategoricalAbsence:
    def _config(self, **kw):
        return make_config(methods=("categorical_absence",), **kw)

    def test_question_and_category_are_consistent(self):
        ctx = make_context(group_id="ctx_defects_0000", category="defects", parent_asin="P1")
        positive = make_positive(context_group_id="ctx_defects_0000", category="defects")
        config = self._config(target_share=0.5)

        abstentions = build_abstentions([positive], [ctx], config)
        assert len(abstentions) == 1
        example = abstentions[0]
        assert example.category == "defects"
        assert example.question in CATEGORICAL_ABSENCE_QUESTIONS["defects"]
        assert example.context_text == ctx.context_text

    def test_never_uses_a_category_with_no_available_context(self):
        # Only a fit_sizing context exists -- every produced example must be fit_sizing,
        # never one of the other 6 categories' questions.
        ctx = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1")
        positive = make_positive(context_group_id="ctx_fit_sizing_0000")
        config = self._config(target_share=0.9)

        abstentions = build_abstentions([positive], [ctx], config)
        assert all(e.category == "fit_sizing" for e in abstentions)
        assert all(e.question in CATEGORICAL_ABSENCE_QUESTIONS["fit_sizing"] for e in abstentions)

    def test_covers_every_taxonomy_category(self):
        assert set(CATEGORICAL_ABSENCE_QUESTIONS) == set(VALID_CATEGORIES)

    def test_no_positives_at_all_yields_nothing(self):
        # build_abstentions short-circuits on empty positives regardless of method.
        ctx = make_context()
        config = self._config(target_share=0.5)
        assert build_abstentions([], [ctx], config) == []


# --------------------------------------------------------------------------
# evidence_stripped
# --------------------------------------------------------------------------


class TestEvidenceStripped:
    def _config(self, **kw):
        return make_config(methods=("evidence_stripped",), **kw)

    def test_strips_exactly_the_cited_docs(self):
        ctx = make_context(group_id="ctx_fit_sizing_0000", doc_ids=("1", "2", "3"))
        positive = make_positive(
            context_group_id="ctx_fit_sizing_0000",
            source_doc_ids=("1", "2", "3"),
            cited_doc_ids=("1", "2"),
            question="Does this run small?",
            context_text=ctx.context_text,
        )
        config = self._config(target_share=0.5, evidence_stripped_min_remaining_docs=1)

        abstentions = build_abstentions([positive], [ctx], config)
        assert len(abstentions) == 1
        example = abstentions[0]
        assert example.source_doc_ids == ("3",)
        assert "[1]" not in example.context_text
        assert "[2]" not in example.context_text
        assert "[3]" in example.context_text
        assert example.provenance["n_docs_removed"] == 2

    def test_min_remaining_docs_guard_rejects_a_fully_stripped_context(self):
        ctx = make_context(group_id="ctx_fit_sizing_0000", doc_ids=("1", "2"))
        positive = make_positive(
            context_group_id="ctx_fit_sizing_0000",
            source_doc_ids=("1", "2"),
            cited_doc_ids=("1", "2"),  # cites everything -- nothing would remain
            context_text=ctx.context_text,
        )
        config = self._config(target_share=0.5, evidence_stripped_min_remaining_docs=1)
        assert build_abstentions([positive], [ctx], config) == []

    def test_residual_overlap_guard_rejects_when_leftover_docs_still_answer(self):
        # Doc 3 is never cited, but it independently discusses the exact same topic as
        # the question -- stripping only the cited docs would leave an answerable
        # context, which the residual overlap check must catch.
        ctx = make_context(
            group_id="ctx_fit_sizing_0000",
            doc_ids=("1", "2", "3"),
            context_text="[1] has_photo: no\nRuns small, size up.\n\n"
            "[2] has_photo: no\nTrue to size for me.\n\n"
            "[3] has_photo: no\nThis definitely runs small too, so size up.",
        )
        positive = make_positive(
            context_group_id="ctx_fit_sizing_0000",
            source_doc_ids=("1", "2", "3"),
            cited_doc_ids=("1", "2"),
            question="Does this run small?",
            context_text=ctx.context_text,
        )
        config = self._config(target_share=0.5, evidence_stripped_min_remaining_docs=1, transplant_overlap_threshold=0.15)
        assert build_abstentions([positive], [ctx], config) == []

    def test_positive_with_no_citations_is_skipped(self):
        # An unusual but valid "positive" shape: a non-abstention answer that just
        # didn't cite anything. evidence_stripped has nothing to strip either way.
        ctx = make_context(group_id="ctx_fit_sizing_0000", doc_ids=("1", "2"))
        positive = make_positive(
            context_group_id="ctx_fit_sizing_0000",
            source_doc_ids=("1", "2"),
            cited_doc_ids=(),
            answer="A generic uncited answer.",
            context_text=ctx.context_text,
        )
        config = self._config(target_share=0.5)
        assert build_abstentions([positive], [ctx], config) == []


# --------------------------------------------------------------------------
# Orchestration: build_abstentions
# --------------------------------------------------------------------------


def make_rich_fixture(n_products: int = 12):
    """`n_products` contexts spread across categories, one positive per context citing
    its first two docs -- enough scale to exercise quota splitting and shortfall logic.
    """
    categories = ["fit_sizing", "colour_appearance", "fabric_quality", "durability", "defects", "occasion", "value"]
    contexts = []
    positives = []
    for i in range(n_products):
        category = categories[i % len(categories)]
        group_id = f"ctx_{category}_{i:04d}"
        doc_ids = (f"{i}0", f"{i}1", f"{i}2")
        context = make_context(group_id=group_id, category=category, parent_asin=f"P{i}", doc_ids=doc_ids)
        contexts.append(context)
        positives.append(
            make_positive(
                example_id=f"{group_id}_00",
                category=category,
                context_group_id=group_id,
                source_doc_ids=doc_ids,
                parent_asin=f"P{i}",
                question=f"Distinctive product-{i} question about {category}?",
                context_text=context.context_text,
                cited_doc_ids=(doc_ids[0], doc_ids[1]),
            )
        )
    return positives, contexts


class TestBuildAbstentionsOrchestration:
    def test_every_example_satisfies_the_abstention_invariants(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.5)
        abstentions = build_abstentions(positives, contexts, config)
        assert len(abstentions) > 0
        for example in abstentions:
            assert example.is_abstention is True
            assert ABSTENTION_TEXT in example.answer
            assert example.cited_doc_ids == ()

    def test_per_method_mix_matches_configured_split(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.6, methods=("transplant", "categorical_absence", "evidence_stripped"))
        n_total_target = round(config.target_share * len(positives) / (1 - config.target_share))
        expected_per_method = {
            "transplant": n_total_target // 3 + 1,  # remainder goes to earlier methods
            "categorical_absence": n_total_target // 3,
            "evidence_stripped": n_total_target // 3,
        }
        abstentions = build_abstentions(positives, contexts, config)
        from collections import Counter

        counts = Counter(e.provenance["method"] for e in abstentions)
        for method, expected in expected_per_method.items():
            assert counts[method] <= expected  # pool may fall short, never exceed target

    def test_total_target_formula(self):
        positives, contexts = make_rich_fixture(n_products=20)
        config = make_config(target_share=0.25, methods=("transplant", "categorical_absence", "evidence_stripped"))
        expected_total = round(0.25 * len(positives) / 0.75)
        abstentions = build_abstentions(positives, contexts, config)
        # Given ample pool size (20 products across categories), the target should be
        # fully met.
        assert len(abstentions) == expected_total

    def test_empty_positives_returns_empty_list_without_raising(self):
        _, contexts = make_rich_fixture()
        config = make_config(target_share=0.5)
        assert build_abstentions([], contexts, config) == []

    def test_shortfall_does_not_raise_and_returns_fewer_than_target(self, caplog):
        # A single product can support at most 1 transplant target (no other product to
        # transplant into) -- requesting many more than that must not raise.
        ctx = make_context(group_id="ctx_fit_sizing_0000", category="fit_sizing", parent_asin="P1")
        positive = make_positive(context_group_id="ctx_fit_sizing_0000")
        config = make_config(target_share=0.99, methods=("transplant",))
        with caplog.at_level("WARNING"):
            abstentions = build_abstentions([positive], [ctx], config)
        assert len(abstentions) <= 1
        assert any("pool exhausted" in r.message for r in caplog.records)

    def test_example_ids_are_unique(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.7)
        abstentions = build_abstentions(positives, contexts, config)
        ids = [e.example_id for e in abstentions]
        assert len(ids) == len(set(ids))

    def test_no_abstention_context_contains_a_doc_its_source_positive_cited(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.7)
        abstentions = build_abstentions(positives, contexts, config)
        by_example_id = {p.example_id: p for p in positives}

        for example in abstentions:
            source_id = example.provenance.get("source_positive_example_id")
            if source_id is None:
                continue  # categorical_absence has no source positive
            source_positive = by_example_id[source_id]
            assert set(example.source_doc_ids).isdisjoint(source_positive.cited_doc_ids)

    def test_deterministic_under_the_same_seed(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.5, seed=7)
        first = build_abstentions(list(positives), list(contexts), config)
        second = build_abstentions(list(positives), list(contexts), config)
        assert [e.to_dict() for e in first] == [e.to_dict() for e in second]

    def test_grouped_by_method_not_interleaved(self):
        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.5, methods=("transplant", "categorical_absence", "evidence_stripped"))
        abstentions = build_abstentions(positives, contexts, config)
        methods_seen = [e.provenance["method"] for e in abstentions]
        # once we've left a method's block we should never see it again
        seen_and_left: set[str] = set()
        current = None
        for m in methods_seen:
            if m != current:
                assert m not in seen_and_left, f"method {m!r} reappeared out of order"
                if current is not None:
                    seen_and_left.add(current)
                current = m


# --------------------------------------------------------------------------
# JSONL round trip (via cragb.finetune.schema, exercised end-to-end here)
# --------------------------------------------------------------------------


class TestWriteAbstentionsJsonl:
    def test_written_file_round_trips_through_schema_loader(self, tmp_path):
        from cragb.finetune.schema import load_training_examples_jsonl, write_training_examples_jsonl

        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.5)
        abstentions = build_abstentions(positives, contexts, config)

        out_path = write_training_examples_jsonl(abstentions, tmp_path / "abstentions_v1.jsonl")
        reloaded = load_training_examples_jsonl(out_path)
        assert reloaded == abstentions

    def test_overwrite_mode_does_not_duplicate_on_rerun(self, tmp_path):
        from cragb.finetune.schema import write_training_examples_jsonl

        positives, contexts = make_rich_fixture()
        config = make_config(target_share=0.5)
        abstentions = build_abstentions(positives, contexts, config)

        path = tmp_path / "abstentions_v1.jsonl"
        write_training_examples_jsonl(abstentions, path)
        write_training_examples_jsonl(abstentions, path)  # re-run "from scratch"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(abstentions)
