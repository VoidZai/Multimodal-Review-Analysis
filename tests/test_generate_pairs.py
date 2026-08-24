"""Unit tests for cragb.finetune.generate_pairs (T7.3; PLAN.md §3 E8, M7.md T7.3).

Every function under test takes an injected `usage_fn` -- a plain Python callable
standing in for `GroqClient.complete_with_usage`, matching `cragb.eval.cost_model.UsageFn`
-- so prompt rendering, parsing, and batch orchestration are all tested with no network
access and no API key, per this project's established testability shape (`ChatFn`,
`UsageFn`). Only `main()` (untested here) constructs a real `GroqClient`.

Covers: `render_generation_prompt`'s category-consistency guard, `parse_raw_items`'s
defensive JSON-array extraction (well-formed / prose-wrapped / truncated / malformed-item
cases), `parse_generated_pairs`'s citation-fabrication and structural-invariant drops,
`generate_all`'s resumability (skip-already-done, interrupt-and-restart producing the same
files, `--limit`, 429/error handling), and `build_generation_cost_row`'s cumulative,
resume-safe cost accounting.
"""

from __future__ import annotations

import json
from string import Template

import pandas as pd
import pytest
import requests

from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec
from cragb.finetune.generate_pairs import (
    GenerationReport,
    append_progress_row,
    append_training_examples_jsonl,
    build_generation_cost_row,
    generate_all,
    load_done_context_group_ids,
    parse_generated_pairs,
    parse_raw_items,
    render_generation_prompt,
)
from cragb.finetune.sample_contexts import ContextGroup, load_contexts_jsonl
from cragb.finetune.schema import TrainingExample
from cragb.generate.api_clients import CompletionResult
from cragb.eval.cost_model import ModelPricing
from cragb.utils.io import resolve_path

FIT_CATEGORY = TaxonomyCategory(
    name="fit_sizing", description="Fit and sizing questions.", target_count=4, negative_count=1
)
COLOUR_CATEGORY = TaxonomyCategory(
    name="colour_appearance", description="Colour questions.", target_count=2, negative_count=1
)


def make_taxonomy(categories=(FIT_CATEGORY, COLOUR_CATEGORY)) -> TaxonomySpec:
    return TaxonomySpec(
        categories=tuple(categories),
        expected_total=sum(c.target_count for c in categories),
        total_tolerance=5,
        min_coverage_pct=0.01,
        min_coverage_pct_hard_floor=0.0,
    )


SIMPLE_TEMPLATE = Template(
    "Category: $category_name\n$category_description\n$context_block\nN: $n_questions"
)


def make_group(
    group_id: str = "ctx_fit_sizing_0000",
    category: str = "fit_sizing",
    parent_asin: str = "P1",
    doc_ids: tuple[str, ...] = ("1", "2", "3"),
    context_text: str = "[1] has_photo: no\nRuns small.\n\n[2] has_photo: no\nTrue to size.\n\n[3] has_photo: yes\nComfy fit.",
    photo_bearing: bool = True,
) -> ContextGroup:
    return ContextGroup(
        group_id=group_id,
        category=category,
        parent_asin=parent_asin,
        doc_ids=doc_ids,
        context_text=context_text,
        photo_bearing=photo_bearing,
    )


def make_response(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"question": q, "answer": a} for q, a in pairs])


def make_usage_fn(responses: list[str], *, cached: bool = False, model: str = "openai/gpt-oss-120b"):
    """A `UsageFn` stub returning `responses` in order, one per call; records call count."""
    calls: list[list[dict]] = []

    def usage_fn(messages: list[dict[str, str]]) -> CompletionResult:
        calls.append(messages)
        text = responses[len(calls) - 1]
        return CompletionResult(
            text=text, prompt_tokens=100, completion_tokens=50, latency_s=0.5, cached=cached, model=model
        )

    usage_fn.calls = calls  # type: ignore[attr-defined]
    return usage_fn


# --------------------------------------------------------------------------
# render_generation_prompt
# --------------------------------------------------------------------------


class TestRenderGenerationPrompt:
    def test_substitutes_category_and_context(self):
        group = make_group()
        prompt = render_generation_prompt(group, FIT_CATEGORY, 3, SIMPLE_TEMPLATE)
        assert "Category: fit_sizing" in prompt
        assert "Fit and sizing questions." in prompt
        assert group.context_text in prompt
        assert "N: 3" in prompt

    def test_mismatched_category_raises(self):
        group = make_group(category="fit_sizing")
        with pytest.raises(ValueError, match="does not match"):
            render_generation_prompt(group, COLOUR_CATEGORY, 3, SIMPLE_TEMPLATE)


# --------------------------------------------------------------------------
# parse_raw_items
# --------------------------------------------------------------------------


class TestParseRawItems:
    def test_well_formed_array_parses(self):
        raw = make_response(("Does this run small?", "Yes [1]."), ("Is it true to size?", "Yes [2]."))
        items = parse_raw_items(raw)
        assert len(items) == 2
        assert items[0] == {"question": "Does this run small?", "answer": "Yes [1]."}

    def test_prose_wrapped_array_still_parses(self):
        raw = "Sure, here are the questions:\n" + make_response(("Q1?", "A1 [1].")) + "\nHope that helps!"
        items = parse_raw_items(raw)
        assert len(items) == 1

    def test_markdown_code_fence_is_stripped(self):
        raw = "```json\n" + make_response(("Q1?", "A1 [1].")) + "\n```"
        items = parse_raw_items(raw)
        assert len(items) == 1

    def test_truncated_array_raises(self):
        # A cut-off response with no closing bracket at all -- the §14.6 lesson: this
        # must raise, not silently yield a partial/empty list.
        raw = '[{"question": "Does this run small?", "answer": "Yes it runs sm'
        with pytest.raises(ValueError, match="Could not parse"):
            parse_raw_items(raw)

    def test_truncated_between_objects_raises(self):
        raw = '[{"question": "Q1?", "answer": "A1 [1]."}, {"question": "Q2?"'
        with pytest.raises(ValueError, match="Could not parse"):
            parse_raw_items(raw)

    def test_empty_array_is_a_valid_empty_result_not_an_error(self):
        assert parse_raw_items("[]") == []

    def test_non_array_json_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON array"):
            parse_raw_items('{"question": "Q1?", "answer": "A1."}')

    def test_item_missing_answer_key_raises(self):
        with pytest.raises(ValueError, match="Malformed item"):
            parse_raw_items('[{"question": "Q1?"}]')

    def test_item_missing_question_key_raises(self):
        with pytest.raises(ValueError, match="Malformed item"):
            parse_raw_items('[{"answer": "A1."}]')

    def test_item_that_is_not_an_object_raises(self):
        with pytest.raises(ValueError, match="Malformed item"):
            parse_raw_items('["just a string"]')

    def test_empty_question_string_raises(self):
        with pytest.raises(ValueError, match="empty question or answer"):
            parse_raw_items('[{"question": "  ", "answer": "A1."}]')

    def test_garbage_text_with_no_brackets_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_raw_items("I cannot help with that request.")


# --------------------------------------------------------------------------
# parse_generated_pairs
# --------------------------------------------------------------------------


class TestParseGeneratedPairs:
    def test_well_formed_response_yields_n_examples(self):
        group = make_group()
        raw = make_response(
            ("Does this run small?", "Yes, runs small [1]."),
            ("Is it true to size?", "True to size [2]."),
        )
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert len(examples) == 2
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_example_fields_are_populated_from_context_group(self):
        group = make_group(doc_ids=("1", "2", "3"), parent_asin="P1", category="fit_sizing")
        raw = make_response(("Does this run small?", "Yes [1]."))
        [example] = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert example.category == "fit_sizing"
        assert example.source_doc_ids == ("1", "2", "3")
        assert example.source_parent_asins == ("P1",)
        assert example.context_text == group.context_text
        assert example.cited_doc_ids == ("1",)
        assert example.is_abstention is False

    def test_example_id_encodes_group_id_and_item_index(self):
        group = make_group(group_id="ctx_fit_sizing_0007")
        raw = make_response(("Q1?", "A1 [1]."), ("Q2?", "A2 [2]."))
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert examples[0].example_id == "ctx_fit_sizing_0007_00"
        assert examples[1].example_id == "ctx_fit_sizing_0007_01"

    def test_provenance_is_merged_with_context_group_id_and_item_index(self):
        group = make_group()
        raw = make_response(("Q1?", "A1 [1]."))
        [example] = parse_generated_pairs(
            raw, group, provenance={"method": "teacher_generation", "teacher_model": "x"}
        )
        assert example.provenance["method"] == "teacher_generation"
        assert example.provenance["teacher_model"] == "x"
        assert example.provenance["context_group_id"] == group.group_id
        assert example.provenance["raw_item_index"] == 0

    def test_fabricated_citation_is_dropped_not_raised(self):
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(
            ("Does this run small?", "Yes, runs small [999]."),  # 999 not in context
            ("Is it true to size?", "True to size [2]."),  # valid
        )
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert len(examples) == 1
        assert examples[0].question == "Is it true to size?"

    def test_partially_fabricated_citation_set_is_dropped(self):
        # [2] is real, [999] is not -- the whole item is still fabricated evidence.
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(("Q1?", "Mixed evidence [2][999]."))
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert examples == []

    def test_accidental_abstention_text_answer_is_dropped_not_raised(self):
        from cragb.bench.reference_answers import ABSTENTION_TEXT

        group = make_group()
        raw = make_response(("Q1?", ABSTENTION_TEXT), ("Q2?", "Real answer [1]."))
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert len(examples) == 1
        assert examples[0].question == "Q2?"

    def test_all_items_dropped_yields_empty_list_not_an_error(self):
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(("Q1?", "Fabricated [999]."))
        assert parse_generated_pairs(raw, group, provenance={"method": "test"}) == []

    def test_malformed_response_still_raises_through(self):
        group = make_group()
        with pytest.raises(ValueError, match="Could not parse"):
            parse_generated_pairs('[{"question": "Q1?"', group, provenance={"method": "test"})

    def test_photo_citation_marker_does_not_count_as_a_doc_citation(self):
        # "[photo of 3]" must not be checked against doc_ids as if it were a bare
        # citation -- extract_citations' \w+ pattern can't match the spaces in it.
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(("Q1?", "Comfortable fit, see photo [3][photo of 3]."))
        examples = parse_generated_pairs(raw, group, provenance={"method": "test"})
        assert len(examples) == 1
        assert examples[0].cited_doc_ids == ("3",)


# --------------------------------------------------------------------------
# generate_all
# --------------------------------------------------------------------------


class TestGenerateAllBasics:
    def test_generates_and_writes_examples(self, tmp_path):
        group = make_group()
        raw = make_response(("Q1?", "A1 [1]."), ("Q2?", "A2 [2]."))
        usage_fn = make_usage_fn([raw])
        raw_path = tmp_path / "raw.jsonl"
        progress_path = tmp_path / "progress.jsonl"

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=2,
            dedup_threshold=0.9,
            raw_pairs_path=raw_path,
            progress_path=progress_path,
            show_progress=False,
        )

        assert report.n_examples_accepted == 2
        assert report.n_contexts_processed_this_run == 1
        assert report.n_contexts_done == 1
        assert report.n_new_calls == 1
        assert len(usage_fn.calls) == 1
        assert len(raw_path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_cache_hit_is_counted_separately_from_new_calls(self, tmp_path):
        group = make_group()
        raw = make_response(("Q1?", "A1 [1]."))
        usage_fn = make_usage_fn([raw], cached=True)

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=1,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_cache_hits == 1
        assert report.n_new_calls == 0

    def test_invalid_citation_items_are_reflected_in_dropped_invalid_count(self, tmp_path):
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(("Q1?", "Fabricated [999]."), ("Q2?", "Real [1]."))
        usage_fn = make_usage_fn([raw])

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=2,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_examples_generated_raw == 2
        assert report.n_examples_dropped_invalid == 1
        assert report.n_examples_accepted == 1

    def test_near_duplicate_questions_within_a_context_are_dropped(self, tmp_path):
        group = make_group()
        raw = make_response(
            ("Does this run small?", "Yes, runs small [1]."),
            ("Does this runs small?", "Yes, it runs small [1]."),  # near-duplicate of above (ratio ~0.98)
        )
        usage_fn = make_usage_fn([raw])

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=2,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_examples_dropped_near_duplicate == 1
        assert report.n_examples_accepted == 1

    def test_parse_failure_is_not_fatal_to_the_whole_run(self, tmp_path):
        good_group = make_group(group_id="ctx_fit_sizing_0000")
        bad_group = make_group(group_id="ctx_fit_sizing_0001")
        good_raw = make_response(("Q1?", "A1 [1]."))
        bad_raw = "not json at all, no brackets here"
        usage_fn = make_usage_fn([bad_raw, good_raw])

        report = generate_all(
            [bad_group, good_group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=1,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_errors == 1
        assert report.n_examples_accepted == 1
        # The failed context must NOT be marked done -- it should retry on resume.
        done = load_done_context_group_ids(tmp_path / "progress.jsonl")
        assert done == {"ctx_fit_sizing_0000"}

    def test_429_after_retries_exhausted_is_not_fatal_and_not_marked_done(self, tmp_path):
        group = make_group()

        response = requests.Response()
        response.status_code = 429

        def usage_fn(messages):
            raise requests.HTTPError(response=response)

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=1,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_rate_limited == 1
        assert report.n_contexts_processed_this_run == 0
        assert load_done_context_group_ids(tmp_path / "progress.jsonl") == set()

    def test_non_429_http_error_is_counted_as_a_generic_error(self, tmp_path):
        group = make_group()
        response = requests.Response()
        response.status_code = 503

        def usage_fn(messages):
            raise requests.HTTPError(response=response)

        report = generate_all(
            [group],
            taxonomy=make_taxonomy(),
            template=SIMPLE_TEMPLATE,
            usage_fn=usage_fn,
            questions_per_context=1,
            dedup_threshold=0.9,
            raw_pairs_path=tmp_path / "raw.jsonl",
            progress_path=tmp_path / "progress.jsonl",
            show_progress=False,
        )
        assert report.n_errors == 1
        assert report.n_rate_limited == 0


class TestGenerateAllResumability:
    def test_already_done_context_is_skipped_without_a_call(self, tmp_path):
        group = make_group()
        raw_path = tmp_path / "raw.jsonl"
        progress_path = tmp_path / "progress.jsonl"

        first_usage_fn = make_usage_fn([make_response(("Q1?", "A1 [1]."))])
        generate_all(
            [group], taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE, usage_fn=first_usage_fn,
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, show_progress=False,
        )

        def failing_usage_fn(messages):
            raise AssertionError("must not be called for an already-done context")

        report = generate_all(
            [group], taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE, usage_fn=failing_usage_fn,
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, show_progress=False,
        )
        assert report.n_contexts_skipped_already_done == 1
        assert report.n_contexts_processed_this_run == 0

    def test_interrupt_and_restart_produces_the_same_file_as_uninterrupted(self, tmp_path):
        # Each group needs genuinely distinct context_text so its rendered prompt is
        # distinguishable from the others -- a real disk cache keys on the request
        # payload (which embeds the prompt), so this is what actually exercises
        # per-group cache-key correctness rather than relying on call order.
        groups = [
            make_group(
                group_id=f"ctx_fit_sizing_{i:04d}",
                doc_ids=(str(i),),
                context_text=f"[{i}] has_photo: no\nReview text for group {i}.",
            )
            for i in range(4)
        ]
        # A content-addressed lookup, built once up front -- exactly how a real
        # DiskCache-backed client behaves: the same prompt always maps to the same
        # response, independent of which usage_fn *instance* (i.e. which "process") asks
        # for it, and independent of call order.
        responses_by_prompt = {
            render_generation_prompt(g, FIT_CATEGORY, 1, SIMPLE_TEMPLATE): make_response(
                (f"Q{i}?", f"A{i} [{i}].")
            )
            for i, g in enumerate(groups)
        }

        def make_lookup_usage_fn():
            def usage_fn(messages):
                return CompletionResult(
                    text=responses_by_prompt[messages[0]["content"]],
                    prompt_tokens=100,
                    completion_tokens=50,
                    latency_s=0.5,
                    cached=False,
                    model="openai/gpt-oss-120b",
                )

            return usage_fn

        # Uninterrupted run.
        raw_a = tmp_path / "a_raw.jsonl"
        progress_a = tmp_path / "a_progress.jsonl"
        generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_lookup_usage_fn(),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_a, progress_path=progress_a, show_progress=False,
        )

        # "Interrupted" run: first 2 contexts, then a fresh usage_fn (simulating a new
        # process) for the remaining 2.
        raw_b = tmp_path / "b_raw.jsonl"
        progress_b = tmp_path / "b_progress.jsonl"
        generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_lookup_usage_fn(),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_b, progress_path=progress_b, limit=2, show_progress=False,
        )
        generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_lookup_usage_fn(),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_b, progress_path=progress_b, show_progress=False,
        )

        def normalize(path):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["provenance"].pop("generated_at", None)  # timestamps legitimately differ
            return rows

        assert normalize(raw_a) == normalize(raw_b)

    def test_limit_caps_new_contexts_but_not_skipped_ones(self, tmp_path):
        groups = [make_group(group_id=f"ctx_fit_sizing_{i:04d}") for i in range(3)]
        responses = [make_response((f"Q{i}?", f"A{i} [1].")) for i in range(3)]
        raw_path = tmp_path / "raw.jsonl"
        progress_path = tmp_path / "progress.jsonl"

        generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_usage_fn(list(responses)),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, limit=1, show_progress=False,
        )
        report = generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_usage_fn(list(responses)),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, limit=1, show_progress=False,
        )
        assert report.n_contexts_skipped_already_done == 1
        assert report.n_contexts_processed_this_run == 1
        assert report.n_contexts_done == 2

    def test_zero_yield_context_is_still_marked_done(self, tmp_path):
        # Every item gets dropped for a fabricated citation -- the context must still be
        # marked done, or a resume would re-spend a call on it forever.
        group = make_group(doc_ids=("1", "2", "3"))
        raw = make_response(("Q1?", "Fabricated [999]."))
        raw_path = tmp_path / "raw.jsonl"
        progress_path = tmp_path / "progress.jsonl"

        report = generate_all(
            [group], taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_usage_fn([raw]),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, show_progress=False,
        )
        assert report.n_examples_accepted == 0
        assert load_done_context_group_ids(progress_path) == {group.group_id}
        # raw.jsonl was never created (nothing to write) -- confirm the progress log
        # alone, not the raw-pairs file, is what's authoritative for "done".
        assert not raw_path.exists() or raw_path.read_text(encoding="utf-8").strip() == ""


# --------------------------------------------------------------------------
# append_progress_row / append_training_examples_jsonl / load_done_context_group_ids
# --------------------------------------------------------------------------


class TestProgressLogHelpers:
    def test_load_done_on_missing_file_returns_empty_set(self, tmp_path):
        assert load_done_context_group_ids(tmp_path / "does_not_exist.jsonl") == set()

    def test_append_progress_row_is_append_only(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        append_progress_row(
            path, "g1", n_accepted=2, cached=False,
            prompt_tokens=100, completion_tokens=50, latency_s=0.5,
        )
        append_progress_row(
            path, "g2", n_accepted=0, cached=True,
            prompt_tokens=None, completion_tokens=None, latency_s=None,
        )
        assert load_done_context_group_ids(path) == {"g1", "g2"}

    def test_append_training_examples_jsonl_appends_across_calls(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        group = make_group()
        [ex1] = parse_generated_pairs(make_response(("Q1?", "A1 [1].")), group, provenance={"method": "t"})
        [ex2] = parse_generated_pairs(make_response(("Q2?", "A2 [2].")), group, provenance={"method": "t"})
        append_training_examples_jsonl([ex1], path)
        append_training_examples_jsonl([ex2], path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2


# --------------------------------------------------------------------------
# build_generation_cost_row
# --------------------------------------------------------------------------


PRICING = {"openai/gpt-oss-120b": ModelPricing(0.15, 0.60, "2026-08-23", "https://example.com")}


class TestBuildGenerationCostRow:
    def test_raises_on_empty_progress_log(self, tmp_path):
        with pytest.raises(ValueError, match="No rows"):
            build_generation_cost_row(tmp_path / "progress.jsonl", "openai/gpt-oss-120b", PRICING)

    def test_sums_tokens_and_calls_across_rows(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        append_progress_row(path, "g1", n_accepted=2, cached=False, prompt_tokens=100, completion_tokens=50, latency_s=1.0)
        append_progress_row(path, "g2", n_accepted=1, cached=False, prompt_tokens=200, completion_tokens=80, latency_s=1.5)

        row = build_generation_cost_row(path, "openai/gpt-oss-120b", PRICING).iloc[0]
        assert row["n_calls"] == 2
        assert row["total_prompt_tokens"] == 300
        assert row["total_completion_tokens"] == 130
        assert row["mean_prompt_tokens"] == 150.0

    def test_cached_calls_excluded_from_wall_clock_but_included_in_tokens(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        append_progress_row(path, "g1", n_accepted=1, cached=False, prompt_tokens=100, completion_tokens=50, latency_s=2.0)
        append_progress_row(path, "g2", n_accepted=1, cached=True, prompt_tokens=100, completion_tokens=50, latency_s=None)

        row = build_generation_cost_row(path, "openai/gpt-oss-120b", PRICING).iloc[0]
        assert row["total_wall_clock_s"] == 2.0  # only the non-cached row's latency
        assert row["total_prompt_tokens"] == 200  # both rows' tokens still counted
        assert row["n_cache_hits"] == 1

    def test_missing_usage_counted_and_treated_as_zero_tokens(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        append_progress_row(path, "g1", n_accepted=1, cached=False, prompt_tokens=None, completion_tokens=None, latency_s=1.0)
        append_progress_row(path, "g2", n_accepted=1, cached=False, prompt_tokens=100, completion_tokens=50, latency_s=1.0)

        row = build_generation_cost_row(path, "openai/gpt-oss-120b", PRICING).iloc[0]
        assert row["n_calls_missing_usage"] == 1
        assert row["total_prompt_tokens"] == 100

    def test_total_usd_matches_hand_computed_cost(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        append_progress_row(path, "g1", n_accepted=1, cached=False, prompt_tokens=1_000_000, completion_tokens=1_000_000, latency_s=1.0)

        row = build_generation_cost_row(path, "openai/gpt-oss-120b", PRICING).iloc[0]
        assert row["total_usd"] == pytest.approx(0.15 + 0.60)

    def test_reflects_cumulative_progress_across_two_generate_all_sessions(self, tmp_path):
        # This is the actual resumability contract build_generation_cost_row exists for:
        # its total must equal the sum of tokens spent across BOTH sessions, not just the
        # most recent one.
        groups = [make_group(group_id=f"ctx_fit_sizing_{i:04d}") for i in range(2)]
        raw_path = tmp_path / "raw.jsonl"
        progress_path = tmp_path / "progress.jsonl"

        generate_all(
            [groups[0]], taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_usage_fn([make_response(("Q0?", "A0 [1]."))]),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, show_progress=False,
        )
        generate_all(
            groups, taxonomy=make_taxonomy(), template=SIMPLE_TEMPLATE,
            usage_fn=make_usage_fn([make_response(("Q1?", "A1 [1]."))]),
            questions_per_context=1, dedup_threshold=0.9,
            raw_pairs_path=raw_path, progress_path=progress_path, show_progress=False,
        )

        row = build_generation_cost_row(progress_path, "openai/gpt-oss-120b", PRICING).iloc[0]
        assert row["n_calls"] == 2


# --------------------------------------------------------------------------
# Real-artifact wiring
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not resolve_path("src/cragb/generate/prompts/finetune_gen_v1.md").is_file(),
    reason="finetune_gen_v1.md prompt not present locally",
)
class TestRealPromptTemplateWiring:
    def test_real_template_renders_and_restates_citation_rules(self):
        from cragb.generate.grounded_qa import load_prompt_template

        template = load_prompt_template("src/cragb/generate/prompts/finetune_gen_v1.md")
        group = make_group()
        prompt = render_generation_prompt(group, FIT_CATEGORY, 3, template)
        assert "Cite every claim" in prompt
        assert "Never invent a review id" in prompt
        assert group.context_text in prompt
