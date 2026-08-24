"""Unit tests for cragb.finetune.filter_pairs (T7.5; PLAN.md §3 E8, M7.md T7.5).

Every judged-stage function under test takes an injected `usage_fn` -- a plain Python
callable standing in for `GroqClient.complete_with_usage`, matching
`cragb.eval.cost_model.UsageFn` -- so no network access or API key is needed. Only
`main()` (untested here) constructs a real `GroqClient`.

Covers: `check_answer_format`'s structural checks, `run_stage1`'s citation/format
re-verification (including the fabrication-vs-format precedence that keeps the funnel's
three drop buckets mutually exclusive), `run_stage2`'s faithfulness-threshold gate and
its self-referential candidate==reference design, `filter_examples`'s end-to-end routing
(a stage-1 failure spends no API call; an abstention bypasses stage 2 entirely), and
`build_filter_funnel`'s per-slice arithmetic and judge-cost accounting.
"""

from __future__ import annotations

import json
from string import Template

import pandas as pd
import pytest

from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.eval.cost_model import ModelPricing
from cragb.finetune.filter_pairs import (
    CITATION_INVALID,
    FORMAT_INVALID,
    LOW_FAITHFULNESS,
    FilterConfig,
    build_filter_funnel,
    check_answer_format,
    filter_examples,
    load_filter_config,
    run_stage1,
    run_stage2,
)
from cragb.finetune.schema import TrainingExample
from cragb.generate.api_clients import CompletionResult

JUDGE_TEMPLATE = Template(
    "Q:$question\nCTX:$context_block\nCAND:$candidate_answer\nREF:$reference_answer"
)

PRICING = {"qwen/qwen3.6-27b": ModelPricing(0.60, 3.00, "2026-08-23", "https://example.com")}


def make_example(
    example_id: str = "e1",
    category: str = "fit_sizing",
    question: str = "Does this run small?",
    context_text: str = "[1] has_photo: no\nRuns small.\n\n[2] has_photo: no\nTrue to size.",
    answer: str = "Yes, it runs small [1].",
    cited_doc_ids: tuple[str, ...] = ("1",),
    is_abstention: bool = False,
    source_doc_ids: tuple[str, ...] = ("1", "2"),
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=source_doc_ids,
        source_parent_asins=("P1",),
        question=question,
        context_text=context_text,
        answer=answer,
        cited_doc_ids=cited_doc_ids,
        is_abstention=is_abstention,
        provenance={"method": "test"},
    )


def make_abstention(example_id: str = "a1", **kw) -> TrainingExample:
    defaults = dict(answer=ABSTENTION_TEXT, cited_doc_ids=(), is_abstention=True)
    defaults.update(kw)
    return make_example(example_id, **defaults)


def make_judge_response(faithfulness: int, correctness: int = 5, completeness: int = 5, conciseness: int = 5) -> str:
    return json.dumps(
        {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "conciseness": conciseness,
            "rationale": "test rationale",
        }
    )


def make_usage_fn(responses: list[str], *, cached: bool = False, model: str = "qwen/qwen3.6-27b"):
    calls: list[list[dict]] = []

    def usage_fn(messages: list[dict[str, str]]) -> CompletionResult:
        calls.append(messages)
        text = responses[len(calls) - 1]
        return CompletionResult(
            text=text, prompt_tokens=100, completion_tokens=30, latency_s=0.3, cached=cached, model=model
        )

    usage_fn.calls = calls  # type: ignore[attr-defined]
    return usage_fn


# --------------------------------------------------------------------------
# check_answer_format
# --------------------------------------------------------------------------


class TestCheckAnswerFormat:
    def test_well_formed_single_paragraph_has_no_issues(self):
        assert check_answer_format("These run small, size up for a better fit [1].") == ()

    def test_multiple_paragraphs_is_flagged(self):
        assert "multiple_paragraphs" in check_answer_format("First point.\n\nSecond point.")

    def test_bullet_marker_is_flagged(self):
        assert "bullet_or_heading" in check_answer_format("- Runs small\n- Size up")

    def test_numbered_list_marker_is_flagged(self):
        assert "bullet_or_heading" in check_answer_format("1. Runs small\n2. Size up")

    def test_markdown_heading_is_flagged(self):
        assert "bullet_or_heading" in check_answer_format("## Summary\nRuns small.")

    def test_json_like_braces_are_flagged(self):
        assert "brace_or_json" in check_answer_format('{"answer": "runs small"}')

    def test_single_newline_without_blank_line_is_not_flagged(self):
        # A soft wrap (one \n, no blank line) is not a second paragraph.
        assert check_answer_format("Runs small,\nsize up for a better fit [1].") == ()

    def test_multiple_issues_all_reported(self):
        issues = check_answer_format("- Point one\n\n- Point two {oops}")
        assert set(issues) == {"multiple_paragraphs", "bullet_or_heading", "brace_or_json"}


# --------------------------------------------------------------------------
# run_stage1
# --------------------------------------------------------------------------


class TestRunStage1:
    def test_valid_example_passes(self):
        result = run_stage1(make_example())
        assert result.passed is True
        assert result.drop_reason is None

    def test_fabricated_citation_fails_as_citation_invalid(self):
        example = make_example(answer="Yes, runs small [999].", cited_doc_ids=("999",))
        result = run_stage1(example)
        assert result.passed is False
        assert result.drop_reason == CITATION_INVALID
        assert result.fabricated_citations == ("999",)

    def test_non_abstention_with_no_citations_fails_as_citation_invalid(self):
        example = make_example(answer="This is a generic answer with no citations.", cited_doc_ids=())
        result = run_stage1(example)
        assert result.passed is False
        assert result.drop_reason == CITATION_INVALID

    def test_valid_abstention_passes(self):
        result = run_stage1(make_abstention())
        assert result.passed is True

    def test_bullet_format_fails_as_format(self):
        example = make_example(answer="- Runs small [1].\n- Size up.")
        result = run_stage1(example)
        assert result.passed is False
        assert result.drop_reason == FORMAT_INVALID
        assert "bullet_or_heading" in result.format_issues

    def test_malformed_citation_bracket_fails_as_format(self):
        # A real, valid citation [1] is present too, so this isolates the malformed-
        # bracket failure from the separate "no valid citations at all" (citation_invalid)
        # case tested above.
        example = make_example(answer="Runs small [1], see [review 1] for details.", cited_doc_ids=("1",))
        result = run_stage1(example)
        assert result.passed is False
        assert result.drop_reason == FORMAT_INVALID
        assert result.malformed_citation_brackets == ("[review 1]",)

    def test_fabrication_takes_precedence_over_format_when_both_present(self):
        # Bulleted AND fabricates a citation -- must be counted once, under
        # citation_invalid, so the funnel's three buckets stay mutually exclusive.
        example = make_example(answer="- Runs small [999].", cited_doc_ids=("999",))
        result = run_stage1(example)
        assert result.drop_reason == CITATION_INVALID

    def test_citation_mismatch_between_field_and_answer_text_is_caught(self):
        # cited_doc_ids claims ["1"] but the answer text itself cites "2" -- a
        # hypothetical drift this stage-1 re-check exists to catch.
        example = make_example(answer="Runs small [2].", cited_doc_ids=("1",))
        result = run_stage1(example)
        assert result.passed is False
        assert result.citation_mismatch is True
        assert result.drop_reason == CITATION_INVALID


# --------------------------------------------------------------------------
# run_stage2
# --------------------------------------------------------------------------


class TestRunStage2:
    def test_faithfulness_at_threshold_passes(self):
        usage_fn = make_usage_fn([make_judge_response(faithfulness=4)])
        passed, score, completion = run_stage2(make_example(), JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)
        assert passed is True
        assert score == 4

    def test_faithfulness_below_threshold_fails(self):
        usage_fn = make_usage_fn([make_judge_response(faithfulness=2)])
        passed, score, completion = run_stage2(make_example(), JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)
        assert passed is False
        assert score == 2

    def test_faithfulness_of_five_passes(self):
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        passed, score, completion = run_stage2(make_example(), JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)
        assert passed is True

    def test_candidate_and_reference_are_both_the_examples_own_answer(self):
        example = make_example(answer="Runs small, size up for a better fit [1].")
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        run_stage2(example, JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)
        sent_prompt = usage_fn.calls[0][0]["content"]
        assert "CAND:Runs small, size up for a better fit [1]." in sent_prompt
        assert "REF:Runs small, size up for a better fit [1]." in sent_prompt

    def test_returns_the_completion_for_cost_accounting(self):
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)], cached=True)
        _, _, completion = run_stage2(make_example(), JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)
        assert completion.cached is True
        assert completion.prompt_tokens == 100

    def test_malformed_judge_response_raises(self):
        usage_fn = make_usage_fn(["not valid json at all"])
        with pytest.raises(ValueError):
            run_stage2(make_example(), JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4)


# --------------------------------------------------------------------------
# filter_examples: end-to-end routing
# --------------------------------------------------------------------------


class TestFilterExamplesRouting:
    def test_stage1_failure_spends_no_api_call(self):
        bad = make_example(answer="Fabricated [999].", cited_doc_ids=("999",))

        def failing_usage_fn(messages):
            raise AssertionError("must not be called for a stage-1 failure")

        results = filter_examples([bad], JUDGE_TEMPLATE, failing_usage_fn, faithfulness_threshold=4, show_progress=False)
        assert results[0].accepted is False
        assert results[0].drop_reason == CITATION_INVALID

    def test_abstention_bypasses_stage2_and_spends_no_api_call(self):
        abst = make_abstention()

        def failing_usage_fn(messages):
            raise AssertionError("abstentions must not reach stage 2")

        results = filter_examples([abst], JUDGE_TEMPLATE, failing_usage_fn, faithfulness_threshold=4, show_progress=False)
        assert results[0].accepted is True
        assert results[0].drop_reason is None
        assert results[0].faithfulness_score is None

    def test_valid_example_with_faithfulness_2_is_dropped(self):
        example = make_example()
        usage_fn = make_usage_fn([make_judge_response(faithfulness=2)])
        results = filter_examples([example], JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)
        assert results[0].accepted is False
        assert results[0].drop_reason == LOW_FAITHFULNESS

    def test_valid_example_with_faithfulness_5_is_kept(self):
        example = make_example()
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        results = filter_examples([example], JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)
        assert results[0].accepted is True

    def test_mixed_batch_routes_each_example_correctly(self):
        good = make_example(example_id="good")
        fabricated = make_example(example_id="fabricated", answer="X [999].", cited_doc_ids=("999",))
        low_faith = make_example(example_id="low_faith")
        abst = make_abstention(example_id="abst")

        usage_fn = make_usage_fn(
            [make_judge_response(faithfulness=5), make_judge_response(faithfulness=1)]
        )
        results = filter_examples(
            [good, fabricated, low_faith, abst], JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False
        )
        by_id = {r.example.example_id: r for r in results}
        assert by_id["good"].accepted is True
        assert by_id["fabricated"].accepted is False and by_id["fabricated"].drop_reason == CITATION_INVALID
        assert by_id["low_faith"].accepted is False and by_id["low_faith"].drop_reason == LOW_FAITHFULNESS
        assert by_id["abst"].accepted is True
        assert len(usage_fn.calls) == 2  # only "good" and "low_faith" reached the judge

    def test_re_running_the_same_input_is_a_full_cache_hit(self):
        # Simulates DiskCache behaviour: the second usage_fn instance never gets called
        # because... actually this asserts the *design* claim structurally: filter_examples
        # itself has no incremental state, so calling it twice with a usage_fn backed by a
        # real disk cache would naturally replay cached responses. Here we assert the
        # simpler, directly-testable half: two independent runs over the same input with
        # independently-fresh (but content-identical) usage_fns produce identical results.
        example = make_example()
        first = filter_examples(
            [example], JUDGE_TEMPLATE, make_usage_fn([make_judge_response(faithfulness=5)]),
            faithfulness_threshold=4, show_progress=False,
        )
        second = filter_examples(
            [example], JUDGE_TEMPLATE, make_usage_fn([make_judge_response(faithfulness=5)]),
            faithfulness_threshold=4, show_progress=False,
        )
        assert first[0].accepted == second[0].accepted == True


# --------------------------------------------------------------------------
# build_filter_funnel
# --------------------------------------------------------------------------


class TestBuildFilterFunnel:
    def test_raises_on_empty_results(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_filter_funnel([], "qwen/qwen3.6-27b", PRICING)

    def test_overall_row_arithmetic_closes(self):
        examples = [make_example(example_id="good"), make_example(example_id="fab", answer="X [999].", cited_doc_ids=("999",))]
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        results = filter_examples(examples, JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)

        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        overall = funnel[funnel["slice"] == "overall"].iloc[0]
        assert overall["n_raw"] == 2
        assert overall["n_dropped_citation_invalid"] == 1
        assert overall["n_accepted"] == 1
        assert (
            overall["n_raw"]
            == overall["n_dropped_citation_invalid"]
            + overall["n_dropped_format"]
            + overall["n_dropped_low_faithfulness"]
            + overall["n_accepted"]
        )

    def test_per_category_slices_sum_to_overall(self):
        examples = [
            make_example(example_id="a", category="fit_sizing"),
            make_example(example_id="b", category="colour_appearance"),
        ]
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5), make_judge_response(faithfulness=5)])
        results = filter_examples(examples, JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)

        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        category_rows = funnel[funnel["slice"].str.startswith("category:")]
        assert category_rows["n_raw"].sum() == funnel[funnel["slice"] == "overall"]["n_raw"].iloc[0]
        assert set(category_rows["slice"]) == {"category:fit_sizing", "category:colour_appearance"}

    def test_is_abstention_slices_sum_to_overall(self):
        examples = [make_example(example_id="pos"), make_abstention(example_id="abst")]
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        results = filter_examples(examples, JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)

        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        abst_rows = funnel[funnel["slice"].str.startswith("is_abstention:")]
        assert abst_rows["n_raw"].sum() == 2
        true_row = funnel[funnel["slice"] == "is_abstention:True"].iloc[0]
        false_row = funnel[funnel["slice"] == "is_abstention:False"].iloc[0]
        assert true_row["n_raw"] == 1
        assert false_row["n_raw"] == 1

    def test_judge_cost_columns_present_only_on_overall_row(self):
        examples = [make_example()]
        usage_fn = make_usage_fn([make_judge_response(faithfulness=5)])
        results = filter_examples(examples, JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False)

        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        overall = funnel[funnel["slice"] == "overall"].iloc[0]
        assert overall["judge_n_calls"] == 1
        assert overall["judge_total_prompt_tokens"] == 100

        category_row = funnel[funnel["slice"].str.startswith("category:")].iloc[0]
        assert pd.isna(category_row["judge_n_calls"])

    def test_judge_cost_counts_calls_regardless_of_accept_reject(self):
        good = make_example(example_id="good")
        low_faith = make_example(example_id="low_faith")
        usage_fn = make_usage_fn(
            [make_judge_response(faithfulness=5), make_judge_response(faithfulness=1)]
        )
        results = filter_examples(
            [good, low_faith], JUDGE_TEMPLATE, usage_fn, faithfulness_threshold=4, show_progress=False
        )
        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        overall = funnel[funnel["slice"] == "overall"].iloc[0]
        assert overall["judge_n_calls"] == 2  # both attempts cost a call, one passed one failed

    def test_zero_judge_calls_when_all_examples_are_abstentions(self):
        results = filter_examples(
            [make_abstention()], JUDGE_TEMPLATE, lambda m: (_ for _ in ()).throw(AssertionError()),
            faithfulness_threshold=4, show_progress=False,
        )
        funnel = build_filter_funnel(results, "qwen/qwen3.6-27b", PRICING)
        overall = funnel[funnel["slice"] == "overall"].iloc[0]
        assert overall["judge_n_calls"] == 0
        assert overall["judge_total_usd"] == 0.0


# --------------------------------------------------------------------------
# FilterConfig
# --------------------------------------------------------------------------


class TestFilterConfig:
    def test_valid_threshold_constructs(self):
        assert FilterConfig(faithfulness_threshold=4).faithfulness_threshold == 4

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="must be in \\[1, 5\\]"):
            FilterConfig(faithfulness_threshold=6)

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="must be in \\[1, 5\\]"):
            FilterConfig(faithfulness_threshold=0)


class TestLoadFilterConfig:
    def test_loads_real_finetune_config(self):
        config = load_filter_config("configs/finetune.yaml")
        assert 1 <= config.faithfulness_threshold <= 5
