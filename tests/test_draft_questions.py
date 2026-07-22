"""Unit tests for cragb.generate.draft_questions.

Every function under test takes an injected `chat_fn` — a plain Python
callable standing in for `GroqClient.complete` — so prompt rendering,
response parsing, dedup, and per-category orchestration are all tested
with no network access and no API key, per M2.md T2.2's testability
requirement. Only `main()` (untested here) constructs a real client.
"""

from __future__ import annotations

import json
from string import Template

import pandas as pd
import pytest

from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec
from cragb.generate.draft_questions import (
    DraftQuestion,
    dedup_questions,
    draft_questions_for_taxonomy,
    generate_candidates_for_category,
    parse_llm_questions,
    render_prompt,
    sample_context_reviews,
)

FIT_CATEGORY = TaxonomyCategory(
    name="fit_sizing", description="Fit/sizing questions.", target_count=4, negative_count=1
)
COLOUR_CATEGORY = TaxonomyCategory(
    name="colour_appearance", description="Colour questions.", target_count=2, negative_count=1
)

SIMPLE_TEMPLATE = Template(
    "Category: $category_name\n$category_description\nReviews:\n$example_reviews\nN: $n_questions"
)


def make_spec(categories) -> TaxonomySpec:
    return TaxonomySpec(
        categories=tuple(categories),
        expected_total=sum(c.target_count for c in categories),
        total_tolerance=5,
        min_coverage_pct=0.01,
        min_coverage_pct_hard_floor=0.0,
    )


class TestRenderPrompt:
    def test_fills_all_placeholders(self):
        prompt = render_prompt(SIMPLE_TEMPLATE, FIT_CATEGORY, ["review one", "review two"], n_questions=5)
        assert "fit_sizing" in prompt
        assert "Fit/sizing questions." in prompt
        assert "review one" in prompt
        assert "N: 5" in prompt

    def test_no_reviews_uses_placeholder_text(self):
        prompt = render_prompt(SIMPLE_TEMPLATE, FIT_CATEGORY, [], n_questions=3)
        assert "no example reviews available" in prompt


class TestParseLlmQuestions:
    def test_parses_clean_json_array(self):
        raw = json.dumps(
            [
                {"question": "Does this run small?", "rationale": "sizing"},
                {"question": "Is it true to size?", "rationale": "sizing"},
            ]
        )
        questions = parse_llm_questions(raw, category="fit_sizing")
        assert len(questions) == 2
        assert questions[0].question == "Does this run small?"
        assert questions[0].category == "fit_sizing"
        assert questions[0].id == "fit_sizing_000"

    def test_strips_markdown_code_fence(self):
        raw = "```json\n" + json.dumps([{"question": "q1", "rationale": "r1"}]) + "\n```"
        questions = parse_llm_questions(raw, category="fit_sizing")
        assert len(questions) == 1
        assert questions[0].question == "q1"

    def test_extracts_json_array_even_with_surrounding_prose(self):
        raw = "Here are the questions:\n" + json.dumps([{"question": "q1", "rationale": "r1"}]) + "\nHope this helps!"
        questions = parse_llm_questions(raw, category="fit_sizing")
        assert len(questions) == 1

    def test_malformed_json_raises_with_raw_text_attached(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_llm_questions("not json at all", category="fit_sizing")

    def test_non_list_json_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON array"):
            parse_llm_questions(json.dumps({"question": "q1"}), category="fit_sizing")

    def test_item_missing_question_key_raises(self):
        with pytest.raises(ValueError, match="Malformed item"):
            parse_llm_questions(json.dumps([{"rationale": "no question here"}]), category="fit_sizing")


class TestDedupQuestions:
    def test_exact_duplicate_dropped(self):
        questions = [
            DraftQuestion("a", "fit_sizing", "Does this run small?", "r"),
            DraftQuestion("b", "fit_sizing", "does this run small?", "r"),
        ]
        kept, n_dropped = dedup_questions(questions, threshold=0.9)
        assert len(kept) == 1
        assert n_dropped == 1

    def test_near_duplicate_dropped(self):
        questions = [
            DraftQuestion("a", "fit_sizing", "Does this shirt run small on most buyers?", "r"),
            DraftQuestion("b", "fit_sizing", "Does this shirt run small for most buyers?", "r"),
        ]
        kept, n_dropped = dedup_questions(questions, threshold=0.9)
        assert len(kept) == 1
        assert n_dropped == 1

    def test_distinct_questions_both_kept(self):
        questions = [
            DraftQuestion("a", "fit_sizing", "Does this run small?", "r"),
            DraftQuestion("b", "fit_sizing", "Is the fabric itchy?", "r"),
        ]
        kept, n_dropped = dedup_questions(questions, threshold=0.9)
        assert len(kept) == 2
        assert n_dropped == 0

    def test_kept_order_preserved(self):
        questions = [
            DraftQuestion("a", "fit_sizing", "Second-distinct question here?", "r"),
            DraftQuestion("b", "fit_sizing", "First one gets dropped as dup?", "r"),
            DraftQuestion("c", "fit_sizing", "First one gets dropped as dup?", "r"),
        ]
        kept, n_dropped = dedup_questions(questions, threshold=0.9)
        assert [q.id for q in kept] == ["a", "b"]
        assert n_dropped == 1


class TestSampleContextReviews:
    def test_samples_only_matching_reviews(self):
        df = pd.DataFrame(
            {
                "text": [
                    "This runs small, I had to size up.",
                    "Totally unrelated review about nothing relevant.",
                    "True to size and comfortable fit.",
                ]
            }
        )
        reviews = sample_context_reviews(df, FIT_CATEGORY, n=5, seed=42)
        assert len(reviews) == 2
        assert all("unrelated" not in r for r in reviews)

    def test_truncates_to_max_chars(self):
        df = pd.DataFrame({"text": ["small " * 200]})
        reviews = sample_context_reviews(df, FIT_CATEGORY, n=1, seed=42, max_chars=20)
        assert len(reviews[0]) <= 20

    def test_raises_when_no_reviews_match(self):
        df = pd.DataFrame({"text": ["nothing relevant at all here"]})
        with pytest.raises(ValueError, match="No corpus reviews match"):
            sample_context_reviews(df, FIT_CATEGORY, n=5, seed=42)

    def test_reproducible_under_fixed_seed(self):
        df = pd.DataFrame(
            {"text": [f"true to size review number {i}" for i in range(20)]}
        )
        first = sample_context_reviews(df, FIT_CATEGORY, n=5, seed=7)
        second = sample_context_reviews(df, FIT_CATEGORY, n=5, seed=7)
        assert first == second


class TestGenerateCandidatesForCategory:
    def test_calls_chat_fn_with_rendered_prompt_and_parses_result(self):
        captured = {}

        def fake_chat_fn(messages):
            captured["messages"] = messages
            return json.dumps([{"question": "Does this run small?", "rationale": "sizing"}])

        result = generate_candidates_for_category(
            FIT_CATEGORY, ["a review"], SIMPLE_TEMPLATE, fake_chat_fn, n_questions=1
        )
        assert len(result) == 1
        assert result[0].question == "Does this run small?"
        assert "fit_sizing" in captured["messages"][0]["content"]


class TestDraftQuestionsForTaxonomy:
    def test_end_to_end_across_categories_with_fake_chat_fn(self):
        spec = make_spec([FIT_CATEGORY, COLOUR_CATEGORY])

        df = pd.DataFrame(
            {
                "text": [
                    "This runs small, I had to size up.",
                    "True to size and comfortable.",
                    "The colour faded after one wash, not as pictured.",
                    "Great colour, matches the photo exactly.",
                ]
            }
        )

        def fake_chat_fn(messages):
            prompt = messages[0]["content"]
            category_name = "fit_sizing" if "fit_sizing" in prompt else "colour_appearance"
            return json.dumps(
                [
                    {"question": f"Does this {category_name} run small on most buyers?", "rationale": "r"},
                    {"question": f"Is the {category_name} inconsistent across orders?", "rationale": "r"},
                ]
            )

        questions, counts = draft_questions_for_taxonomy(
            corpus=df,
            spec=spec,
            template=SIMPLE_TEMPLATE,
            chat_fn=fake_chat_fn,
            overgenerate_factor=1.0,
            reviews_per_category=2,
            max_review_chars=200,
            dedup_threshold=0.9,
            seed=42,
        )

        assert counts["fit_sizing"] == 2
        assert counts["colour_appearance"] == 2
        assert {q.category for q in questions} == {"fit_sizing", "colour_appearance"}

    def test_dedup_applied_per_category(self):
        spec = make_spec([FIT_CATEGORY])
        df = pd.DataFrame({"text": ["This runs small.", "True to size."]})

        def fake_chat_fn(_messages):
            return json.dumps(
                [
                    {"question": "Does this run small on most buyers?", "rationale": "r"},
                    {"question": "Does this run small for most buyers?", "rationale": "r"},
                ]
            )

        questions, counts = draft_questions_for_taxonomy(
            corpus=df,
            spec=spec,
            template=SIMPLE_TEMPLATE,
            chat_fn=fake_chat_fn,
            overgenerate_factor=1.0,
            reviews_per_category=2,
            max_review_chars=200,
            dedup_threshold=0.9,
            seed=42,
        )

        assert counts["fit_sizing"] == 1
        assert len(questions) == 1
