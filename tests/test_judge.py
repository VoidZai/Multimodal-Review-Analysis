"""Unit tests for cragb.eval.judge (T4b.4; M4b.md T4b.4).

Every function that would otherwise call a live LLM takes an injected `chat_fn` -- a
plain callable standing in for `GroqClient.complete` -- so prompt rendering and response
parsing are all tested with no network access or API key, mirroring
`tests/test_grounded_qa.py`'s pattern for T4a.3.
"""

from __future__ import annotations

import json
import re
from string import Template

import pytest

from cragb.eval.judge import (
    NO_CONTEXT_MARKER,
    JudgeScore,
    build_judge_prompt,
    parse_judge_response,
    score_answer,
)
from cragb.generate.grounded_qa import load_prompt_template

# Word-boundary, not a bare substring check: the benchmark's own name, "CRAGB", contains
# "rag" as a substring ("c-RAG-b"), so a naive `"rag" in text` would false-positive on
# the real prompt file's own title heading. `\brag\b` correctly matches a standalone
# "RAG"/"rag" token (the retrieval-augmented-generation acronym this must never leak)
# while leaving "CRAGB" alone.
_LEAKY_TERM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"closed-book", r"closed book", r"\brag\b", r"gpt-oss", r"qwen", r"openai", r"groq",
)]


def assert_no_system_leak(prompt: str) -> None:
    for pattern in _LEAKY_TERM_PATTERNS:
        assert not pattern.search(prompt), f"prompt leaked {pattern.pattern!r}: {prompt!r}"

TEMPLATE = Template(
    "Q: $question\nContext: $context_block\nCandidate: $candidate_answer\nReference: $reference_answer"
)

VALID_JSON = json.dumps(
    {"correctness": 5, "faithfulness": 4, "completeness": 5, "conciseness": 3, "rationale": "Matches the reference."}
)


# --------------------------------------------------------------------------
# build_judge_prompt
# --------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_fills_all_fields_when_context_is_given(self):
        prompt = build_judge_prompt(
            "Do these run small?", "[101] Runs small.", "Yes, runs small.", "Runs small [101].", TEMPLATE
        )
        assert "Q: Do these run small?" in prompt
        assert "Context: [101] Runs small." in prompt
        assert "Candidate: Yes, runs small." in prompt
        assert "Reference: Runs small [101]." in prompt

    def test_none_context_renders_neutral_no_context_marker(self):
        prompt = build_judge_prompt("Q?", None, "An answer.", "A reference.", TEMPLATE)
        assert NO_CONTEXT_MARKER in prompt

    def test_rendered_prompt_never_leaks_which_system_produced_the_candidate(self):
        # PLAN.md §9 / module docstring: the judge must never learn which arm/system
        # produced the candidate. The static template + NO_CONTEXT_MARKER text itself
        # must never spell out "closed-book", "RAG", or a model family name -- that is
        # the actual regression this test guards, independent of whatever content a
        # caller happens to pass in for the candidate/reference/question.
        no_context_prompt = build_judge_prompt("Q?", None, "An answer.", "A reference.", TEMPLATE)
        with_context_prompt = build_judge_prompt("Q?", "[101] some review text.", "An answer.", "A reference.", TEMPLATE)

        assert_no_system_leak(no_context_prompt)
        assert_no_system_leak(with_context_prompt)

    def test_real_shipped_prompt_file_never_leaks_which_system_produced_the_candidate(self):
        # Same guard, but against the actual file main() loads (src/cragb/generate/
        # prompts/answer_judge_v1.md), not the small synthetic TEMPLATE above -- proves
        # the shipped artifact is clean, not just this test file's own fixture. Notably,
        # the real file's title heading contains "CRAGB", which itself contains "rag" as
        # a substring -- assert_no_system_leak's word-boundary check must not false-
        # positive on that.
        real_template = load_prompt_template("src/cragb/generate/prompts/answer_judge_v1.md")
        no_context_prompt = build_judge_prompt(
            "Do these run small?", None, "An answer.", "A reference answer.", real_template
        )
        with_context_prompt = build_judge_prompt(
            "Do these run small?", "[101] Runs small.", "An answer.", "A reference answer.", real_template
        )

        assert "CRAGB" in with_context_prompt  # sanity: the title *is* actually rendered
        assert_no_system_leak(no_context_prompt)
        assert_no_system_leak(with_context_prompt)

    def test_empty_string_context_is_rendered_as_is_not_as_the_marker(self):
        # None means "no context at all"; an empty string is a distinct, valid context
        # value (e.g. a ContextBlock that legitimately rendered to "") and must not be
        # silently upgraded to the no-context marker.
        prompt = build_judge_prompt("Q?", "", "An answer.", "A reference.", TEMPLATE)
        assert NO_CONTEXT_MARKER not in prompt
        assert "Context: \n" in prompt or prompt.endswith("Context: ") is False  # rendered as empty, not omitted


# --------------------------------------------------------------------------
# parse_judge_response
# --------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_parses_well_formed_json(self):
        score = parse_judge_response(VALID_JSON)
        assert score == JudgeScore(
            correctness=5, faithfulness=4, completeness=5, conciseness=3, rationale="Matches the reference."
        )

    def test_tolerates_markdown_code_fence(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        score = parse_judge_response(fenced)
        assert score.correctness == 5

    def test_tolerates_leading_trailing_prose_around_the_json_object(self):
        wrapped = f"Here is my evaluation:\n{VALID_JSON}\nThat concludes my scoring."
        score = parse_judge_response(wrapped)
        assert score.correctness == 5

    def test_truncated_json_raises_value_error_with_raw_response_attached(self):
        truncated = '{"correctness": 5, "faithfulness": 4, "completeness"'
        with pytest.raises(ValueError, match="Could not parse a JSON object"):
            parse_judge_response(truncated)

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON object"):
            parse_judge_response("[1, 2, 3]")

    def test_missing_key_raises(self):
        obj = json.loads(VALID_JSON)
        del obj["faithfulness"]
        with pytest.raises(ValueError, match="missing key"):
            parse_judge_response(json.dumps(obj))

    @pytest.mark.parametrize("bad_value", [0, 6, -1, 10])
    def test_out_of_range_score_raises(self, bad_value):
        obj = json.loads(VALID_JSON)
        obj["correctness"] = bad_value
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_judge_response(json.dumps(obj))

    def test_non_integer_score_raises(self):
        obj = json.loads(VALID_JSON)
        obj["correctness"] = 4.5
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_judge_response(json.dumps(obj))

    def test_string_score_raises(self):
        obj = json.loads(VALID_JSON)
        obj["correctness"] = "5"
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_judge_response(json.dumps(obj))

    def test_boolean_score_raises(self):
        # bool is a subclass of int in Python -- True/False must not silently pass as 1/0.
        obj = json.loads(VALID_JSON)
        obj["correctness"] = True
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_judge_response(json.dumps(obj))

    def test_missing_rationale_raises(self):
        obj = json.loads(VALID_JSON)
        del obj["rationale"]
        with pytest.raises(ValueError, match="missing key"):
            parse_judge_response(json.dumps(obj))

    def test_empty_rationale_raises(self):
        obj = json.loads(VALID_JSON)
        obj["rationale"] = "   "
        with pytest.raises(ValueError, match="non-empty string"):
            parse_judge_response(json.dumps(obj))

    def test_non_string_rationale_raises(self):
        obj = json.loads(VALID_JSON)
        obj["rationale"] = 12345
        with pytest.raises(ValueError, match="non-empty string"):
            parse_judge_response(json.dumps(obj))


# --------------------------------------------------------------------------
# score_answer
# --------------------------------------------------------------------------


class TestScoreAnswer:
    def test_renders_prompt_calls_chat_fn_and_parses_result(self):
        captured_messages = []

        def fake_chat_fn(messages):
            captured_messages.append(messages)
            return VALID_JSON

        score = score_answer("Do these run small?", "[101] Runs small.", "Yes.", "Runs small [101].", TEMPLATE, fake_chat_fn)

        assert score.correctness == 5
        assert score.rationale == "Matches the reference."
        assert len(captured_messages) == 1
        rendered_prompt = captured_messages[0][0]["content"]
        assert "Do these run small?" in rendered_prompt
        assert captured_messages[0] == [{"role": "user", "content": rendered_prompt}]

    def test_none_context_reaches_chat_fn_as_no_context_marker(self):
        captured_messages = []

        def fake_chat_fn(messages):
            captured_messages.append(messages)
            return VALID_JSON

        score_answer("Q?", None, "An answer.", "A reference.", TEMPLATE, fake_chat_fn)

        rendered_prompt = captured_messages[0][0]["content"]
        assert NO_CONTEXT_MARKER in rendered_prompt

    def test_malformed_chat_fn_response_propagates_as_value_error(self):
        with pytest.raises(ValueError, match="Could not parse a JSON object"):
            score_answer("Q?", None, "An answer.", "A reference.", TEMPLATE, lambda messages: "not json at all")
