"""Unit tests for cragb.finetune.schema (T7.1; PLAN.md §3 E8, §10, M7.md T7.1).

Covers: `TrainingExample`'s JSON round-trip, its abstention-contradiction guard (the
§14.3 lesson, re-applied here since `TrainingExample` carries `is_abstention` as an
explicit field rather than deriving it), category validation, and -- the task's actual
point -- that `render_training_prompt` reproduces a training prompt that is
character-for-character identical to the prompt a real inference call would send for the
same `(question, context_text)` pair. The parity check is verified two ways: a synthetic,
file-independent determinism test, and (skipped if the real artifacts aren't present
locally, mirroring `tests/test_no_leakage.py`'s `TestRealLeakageManifestSelfConsistency`
pattern) a check against real rows from `results/tables/grounded_qa_transcripts_v1.jsonl`,
compared against the exact same reconstruction path `cragb.eval.cost_model
.build_messages_for_row` (T5.4) already uses and relies on for cost accounting.
"""

from __future__ import annotations

import json
from string import Template

import pytest

from cragb.bench.reference_answers import ABSTENTION_TEXT
from cragb.eval.cost_model import build_messages_for_row
from cragb.finetune.schema import (
    DEFAULT_PROMPT_TEMPLATE_PATH,
    TrainingExample,
    render_training_prompt,
    to_chat_messages,
)
from cragb.generate.grounded_qa import load_prompt_template
from cragb.utils.io import resolve_path


def make_example(
    example_id: str = "fit_sizing_ex_000",
    category: str = "fit_sizing",
    question: str = "Do these run true to size?",
    context_text: str = "[101] has_photo: no\nRuns small, size up.",
    answer: str = "These run small; size up for a better fit [101].",
    cited_doc_ids: tuple[str, ...] = ("101",),
    is_abstention: bool = False,
    source_doc_ids: tuple[str, ...] = ("101",),
    source_parent_asins: tuple[str, ...] = ("B000ASIN01",),
    provenance: dict | None = None,
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=source_doc_ids,
        source_parent_asins=source_parent_asins,
        question=question,
        context_text=context_text,
        answer=answer,
        cited_doc_ids=cited_doc_ids,
        is_abstention=is_abstention,
        provenance=provenance or {"method": "test_fixture"},
    )


def make_abstention_example(example_id: str = "fabric_quality_ex_000") -> TrainingExample:
    return make_example(
        example_id=example_id,
        category="fabric_quality",
        context_text="[202] has_photo: no\nSomewhat related but non-supporting text.",
        answer=ABSTENTION_TEXT,
        cited_doc_ids=(),
        is_abstention=True,
        source_doc_ids=("202",),
    )


# --------------------------------------------------------------------------
# TrainingExample: construction / validation
# --------------------------------------------------------------------------


class TestTrainingExampleValidation:
    def test_valid_positive_example_constructs(self):
        example = make_example()
        assert example.category == "fit_sizing"
        assert example.is_abstention is False

    def test_valid_abstention_example_constructs(self):
        example = make_abstention_example()
        assert example.is_abstention is True
        assert example.cited_doc_ids == ()

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="unknown category"):
            make_example(category="not_a_real_category")

    def test_abstention_text_with_citations_raises(self):
        # The §14.3 contradiction: abstention text plus a citation marker.
        with pytest.raises(ValueError, match="must not contain citations"):
            make_example(
                answer=ABSTENTION_TEXT + " [101]",
                cited_doc_ids=("101",),
                is_abstention=True,
            )

    def test_abstention_text_via_containment_not_equality(self):
        # A trailing citation still contains ABSTENTION_TEXT even though it
        # is no longer *equal* to it -- this is the exact case an
        # equality-based guard could never catch (PLAN.md §14.3).
        with pytest.raises(ValueError, match="must not contain citations"):
            make_example(
                answer=ABSTENTION_TEXT + " [999]",
                cited_doc_ids=("999",),
                is_abstention=False,  # flag doesn't even need to agree to be caught
            )

    def test_is_abstention_true_but_answer_lacks_abstention_text_raises(self):
        with pytest.raises(ValueError, match="is_abstention=True but answer does not contain"):
            make_example(answer="This runs small [101].", cited_doc_ids=("101",), is_abstention=True)

    def test_is_abstention_false_but_answer_is_abstention_text_raises(self):
        with pytest.raises(ValueError, match="is_abstention=False but answer does contain"):
            make_example(answer=ABSTENTION_TEXT, cited_doc_ids=(), is_abstention=False)

    def test_provenance_defaults_to_empty_dict(self):
        example = TrainingExample(
            example_id="x",
            category="value",
            source_doc_ids=("1",),
            source_parent_asins=("A1",),
            question="Is this worth the price?",
            context_text="[1] has_photo: no\ncheap and worth it",
            answer="Buyers feel it's worth the price [1].",
            cited_doc_ids=("1",),
            is_abstention=False,
        )
        assert example.provenance == {}


# --------------------------------------------------------------------------
# TrainingExample: to_dict / from_dict round-trip
# --------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_dict_from_dict_is_identity(self):
        example = make_example()
        assert TrainingExample.from_dict(example.to_dict()) == example

    def test_abstention_example_round_trips(self):
        example = make_abstention_example()
        assert TrainingExample.from_dict(example.to_dict()) == example

    def test_round_trips_through_json_text(self):
        # The actual on-disk shape: to_dict -> json.dumps -> json.loads -> from_dict.
        example = make_example()
        line = json.dumps(example.to_dict(), ensure_ascii=False)
        reloaded = TrainingExample.from_dict(json.loads(line))
        assert reloaded == example

    def test_to_dict_uses_lists_not_tuples(self):
        # JSON has no tuple type -- to_dict must hand back JSON-native lists.
        obj = make_example().to_dict()
        assert isinstance(obj["source_doc_ids"], list)
        assert isinstance(obj["cited_doc_ids"], list)
        assert isinstance(obj["source_parent_asins"], list)

    def test_from_dict_restores_tuples(self):
        example = TrainingExample.from_dict(make_example().to_dict())
        assert isinstance(example.source_doc_ids, tuple)
        assert isinstance(example.cited_doc_ids, tuple)
        assert isinstance(example.source_parent_asins, tuple)


# --------------------------------------------------------------------------
# render_training_prompt: synthetic determinism (no file dependency)
# --------------------------------------------------------------------------


class TestRenderTrainingPromptSynthetic:
    """A file-independent parity check: build a tiny stand-in template with the
    same `$question`/`$context_block` placeholders the real prompt uses, and
    confirm `render_training_prompt` substitutes exactly as
    `string.Template.substitute` would -- i.e. it is not silently
    reformatting, truncating, or re-deriving the context text.
    """

    def test_substitutes_question_and_context_text_verbatim(self):
        template = Template("Q: $question\n\n$context_block\n\nEND")
        example = make_example(
            question="Does this run small?", context_text="[101] has_photo: no\nRuns small."
        )
        prompt = render_training_prompt(example, template)
        assert prompt == "Q: Does this run small?\n\n[101] has_photo: no\nRuns small.\n\nEND"

    def test_default_template_loads_the_real_prompt_file(self):
        default_rendered = render_training_prompt(make_example())
        explicit_template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)
        explicit_rendered = render_training_prompt(make_example(), explicit_template)
        assert default_rendered == explicit_rendered

    def test_real_template_produces_the_documented_citation_rules(self):
        # Sanity check that the default template really is grounded_qa_v1.md
        # (not some other file) -- its rule text should appear verbatim.
        prompt = render_training_prompt(make_example())
        assert "Cite every claim." in prompt
        assert "Do these run true to size?" in prompt  # the question was substituted in

    def test_two_examples_with_the_same_context_but_different_questions_differ_only_there(self):
        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)
        a = render_training_prompt(make_example(question="Does this run small?"), template)
        b = render_training_prompt(make_example(question="Is the colour accurate?"), template)
        assert a != b
        assert "Does this run small?" in a
        assert "Is the colour accurate?" in b


# --------------------------------------------------------------------------
# render_training_prompt: parity against real generated transcripts
# --------------------------------------------------------------------------

_TRANSCRIPTS_PATH = "results/tables/grounded_qa_transcripts_v1.jsonl"
_CRAGB_PATH = "benchmark/cragb_v1.jsonl"


def _load_real_row(question_id: str) -> dict:
    with resolve_path(_TRANSCRIPTS_PATH).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["question_id"] == question_id:
                return row
    raise AssertionError(f"question_id {question_id!r} not found in {_TRANSCRIPTS_PATH}")


def _load_category(question_id: str) -> str:
    with resolve_path(_CRAGB_PATH).open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] == question_id:
                return entry["type"]
    raise AssertionError(f"question_id {question_id!r} not found in {_CRAGB_PATH}")


@pytest.mark.skipif(
    not (resolve_path(_TRANSCRIPTS_PATH).is_file() and resolve_path(_CRAGB_PATH).is_file()),
    reason="real grounded_qa_transcripts_v1.jsonl / cragb_v1.jsonl not present locally",
)
class TestRenderTrainingPromptParityWithRealTranscripts:
    """The task's actual point: a `TrainingExample` rebuilt from a real T4a.5
    transcript row must render to *exactly* the prompt that transcript was
    generated from -- reconstructed via `cragb.eval.cost_model
    .build_messages_for_row`, the same byte-identical-reconstruction path
    T5.4 already built and relies on for cost accounting. If this test
    fails, every training prompt this milestone generates is subtly
    different from what the model will see at inference, silently.
    """

    def _example_from_row(self, row: dict) -> TrainingExample:
        return TrainingExample(
            example_id=row["question_id"],
            category=_load_category(row["question_id"]),
            source_doc_ids=tuple(row["context_doc_ids"]),
            source_parent_asins=(),
            question=row["question"],
            context_text=row["context_text"],
            answer=row["answer_text"],
            cited_doc_ids=tuple(row["cited_doc_ids"]),
            is_abstention=row["abstained"],
            provenance={"source": "real_transcript_regression_test"},
        )

    def test_parity_on_a_positive_grounded_answer(self):
        row = _load_real_row("fit_sizing_neg_001")
        assert row["abstained"] is False  # sanity: this fixture really is a positive case
        example = self._example_from_row(row)
        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)

        expected_prompt = build_messages_for_row(row, template)[0]["content"]
        actual_prompt = render_training_prompt(example, template)

        assert actual_prompt == expected_prompt

    def test_parity_on_an_abstention_answer(self):
        row = _load_real_row("fabric_quality_neg_000")
        assert row["abstained"] is True  # sanity: this fixture really is an abstention case
        example = self._example_from_row(row)
        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)

        expected_prompt = build_messages_for_row(row, template)[0]["content"]
        actual_prompt = render_training_prompt(example, template)

        assert actual_prompt == expected_prompt

    def test_parity_holds_across_every_real_transcript_row(self):
        # Not just two hand-picked rows -- every transcript T4a.5 produced.
        with resolve_path(_TRANSCRIPTS_PATH).open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) > 0

        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)
        for row in rows:
            example = self._example_from_row(row)
            expected_prompt = build_messages_for_row(row, template)[0]["content"]
            actual_prompt = render_training_prompt(example, template)
            assert actual_prompt == expected_prompt, f"parity failed for {row['question_id']}"


# --------------------------------------------------------------------------
# to_chat_messages
# --------------------------------------------------------------------------


class TestToChatMessages:
    def test_returns_user_then_assistant_turn(self):
        example = make_example()
        messages = to_chat_messages(example)
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_user_turn_is_the_rendered_prompt(self):
        example = make_example()
        template = load_prompt_template(DEFAULT_PROMPT_TEMPLATE_PATH)
        messages = to_chat_messages(example, template)
        assert messages[0]["content"] == render_training_prompt(example, template)

    def test_assistant_turn_is_the_answer_verbatim(self):
        example = make_example(answer="These run small; size up [101].")
        messages = to_chat_messages(example)
        assert messages[1]["content"] == "These run small; size up [101]."

    def test_abstention_example_assistant_turn_is_exact_abstention_text(self):
        example = make_abstention_example()
        messages = to_chat_messages(example)
        assert messages[1]["content"] == ABSTENTION_TEXT
