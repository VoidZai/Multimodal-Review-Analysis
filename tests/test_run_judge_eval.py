"""Unit tests for cragb.eval.run_judge_eval (T4b.5; M4b.md T4b.5).

Covers the pure logic this module adds on top of the already-tested T4b.1-T4b.4
building blocks: transcript-type dispatch, the closed-book-gets-no-context rule, batch
scoring, and the pre-write completeness/range validation guard. `main()` itself (real
transcript files, real judge config, real API, real file write) is intentionally not
unit-tested here -- the same convention `cragb.eval.run_grounded_qa_pilot`'s and
`cragb.eval.run_answer_generation`'s own tests already follow for the same kind of
module.
"""

from __future__ import annotations

from string import Template

import pandas as pd
import pytest

from cragb.bench.reference_answers import make_reference_answer
from cragb.eval.run_judge_eval import (
    context_text_for,
    load_arm_transcripts,
    score_transcripts,
    validate_judge_scores,
)
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript, write_transcripts_jsonl as write_grounded_qa_jsonl
from cragb.generate.closed_book_qa import write_transcripts_jsonl as write_closed_book_jsonl

JUDGE_TEMPLATE = Template("Q: $question\nCtx: $context_block\nCand: $candidate_answer\nRef: $reference_answer")


def make_closed_book_transcript(qid: str, question: str = "Q?", answer_text: str = "An answer.") -> ClosedBookTranscript:
    return ClosedBookTranscript(
        question_id=qid, question=question, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), abstained=False,
    )


def make_grounded_transcript(
    qid: str, question: str = "Q?", answer_text: str = "Runs small [101].", context_text: str = "[101] Runs small."
) -> GroundedQATranscript:
    context = ContextBlock(text=context_text, doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question=question, context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=("101",) if "[101]" in answer_text else (),
        cited_photo_ids=(), abstained=False,
    )


VALID_JUDGE_JSON = (
    '{"correctness": 5, "faithfulness": 4, "completeness": 5, "conciseness": 3, "rationale": "Solid match."}'
)


# --------------------------------------------------------------------------
# load_arm_transcripts
# --------------------------------------------------------------------------


class TestLoadArmTranscripts:
    def test_loads_closed_book_transcripts(self, tmp_path):
        path = write_closed_book_jsonl([make_closed_book_transcript("q1")], tmp_path / "cb.jsonl")
        loaded = load_arm_transcripts("closed_book", path)
        assert len(loaded) == 1
        assert isinstance(loaded[0], ClosedBookTranscript)

    def test_loads_rag_small_transcripts_as_grounded_qa_transcripts(self, tmp_path):
        path = write_grounded_qa_jsonl([make_grounded_transcript("q1")], tmp_path / "rag.jsonl")
        loaded = load_arm_transcripts("rag_small", path)
        assert len(loaded) == 1
        assert isinstance(loaded[0], GroundedQATranscript)

    def test_loads_rag_large_transcripts_as_grounded_qa_transcripts(self, tmp_path):
        path = write_grounded_qa_jsonl([make_grounded_transcript("q1")], tmp_path / "rag.jsonl")
        loaded = load_arm_transcripts("rag_large", path)
        assert isinstance(loaded[0], GroundedQATranscript)

    def test_unknown_arm_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown arm"):
            load_arm_transcripts("not_a_real_arm", tmp_path / "whatever.jsonl")


# --------------------------------------------------------------------------
# context_text_for
# --------------------------------------------------------------------------


class TestContextTextFor:
    def test_closed_book_is_always_none(self):
        transcript = make_closed_book_transcript("q1")
        assert context_text_for("closed_book", transcript) is None

    def test_rag_small_returns_the_transcripts_own_context_text(self):
        transcript = make_grounded_transcript("q1", context_text="[101] Runs small, size up.")
        assert context_text_for("rag_small", transcript) == "[101] Runs small, size up."

    def test_rag_large_returns_the_transcripts_own_context_text(self):
        transcript = make_grounded_transcript("q1", context_text="[202] Fits true to size.")
        assert context_text_for("rag_large", transcript) == "[202] Fits true to size."


# --------------------------------------------------------------------------
# score_transcripts
# --------------------------------------------------------------------------


class TestScoreTranscripts:
    def test_scores_every_transcript_and_tags_the_arm(self):
        transcripts = [
            make_closed_book_transcript("q1", question="Does it run small?"),
            make_closed_book_transcript("q2", question="Is it durable?"),
        ]
        references = {
            "q1": make_reference_answer("q1", "Runs small [101]."),
            "q2": make_reference_answer("q2", "Holds up well [202]."),
        }

        result = score_transcripts("closed_book", transcripts, references, JUDGE_TEMPLATE, lambda m: VALID_JUDGE_JSON)

        assert list(result.columns) == [
            "arm", "question_id", "correctness", "faithfulness", "completeness", "conciseness", "rationale",
        ]
        assert (result["arm"] == "closed_book").all()
        assert list(result["question_id"]) == ["q1", "q2"]
        assert result["correctness"].tolist() == [5, 5]

    def test_closed_book_prompt_shown_to_judge_carries_no_context(self):
        captured_prompts = []

        def fake_chat_fn(messages):
            captured_prompts.append(messages[0]["content"])
            return VALID_JUDGE_JSON

        transcripts = [make_closed_book_transcript("q1", question="Q?", answer_text="No idea.")]
        references = {"q1": make_reference_answer("q1", "Reference text.")}

        score_transcripts("closed_book", transcripts, references, JUDGE_TEMPLATE, fake_chat_fn)

        assert "Ctx: " in captured_prompts[0]
        # The template's own $context_block placeholder got *something* neutral, not
        # the transcript's own context (closed-book transcripts have none anyway).
        assert "[101]" not in captured_prompts[0]

    def test_rag_prompt_shown_to_judge_carries_the_real_context(self):
        captured_prompts = []

        def fake_chat_fn(messages):
            captured_prompts.append(messages[0]["content"])
            return VALID_JUDGE_JSON

        transcripts = [make_grounded_transcript("q1", context_text="[101] Runs small, size up.")]
        references = {"q1": make_reference_answer("q1", "Runs small [101].")}

        score_transcripts("rag_small", transcripts, references, JUDGE_TEMPLATE, fake_chat_fn)

        assert "[101] Runs small, size up." in captured_prompts[0]

    def test_missing_reference_raises_key_error(self):
        transcripts = [make_closed_book_transcript("q1")]
        with pytest.raises(KeyError, match="q1"):
            score_transcripts("closed_book", transcripts, references={}, template=JUDGE_TEMPLATE, chat_fn=lambda m: VALID_JUDGE_JSON)


# --------------------------------------------------------------------------
# validate_judge_scores
# --------------------------------------------------------------------------


def make_scores_df(rows: list[dict]) -> pd.DataFrame:
    columns = ["arm", "question_id", "correctness", "faithfulness", "completeness", "conciseness", "rationale"]
    return pd.DataFrame(rows, columns=columns)


def full_grid_row(arm: str, qid: str, **overrides) -> dict:
    row = {
        "arm": arm, "question_id": qid, "correctness": 5, "faithfulness": 5,
        "completeness": 5, "conciseness": 5, "rationale": "Good.",
    }
    row.update(overrides)
    return row


class TestValidateJudgeScores:
    def test_passes_on_a_complete_grid(self):
        rows = [full_grid_row(arm, qid) for arm in ("closed_book", "rag_small") for qid in ("q1", "q2")]
        validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book", "rag_small"), expected_question_ids=("q1", "q2"))  # no raise

    def test_raises_on_wrong_row_count(self):
        rows = [full_grid_row("closed_book", "q1")]
        with pytest.raises(ValueError, match="Expected 4 judge score row"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book", "rag_small"), expected_question_ids=("q1", "q2"))

    def test_raises_on_missing_pair(self):
        rows = [
            full_grid_row("closed_book", "q1"),
            full_grid_row("closed_book", "q2"),
            full_grid_row("rag_small", "q1"),
            full_grid_row("rag_small", "q1"),  # duplicate, not q2 -- wrong count masks missing q2 too
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book", "rag_small"), expected_question_ids=("q1", "q2"))

    def test_raises_on_unexpected_extra_pair(self):
        rows = [full_grid_row("closed_book", "q1"), full_grid_row("closed_book", "q2"), full_grid_row("closed_book", "q3")]
        with pytest.raises(ValueError, match="Expected 2 judge score row"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book",), expected_question_ids=("q1", "q2"))

    def test_raises_on_out_of_range_score(self):
        rows = [full_grid_row("closed_book", "q1", correctness=6), full_grid_row("closed_book", "q2")]
        with pytest.raises(ValueError, match="out of \\[1, 5\\] range"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book",), expected_question_ids=("q1", "q2"))

    def test_raises_on_null_score(self):
        rows = [full_grid_row("closed_book", "q1", correctness=None), full_grid_row("closed_book", "q2")]
        with pytest.raises(ValueError, match="null rubric"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book",), expected_question_ids=("q1", "q2"))

    def test_raises_on_empty_rationale(self):
        rows = [full_grid_row("closed_book", "q1", rationale="   "), full_grid_row("closed_book", "q2")]
        with pytest.raises(ValueError, match="empty/null rationale"):
            validate_judge_scores(make_scores_df(rows), expected_arms=("closed_book",), expected_question_ids=("q1", "q2"))
