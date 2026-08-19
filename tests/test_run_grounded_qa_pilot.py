"""Unit tests for cragb.eval.run_grounded_qa_pilot (T4a.5; M4a.md T4a.5).

Covers the pure logic this module adds on top of the already-tested T4a.2-T4a.4
building blocks: curated-question selection/ordering and the pre-write validation
guard. `main()` itself (real corpus, real API, real file writes) is intentionally not
unit-tested here -- the same convention `cragb.generate.draft_questions.main` and
`cragb.eval.run_retrieval_eval.main` already follow in this project.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.run_grounded_qa_pilot import (
    CURATED_QUESTION_IDS,
    select_pilot_questions,
    validate_pilot_run,
    write_csv,
)
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript


def make_question(qid: str) -> RetrievalQuestion:
    return RetrievalQuestion(
        id=qid, type="fit_sizing", question=f"question for {qid}?",
        is_negative=False, relevant_ids=frozenset({"101"}),
    )


def make_transcript(qid: str, answer_text: str = "Runs small [101].") -> GroundedQATranscript:
    context = ContextBlock(text="ctx", doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question="q?", context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=("101",) if "[101]" in answer_text else (),
        cited_photo_ids=(), abstained=False,
    )


# --------------------------------------------------------------------------
# CURATED_QUESTION_IDS
# --------------------------------------------------------------------------


class TestCuratedQuestionIds:
    def test_at_least_ten_questions(self):
        assert len(CURATED_QUESTION_IDS) >= 10

    def test_no_duplicate_ids(self):
        assert len(CURATED_QUESTION_IDS) == len(set(CURATED_QUESTION_IDS))

    def test_includes_both_genuine_abstentions(self):
        # PLAN.md §14.2: the only two CRAGB v1 negatives whose pools came back empty.
        assert "fabric_quality_neg_000" in CURATED_QUESTION_IDS
        assert "defects_neg_000" in CURATED_QUESTION_IDS

    def test_includes_a_surprisingly_answerable_negative(self):
        assert "fit_sizing_neg_001" in CURATED_QUESTION_IDS


# --------------------------------------------------------------------------
# select_pilot_questions
# --------------------------------------------------------------------------


class TestSelectPilotQuestions:
    def test_returns_questions_in_requested_order_not_source_order(self):
        all_questions = [make_question("b"), make_question("a"), make_question("c")]
        selected = select_pilot_questions(all_questions, question_ids=("c", "a"))
        assert [q.id for q in selected] == ["c", "a"]

    def test_missing_id_raises_value_error(self):
        all_questions = [make_question("a")]
        with pytest.raises(ValueError, match="not found"):
            select_pilot_questions(all_questions, question_ids=("a", "does_not_exist"))

    def test_default_ids_select_against_real_shaped_question_list(self):
        all_questions = [make_question(qid) for qid in CURATED_QUESTION_IDS] + [make_question("extra")]
        selected = select_pilot_questions(all_questions)
        assert [q.id for q in selected] == list(CURATED_QUESTION_IDS)


# --------------------------------------------------------------------------
# validate_pilot_run
# --------------------------------------------------------------------------


class TestValidatePilotRun:
    def test_passes_on_matching_nonempty_transcripts(self):
        transcripts = [make_transcript("q1"), make_transcript("q2")]
        validate_pilot_run(transcripts, expected_question_ids=("q1", "q2"))  # no raise

    def test_raises_on_id_mismatch(self):
        transcripts = [make_transcript("q1")]
        with pytest.raises(ValueError, match="do not match"):
            validate_pilot_run(transcripts, expected_question_ids=("q1", "q2"))

    def test_raises_on_wrong_order(self):
        transcripts = [make_transcript("q1"), make_transcript("q2")]
        with pytest.raises(ValueError, match="do not match"):
            validate_pilot_run(transcripts, expected_question_ids=("q2", "q1"))

    def test_raises_on_empty_answer_text(self):
        # Reproduces the exact failure T4a.3 hit for real against the live
        # API (a max_tokens cap too low for a reasoning model's visible
        # answer) -- this is the regression guard for it.
        transcripts = [make_transcript("q1", answer_text="Runs small [101]."), make_transcript("q2", answer_text="")]
        with pytest.raises(ValueError, match="empty answer_text"):
            validate_pilot_run(transcripts, expected_question_ids=("q1", "q2"))

    def test_raises_on_whitespace_only_answer_text(self):
        transcripts = [make_transcript("q1", answer_text="   \n  ")]
        with pytest.raises(ValueError, match="empty answer_text"):
            validate_pilot_run(transcripts, expected_question_ids=("q1",))


# --------------------------------------------------------------------------
# write_csv
# --------------------------------------------------------------------------


class TestWriteCsv:
    def test_writes_dataframe_and_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame([{"a": 1, "b": 2}])
        out_path = write_csv(df, tmp_path / "nested" / "out.csv")
        assert out_path.is_file()
        read_back = pd.read_csv(out_path)
        assert read_back.iloc[0]["a"] == 1
