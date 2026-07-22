"""Unit tests for cragb.bench.reference_answers.

Covers: citation extraction from answer prose (order, de-duplication),
abstention detection and its citation-contradiction guard, relevant-id
loading, reference-answer loading (including duplicate detection), and
`validate_reference_answers`'s four hard-failure checks plus its one
soft flag — all independent of each other, per the module's own
docstring. Also checks the real `benchmark/reference_answers_v1.jsonl`
(if present) against the real `cragb_questions_v1.jsonl` and
`relevance_labels_v1.jsonl` for internal self-consistency.
"""

from __future__ import annotations

import json

import pytest

from cragb.bench.pooling import QuestionRecord
from cragb.bench.reference_answers import (
    ABSTENTION_TEXT,
    ReferenceAnswer,
    extract_citations,
    load_reference_answers,
    load_relevant_doc_ids,
    make_reference_answer,
    validate_reference_answers,
    write_reference_answers_jsonl,
)
from cragb.utils.io import resolve_path


class TestExtractCitations:
    def test_extracts_in_first_seen_order(self):
        text = "Runs small [128775] and small again [161398] and small once more [128775]."
        assert extract_citations(text) == ("128775", "161398")

    def test_no_citations_returns_empty_tuple(self):
        assert extract_citations("No citations here.") == ()

    def test_deduplicates(self):
        text = "[1][1][2][1]"
        assert extract_citations(text) == ("1", "2")


class TestMakeReferenceAnswer:
    def test_parses_citations_and_marks_non_abstention(self):
        answer = make_reference_answer("q1", "Runs small [128775][161398].")
        assert answer.cited_doc_ids == ("128775", "161398")
        assert answer.is_abstention is False
        assert answer.answer == "Runs small [128775][161398]."

    def test_canonical_abstention_text_is_flagged(self):
        answer = make_reference_answer("q1", ABSTENTION_TEXT)
        assert answer.is_abstention is True
        assert answer.cited_doc_ids == ()

    def test_abstention_text_with_citations_raises(self):
        with pytest.raises(ValueError, match="must not contain citations"):
            make_reference_answer("q1", ABSTENTION_TEXT + " [999]")

    def test_strips_surrounding_whitespace(self):
        answer = make_reference_answer("q1", "  Runs small [1].  \n")
        assert answer.answer == "Runs small [1]."


class TestLoadRelevantDocIds:
    def test_only_relevant_rows_are_kept(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"question_id": "q1", "doc_id": "a", "relevant": 1, "reason": ""}\n'
            '{"question_id": "q1", "doc_id": "b", "relevant": 0, "reason": ""}\n'
            '{"question_id": "q2", "doc_id": "c", "relevant": 1, "reason": ""}\n',
            encoding="utf-8",
        )
        relevant = load_relevant_doc_ids(path)
        assert relevant == {"q1": {"a"}, "q2": {"c"}}

    def test_question_with_no_relevant_rows_is_absent(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text('{"question_id": "q1", "doc_id": "a", "relevant": 0, "reason": ""}\n', encoding="utf-8")
        assert load_relevant_doc_ids(path) == {}


class TestLoadReferenceAnswers:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        path.write_text(
            '{"question_id": "q1", "answer": "Runs small [a].", "cited_doc_ids": ["a"], "is_abstention": false}\n',
            encoding="utf-8",
        )
        answers = load_reference_answers(path)
        assert answers["q1"] == ReferenceAnswer("q1", "Runs small [a].", ("a",), False)

    def test_duplicate_question_id_raises(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        path.write_text(
            '{"question_id": "q1", "answer": "A [x].", "cited_doc_ids": ["x"], "is_abstention": false}\n'
            '{"question_id": "q1", "answer": "B [y].", "cited_doc_ids": ["y"], "is_abstention": false}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate reference answer"):
            load_reference_answers(path)


def make_q(id_="q1", is_negative=False) -> QuestionRecord:
    return QuestionRecord(id=id_, category="fit_sizing", is_negative=is_negative, question="Q?")


class TestValidateReferenceAnswers:
    def test_well_formed_grounded_answer_passes(self):
        questions = [make_q("q1")]
        relevant = {"q1": {"a", "b"}}
        answers = {"q1": make_reference_answer("q1", "Runs small [a].")}
        report = validate_reference_answers(questions, relevant, answers)
        assert report.ok is True
        assert report.missing_questions == ()
        assert report.citation_violations == ()

    def test_missing_answer_is_hard_failure(self):
        questions = [make_q("q1"), make_q("q2")]
        report = validate_reference_answers(questions, {}, {"q1": make_reference_answer("q1", ABSTENTION_TEXT)})
        assert report.ok is False
        assert report.missing_questions == ("q2",)

    def test_citation_not_in_relevant_set_is_hard_failure(self):
        questions = [make_q("q1")]
        relevant = {"q1": {"a"}}
        answers = {"q1": make_reference_answer("q1", "Runs small [a][ghost].")}
        report = validate_reference_answers(questions, relevant, answers)
        assert report.ok is False
        assert report.citation_violations == ("q1",)

    def test_non_abstention_with_no_citations_is_hard_failure(self):
        questions = [make_q("q1")]
        relevant = {"q1": {"a"}}
        answers = {"q1": make_reference_answer("q1", "Runs small, no citation.")}
        report = validate_reference_answers(questions, relevant, answers)
        assert report.ok is False
        assert report.ungrounded_answers == ("q1",)

    def test_answering_with_zero_relevant_evidence_is_hard_failure(self):
        questions = [make_q("q1")]
        # relevant dict has no entry for q1 at all -> zero evidence
        answers = {"q1": make_reference_answer("q1", "Runs small [a].")}
        report = validate_reference_answers(questions, {}, answers)
        assert report.ok is False
        # 'a' is also a citation violation (not in the empty relevant set) --
        # both checks legitimately fire together here.
        assert report.false_confidence == ("q1",)

    def test_correct_abstention_with_zero_evidence_passes(self):
        questions = [make_q("q1_neg", is_negative=True)]
        answers = {"q1_neg": make_reference_answer("q1_neg", ABSTENTION_TEXT)}
        report = validate_reference_answers(questions, {}, answers)
        assert report.ok is True

    def test_abstention_despite_available_evidence_is_a_soft_flag(self):
        questions = [make_q("q1_neg", is_negative=True)]
        relevant = {"q1_neg": {"a", "b"}}
        answers = {"q1_neg": make_reference_answer("q1_neg", ABSTENTION_TEXT)}
        report = validate_reference_answers(questions, relevant, answers)
        assert report.ok is True  # soft flag only
        assert report.unjustified_abstentions == ("q1_neg",)

    def test_clean_run_has_a_positive_summary_message(self):
        questions = [make_q("q1")]
        relevant = {"q1": {"a"}}
        answers = {"q1": make_reference_answer("q1", "Runs small [a].")}
        report = validate_reference_answers(questions, relevant, answers)
        assert "strictly-grounded reference answer" in report.messages[0]


class TestWriteReferenceAnswersJsonl:
    def test_writes_in_question_order_and_skips_missing(self, tmp_path):
        questions = [make_q("q1"), make_q("q2"), make_q("q3")]
        answers = {
            "q3": make_reference_answer("q3", "Answer three [c]."),
            "q1": make_reference_answer("q1", "Answer one [a]."),
            # q2 intentionally has no answer
        }
        out_path = write_reference_answers_jsonl(questions, answers, tmp_path / "out.jsonl")
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert [r["question_id"] for r in rows] == ["q1", "q3"]
        assert rows[0]["cited_doc_ids"] == ["a"]
        assert rows[0]["is_abstention"] is False


@pytest.mark.skipif(
    not resolve_path("benchmark/reference_answers_v1.jsonl").is_file(),
    reason="benchmark/reference_answers_v1.jsonl not present locally",
)
class TestRealReferenceAnswersSelfConsistency:
    """Sanity-checks the real T2.8 artifact against the real T2.3/T2.7 files.

    No corpus dependency, so this runs in a fresh checkout as long as
    the committed benchmark/ files are present.
    """

    def test_real_answers_pass_validation(self):
        from cragb.bench.pooling import load_questions

        questions = load_questions("benchmark/cragb_questions_v1.jsonl")
        relevant = load_relevant_doc_ids("benchmark/relevance_labels_v1.jsonl")
        answers = load_reference_answers("benchmark/reference_answers_v1.jsonl")

        report = validate_reference_answers(questions, relevant, answers)
        assert report.ok, report.messages
        assert report.answered == report.total_questions
