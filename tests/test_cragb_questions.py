"""Unit tests for cragb.eval.cragb_questions (T3.4; M3.md T3.4).

Covers parsing a synthetic jsonl into `RetrievalQuestion`s, and checks
against the real `benchmark/cragb_v1.jsonl` that the loader sees all 60
questions and `filter_scorable` drops exactly the two genuinely-empty
negatives documented in PLAN.md §14.2
(`fabric_quality_neg_000`, `defects_neg_000`).
"""

from __future__ import annotations

import pytest

from cragb.eval.cragb_questions import (
    RetrievalQuestion,
    filter_scorable,
    load_retrieval_questions,
)


def write_jsonl(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    import json

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class TestLoadRetrievalQuestions:
    def test_parses_fields_correctly(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            [
                {
                    "id": "fit_sizing_000",
                    "type": "fit_sizing",
                    "question": "Do these run true to size?",
                    "is_negative": False,
                    "relevant_ids": ["1", "2", "3"],
                }
            ],
        )
        questions = load_retrieval_questions(path)
        assert questions == [
            RetrievalQuestion(
                id="fit_sizing_000",
                type="fit_sizing",
                question="Do these run true to size?",
                is_negative=False,
                relevant_ids=frozenset({"1", "2", "3"}),
            )
        ]

    def test_empty_relevant_ids_preserved_as_empty_frozenset(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            [
                {
                    "id": "neg_000",
                    "type": "defects",
                    "question": "unanswerable",
                    "is_negative": True,
                    "relevant_ids": [],
                }
            ],
        )
        questions = load_retrieval_questions(path)
        assert questions[0].relevant_ids == frozenset()

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        path.write_text(
            '{"id": "q1", "type": "value", "question": "q?", "is_negative": false, "relevant_ids": ["1"]}\n'
            "\n"
            "   \n",
            encoding="utf-8",
        )
        questions = load_retrieval_questions(path)
        assert len(questions) == 1

    def test_preserves_file_order(self, tmp_path):
        path = write_jsonl(
            tmp_path,
            [
                {"id": "q2", "type": "value", "question": "b", "is_negative": False, "relevant_ids": ["1"]},
                {"id": "q1", "type": "value", "question": "a", "is_negative": False, "relevant_ids": ["1"]},
            ],
        )
        questions = load_retrieval_questions(path)
        assert [q.id for q in questions] == ["q2", "q1"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_retrieval_questions("benchmark/does_not_exist.jsonl")

    def test_real_cragb_v1_has_60_questions(self):
        questions = load_retrieval_questions("benchmark/cragb_v1.jsonl")
        assert len(questions) == 60

    def test_real_cragb_v1_types_match_taxonomy(self):
        questions = load_retrieval_questions("benchmark/cragb_v1.jsonl")
        expected_types = {
            "fit_sizing", "colour_appearance", "fabric_quality",
            "durability", "defects", "occasion", "value",
        }
        assert {q.type for q in questions} == expected_types


class TestFilterScorable:
    def test_drops_empty_relevant_ids(self):
        questions = [
            RetrievalQuestion("q1", "value", "a?", False, frozenset({"1"})),
            RetrievalQuestion("q2", "value", "b?", True, frozenset()),
        ]
        scorable = filter_scorable(questions)
        assert [q.id for q in scorable] == ["q1"]

    def test_keeps_negatives_with_relevant_docs(self):
        questions = [
            RetrievalQuestion("q1", "value", "a?", True, frozenset({"1"})),
        ]
        assert filter_scorable(questions) == questions

    def test_real_cragb_v1_drops_exactly_the_two_known_empty_negatives(self):
        questions = load_retrieval_questions("benchmark/cragb_v1.jsonl")
        scorable = filter_scorable(questions)
        dropped = {q.id for q in questions} - {q.id for q in scorable}
        assert dropped == {"fabric_quality_neg_000", "defects_neg_000"}
        assert len(scorable) == 58
