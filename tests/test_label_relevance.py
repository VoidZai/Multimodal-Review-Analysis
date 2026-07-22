"""Unit tests for cragb.bench.label_relevance.

Covers: pool+question loading and joining, corpus-snippet lookup and
truncation, worksheet rendering, label loading (including duplicate
detection), and `validate_labels`'s completeness check and its two
distinct flags — a non-negative question with zero relevant docs, and a
negative question that unexpectedly has one — which are independent of
each other and of the hard "every pair labeled" requirement.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.bench.label_relevance import (
    CorpusSnippet,
    LabelDecision,
    PoolRecord,
    export_worksheet,
    load_corpus_snippets,
    load_labels,
    load_pools,
    render_worksheet,
    validate_labels,
    write_labels_jsonl,
)


def write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


class TestLoadPools:
    def test_joins_question_text_from_questions_file(self, tmp_path):
        pools_path = tmp_path / "pools.jsonl"
        questions_path = tmp_path / "questions.jsonl"
        write_jsonl(
            pools_path,
            [{"question_id": "q1", "category": "fit_sizing", "is_negative": False, "doc_ids": ["1", "2"]}],
        )
        write_jsonl(
            questions_path,
            [{"id": "q1", "category": "fit_sizing", "question": "Does this run small?", "is_negative": False, "image_target": True}],
        )
        pools = load_pools(pools_path, questions_path)
        assert pools == [PoolRecord("q1", "fit_sizing", False, "Does this run small?", ("1", "2"))]

    def test_missing_question_mapping_raises(self, tmp_path):
        pools_path = tmp_path / "pools.jsonl"
        questions_path = tmp_path / "questions.jsonl"
        write_jsonl(pools_path, [{"question_id": "ghost", "category": "fit_sizing", "is_negative": False, "doc_ids": []}])
        write_jsonl(questions_path, [])
        with pytest.raises(ValueError, match="no matching entry"):
            load_pools(pools_path, questions_path)


class TestLoadCorpusSnippets:
    def _corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"text": ["Runs small, size up.", "Long review " * 50], "rating": [4.0, 2.0], "has_image": [True, False]},
            index=[10, 20],
        )

    def test_looks_up_by_stringified_index(self):
        snippets = load_corpus_snippets(self._corpus(), {"10"}, max_chars=600)
        assert snippets["10"] == CorpusSnippet(text="Runs small, size up.", rating=4.0, has_image=True)

    def test_truncates_to_max_chars(self):
        snippets = load_corpus_snippets(self._corpus(), {"20"}, max_chars=20)
        assert len(snippets["20"].text) == 20

    def test_ids_not_in_corpus_are_absent(self):
        snippets = load_corpus_snippets(self._corpus(), {"10", "999"}, max_chars=600)
        assert "999" not in snippets
        assert "10" in snippets


class TestRenderWorksheet:
    def test_contains_question_and_candidates(self):
        pools = [PoolRecord("q1", "fit_sizing", False, "Does this run small?", ("10",))]
        snippets = {"10": CorpusSnippet(text="Runs small.", rating=4.0, has_image=True)}
        md = render_worksheet(pools, snippets)
        assert "q1" in md
        assert "Does this run small?" in md
        assert "Runs small." in md
        assert "[has image]" in md

    def test_negative_question_is_tagged(self):
        pools = [PoolRecord("q1_neg", "fit_sizing", True, "Exact chest measurement?", ())]
        md = render_worksheet(pools, {})
        assert "NEGATIVE" in md

    def test_missing_snippet_handled_gracefully(self):
        pools = [PoolRecord("q1", "fit_sizing", False, "Q?", ("missing_id",))]
        md = render_worksheet(pools, {})
        assert "text not found" in md


class TestExportWorksheet:
    def test_writes_file(self, tmp_path):
        pools = [PoolRecord("q1", "fit_sizing", False, "Does this run small?", ("10",))]
        corpus = pd.DataFrame({"text": ["Runs small."], "rating": [4.0], "has_image": [False]}, index=[10])
        out_path = export_worksheet(pools, corpus, tmp_path / "worksheet.md", text_col="text", max_chars=600)
        assert out_path.is_file()
        assert "Does this run small?" in out_path.read_text(encoding="utf-8")


class TestLoadLabels:
    def test_round_trips_labels(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        write_jsonl(path, [{"question_id": "q1", "doc_id": "10", "relevant": True, "reason": "matches"}])
        labels = load_labels(path)
        assert labels[("q1", "10")] == LabelDecision("q1", "10", True, "matches")

    def test_missing_reason_defaults_to_empty_string(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        write_jsonl(path, [{"question_id": "q1", "doc_id": "10", "relevant": False}])
        labels = load_labels(path)
        assert labels[("q1", "10")].reason == ""

    def test_duplicate_pair_raises(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        write_jsonl(
            path,
            [
                {"question_id": "q1", "doc_id": "10", "relevant": True},
                {"question_id": "q1", "doc_id": "10", "relevant": False},
            ],
        )
        with pytest.raises(ValueError, match="Duplicate label"):
            load_labels(path)


class TestValidateLabels:
    def _pools(self):
        return [
            PoolRecord("q1", "fit_sizing", False, "Q1?", ("a", "b")),
            PoolRecord("q1_neg", "fit_sizing", True, "Q1 negative?", ("c",)),
        ]

    def test_complete_labels_with_a_positive_pass(self):
        labels = {
            ("q1", "a"): LabelDecision("q1", "a", True, "yes"),
            ("q1", "b"): LabelDecision("q1", "b", False, "no"),
            ("q1_neg", "c"): LabelDecision("q1_neg", "c", False, "no"),
        }
        report = validate_labels(self._pools(), labels)
        assert report.ok is True
        assert report.missing_pairs == ()
        assert report.n_questions_with_relevant == 1
        assert report.pct_questions_with_relevant == 50.0
        assert report.zero_positive_questions == ()
        assert report.unexpected_positive_negatives == ()

    def test_missing_pair_is_a_hard_failure(self):
        labels = {("q1", "a"): LabelDecision("q1", "a", True, "yes")}  # "b" and "c" missing
        report = validate_labels(self._pools(), labels)
        assert report.ok is False
        assert set(report.missing_pairs) == {("q1", "b"), ("q1_neg", "c")}

    def test_zero_positive_answerable_question_is_flagged_not_failed(self):
        labels = {
            ("q1", "a"): LabelDecision("q1", "a", False, "no"),
            ("q1", "b"): LabelDecision("q1", "b", False, "no"),
            ("q1_neg", "c"): LabelDecision("q1_neg", "c", False, "no"),
        }
        report = validate_labels(self._pools(), labels)
        assert report.ok is True  # complete, just flagged
        assert report.zero_positive_questions == ("q1",)

    def test_unexpected_positive_negative_is_flagged(self):
        labels = {
            ("q1", "a"): LabelDecision("q1", "a", True, "yes"),
            ("q1", "b"): LabelDecision("q1", "b", False, "no"),
            ("q1_neg", "c"): LabelDecision("q1_neg", "c", True, "surprisingly relevant"),
        }
        report = validate_labels(self._pools(), labels)
        assert report.ok is True
        assert report.unexpected_positive_negatives == ("q1_neg",)

    def test_total_and_labeled_pair_counts(self):
        labels = {
            ("q1", "a"): LabelDecision("q1", "a", True, "yes"),
            ("q1", "b"): LabelDecision("q1", "b", False, "no"),
            ("q1_neg", "c"): LabelDecision("q1_neg", "c", False, "no"),
        }
        report = validate_labels(self._pools(), labels)
        assert report.total_pairs == 3
        assert report.labeled_pairs == 3


class TestWriteLabelsJsonl:
    def test_writes_only_pooled_and_labeled_pairs(self, tmp_path):
        pools = [PoolRecord("q1", "fit_sizing", False, "Q1?", ("a", "b"))]
        labels = {
            ("q1", "a"): LabelDecision("q1", "a", True, "matches"),
            ("q1", "stray"): LabelDecision("q1", "stray", True, "not in any pool"),
            # "b" intentionally unlabeled
        }
        out_path = write_labels_jsonl(pools, labels, tmp_path / "labels_out.jsonl")
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0] == {"question_id": "q1", "doc_id": "a", "relevant": 1, "reason": "matches"}
