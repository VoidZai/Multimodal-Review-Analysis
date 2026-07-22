"""Unit tests for cragb.bench.best_photo.

Covers: question loading (keeping `image_target`), the
relevant-∩-has_image candidate intersection (the core "don't surface an
irrelevant pooled photo" guarantee), candidate-snippet lookup, worksheet
rendering, selection loading (duplicate detection), and
`validate_selections`'s three checks — missing decisions, a selection
citing a non-candidate doc_id, and honest "no candidate" bookkeeping —
plus a self-consistency check against the real T2.3/T2.7/T2.9 artifacts
if present.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.bench.best_photo import (
    BestPhotoSelection,
    ImageQuestionRecord,
    export_worksheet,
    find_image_bearing_candidates,
    load_candidate_snippets,
    load_image_target_questions,
    load_selections,
    render_worksheet,
    validate_selections,
    write_best_photo_jsonl,
)
from cragb.utils.io import resolve_path


def make_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["Great colour, matches photo exactly.", "Runs small, no photo here.", "Nice fit, see photo."],
            "rating": [5.0, 3.0, 4.0],
            "has_image": [True, False, True],
            "image_urls": [["http://x/1.jpg"], [], ["http://x/3.jpg", "http://x/3b.jpg"]],
        },
        index=[10, 20, 30],
    )


class TestLoadImageTargetQuestions:
    def test_keeps_image_target_flag(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        path.write_text(
            '{"id": "q1", "category": "colour_appearance", "question": "Is colour as pictured?", '
            '"is_negative": false, "image_target": true}\n',
            encoding="utf-8",
        )
        questions = load_image_target_questions(path)
        assert questions == [ImageQuestionRecord("q1", "colour_appearance", "Is colour as pictured?", False, True)]


class TestFindImageBearingCandidates:
    def test_intersects_relevant_and_has_image(self):
        corpus = make_corpus()
        candidates = find_image_bearing_candidates({"10", "20", "30"}, corpus)
        assert candidates == ["10", "30"]  # 20 has no image

    def test_relevant_but_no_image_excluded(self):
        corpus = make_corpus()
        assert find_image_bearing_candidates({"20"}, corpus) == []

    def test_image_but_not_relevant_excluded(self):
        corpus = make_corpus()
        # doc 30 has an image but isn't in the relevant set
        assert find_image_bearing_candidates({"10"}, corpus) == ["10"]

    def test_sorted_numerically_not_lexically(self):
        corpus = pd.DataFrame(
            {"text": ["a", "b", "c"], "rating": [5.0, 5.0, 5.0], "has_image": [True, True, True]},
            index=[2, 10, 1],
        )
        assert find_image_bearing_candidates({"2", "10", "1"}, corpus) == ["1", "2", "10"]


class TestLoadCandidateSnippets:
    def test_returns_first_image_url(self):
        corpus = make_corpus()
        snippets = load_candidate_snippets(corpus, ["30"])
        assert snippets["30"].image_url == "http://x/3.jpg"

    def test_no_image_urls_is_none(self):
        corpus = make_corpus()
        snippets = load_candidate_snippets(corpus, ["20"])
        assert snippets["20"].image_url is None

    def test_truncates_text(self):
        corpus = make_corpus()
        snippets = load_candidate_snippets(corpus, ["10"], max_chars=5)
        assert len(snippets["10"].text) == 5


class TestRenderWorksheet:
    def test_skips_non_target_questions(self):
        questions = [ImageQuestionRecord("q1", "fit_sizing", "Q?", False, image_target=False)]
        md = render_worksheet(questions, {}, make_corpus(), "text", "has_image", "image_urls", 500)
        assert "q1" not in md

    def test_lists_candidates_for_target_question(self):
        questions = [ImageQuestionRecord("q1", "colour_appearance", "Colour as pictured?", False, True)]
        md = render_worksheet(questions, {"q1": {"10"}}, make_corpus(), "text", "has_image", "image_urls", 500)
        assert "q1" in md
        assert "`10`" in md
        assert "http://x/1.jpg" in md

    def test_no_candidate_case_is_explicit(self):
        questions = [ImageQuestionRecord("q1", "colour_appearance", "Colour as pictured?", False, True)]
        md = render_worksheet(questions, {"q1": {"20"}}, make_corpus(), "text", "has_image", "image_urls", 500)
        assert "no image-bearing relevant candidate found" in md


class TestExportWorksheet:
    def test_writes_file(self, tmp_path):
        questions = [ImageQuestionRecord("q1", "colour_appearance", "Colour as pictured?", False, True)]
        out_path = export_worksheet(
            questions, {"q1": {"10"}}, make_corpus(), tmp_path / "worksheet.md", "text", "has_image", "image_urls", 500
        )
        assert out_path.is_file()
        assert "q1" in out_path.read_text(encoding="utf-8")


class TestLoadSelections:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "selections.jsonl"
        path.write_text('{"question_id": "q1", "doc_id": "10", "reason": "clearly shows colour"}\n', encoding="utf-8")
        selections = load_selections(path)
        assert selections["q1"] == BestPhotoSelection("q1", "10", "clearly shows colour")

    def test_null_doc_id_round_trips(self, tmp_path):
        path = tmp_path / "selections.jsonl"
        path.write_text('{"question_id": "q1", "doc_id": null, "reason": "no candidate"}\n', encoding="utf-8")
        selections = load_selections(path)
        assert selections["q1"].doc_id is None

    def test_duplicate_question_id_raises(self, tmp_path):
        path = tmp_path / "selections.jsonl"
        path.write_text(
            '{"question_id": "q1", "doc_id": "10", "reason": "a"}\n'
            '{"question_id": "q1", "doc_id": "20", "reason": "b"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate best-photo selection"):
            load_selections(path)


class TestValidateSelections:
    def _questions(self):
        return [
            ImageQuestionRecord("q1", "colour_appearance", "Colour as pictured?", False, True),
            ImageQuestionRecord("q2", "fit_sizing", "Not a target", False, False),
        ]

    def test_valid_selection_passes(self):
        questions = self._questions()
        relevant = {"q1": {"10"}}
        selections = {"q1": BestPhotoSelection("q1", "10", "shows colour clearly")}
        report = validate_selections(questions, relevant, make_corpus(), selections)
        assert report.ok is True
        assert report.n_selected == 1
        assert report.coverage_pct == 100.0

    def test_missing_decision_is_hard_failure(self):
        questions = self._questions()
        report = validate_selections(questions, {"q1": {"10"}}, make_corpus(), {})
        assert report.ok is False
        assert report.missing_decisions == ("q1",)

    def test_invalid_doc_id_is_hard_failure(self):
        questions = self._questions()
        relevant = {"q1": {"10"}}
        selections = {"q1": BestPhotoSelection("q1", "999", "made up")}
        report = validate_selections(questions, relevant, make_corpus(), selections)
        assert report.ok is False
        assert report.invalid_selections == ("q1",)

    def test_honest_null_selection_with_real_candidates_is_hard_failure(self):
        questions = self._questions()
        relevant = {"q1": {"10"}}  # a real candidate exists
        selections = {"q1": BestPhotoSelection("q1", None, "skipped")}
        report = validate_selections(questions, relevant, make_corpus(), selections)
        assert report.ok is False
        assert report.invalid_selections == ("q1",)

    def test_null_selection_with_no_candidates_passes_and_is_counted(self):
        questions = self._questions()
        relevant = {"q1": {"20"}}  # relevant but no image
        selections = {"q1": BestPhotoSelection("q1", None, "no image-bearing relevant review")}
        report = validate_selections(questions, relevant, make_corpus(), selections)
        assert report.ok is True
        assert report.n_no_candidate == ("q1",)
        assert report.coverage_pct == 0.0

    def test_non_target_questions_are_ignored(self):
        questions = self._questions()
        report = validate_selections(questions, {}, make_corpus(), {"q1": BestPhotoSelection("q1", None, "x")})
        assert report.n_image_target_questions == 1  # q2 excluded


class TestWriteBestPhotoJsonl:
    def test_writes_only_image_target_questions(self, tmp_path):
        questions = [
            ImageQuestionRecord("q1", "colour_appearance", "Q1?", False, True),
            ImageQuestionRecord("q2", "fit_sizing", "Q2?", False, False),
        ]
        selections = {
            "q1": BestPhotoSelection("q1", "10", "clear evidence"),
            "q2": BestPhotoSelection("q2", "20", "should be dropped, not a target"),
        }
        out_path = write_best_photo_jsonl(questions, selections, tmp_path / "out.jsonl")
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0] == {"question_id": "q1", "doc_id": "10", "reason": "clear evidence"}


@pytest.mark.skipif(
    not resolve_path("benchmark/best_photo_v1.jsonl").is_file(),
    reason="benchmark/best_photo_v1.jsonl not present locally",
)
class TestRealBestPhotoSelfConsistency:
    """No corpus dependency for the parts that don't need it; the full
    validation still needs corpus_v1.parquet, so that half is skipped if
    it's not present (git-ignored)."""

    def test_every_image_target_question_has_a_selection(self):
        from cragb.bench.reference_answers import load_relevant_doc_ids

        questions = load_image_target_questions("benchmark/cragb_questions_v1.jsonl")
        selections = load_selections("benchmark/best_photo_v1.jsonl")
        target_ids = {q.id for q in questions if q.image_target}
        assert target_ids == set(selections)

    @pytest.mark.skipif(
        not resolve_path("data/processed/corpus_v1.parquet").is_file(),
        reason="data/processed/corpus_v1.parquet not present locally",
    )
    def test_real_selections_pass_validation(self):
        from cragb.bench.reference_answers import load_relevant_doc_ids

        questions = load_image_target_questions("benchmark/cragb_questions_v1.jsonl")
        relevant = load_relevant_doc_ids("benchmark/relevance_labels_v1.jsonl")
        corpus = pd.read_parquet(resolve_path("data/processed/corpus_v1.parquet"))
        selections = load_selections("benchmark/best_photo_v1.jsonl")

        report = validate_selections(questions, relevant, corpus, selections)
        assert report.ok, report.messages
