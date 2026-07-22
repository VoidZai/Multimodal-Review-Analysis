"""Unit tests for cragb.bench.assemble (assembly, validation, datasheet stats).

Leakage-guard specific tests live in tests/test_no_leakage.py, matching
M2.md T2.10's named test file. This file covers: merging T2.3/T2.7/T2.8/
T2.9 outputs into `CragbEntry`s (including the two failure modes — a
missing reference answer, and an image-target question with no photo
decision), `validate_entries`'s three cross-task re-checks, datasheet
stat computation, and JSONL/manifest writing. Also self-consistency
checks the real `benchmark/cragb_v1.jsonl` if present.
"""

from __future__ import annotations

import json

import pytest

from cragb.bench.assemble import (
    CragbEntry,
    assemble_entries,
    compute_datasheet_stats,
    render_datasheet,
    validate_entries,
    write_cragb_jsonl,
    write_datasheet,
    write_leakage_manifest,
)
from cragb.bench.best_photo import BestPhotoSelection, ImageQuestionRecord
from cragb.bench.reference_answers import ReferenceAnswer
from cragb.utils.io import resolve_path


def make_question(id_="q1", category="fit_sizing", is_negative=False, image_target=False) -> ImageQuestionRecord:
    return ImageQuestionRecord(id=id_, category=category, question=f"Question for {id_}?", is_negative=is_negative, image_target=image_target)


def make_answer(id_="q1", text="Runs small [10].", cited=("10",), abstention=False) -> ReferenceAnswer:
    return ReferenceAnswer(question_id=id_, answer=text, cited_doc_ids=cited, is_abstention=abstention)


class TestAssembleEntries:
    """Doc ids here are numeric strings (e.g. "10"), matching real usage:
    doc ids are always corpus row-index strings end to end, and
    `assemble_entries` sorts `relevant_ids` numerically on that
    assumption (see test_relevant_ids_sorted_numerically below)."""

    def test_merges_all_four_sources(self):
        questions = [make_question("q1", image_target=True)]
        relevant = {"q1": {"10", "20"}}
        answers = {"q1": make_answer("q1")}
        photos = {"q1": BestPhotoSelection("q1", "10", "clear photo")}

        entries = assemble_entries(questions, relevant, answers, photos)
        assert entries == [
            CragbEntry(
                id="q1",
                type="fit_sizing",
                is_negative=False,
                question="Question for q1?",
                relevant_ids=("10", "20"),
                reference_answer="Runs small [10].",
                cited_doc_ids=("10",),
                is_abstention=False,
                image_target=True,
                best_photo_id="10",
            )
        ]

    def test_non_target_question_gets_null_photo_without_needing_a_decision(self):
        questions = [make_question("q1", image_target=False)]
        entries = assemble_entries(questions, {"q1": {"10"}}, {"q1": make_answer("q1")}, {})
        assert entries[0].best_photo_id is None

    def test_missing_reference_answer_raises(self):
        questions = [make_question("q1")]
        with pytest.raises(ValueError, match="no reference answer found"):
            assemble_entries(questions, {}, {}, {})

    def test_image_target_without_photo_decision_raises(self):
        questions = [make_question("q1", image_target=True)]
        with pytest.raises(ValueError, match="no best-photo decision"):
            assemble_entries(questions, {"q1": {"10"}}, {"q1": make_answer("q1")}, {})

    def test_relevant_ids_sorted_numerically(self):
        questions = [make_question("q1")]
        relevant = {"q1": {"20", "3", "100"}}
        entries = assemble_entries(questions, relevant, {"q1": make_answer("q1", cited=())}, {})
        assert entries[0].relevant_ids == ("3", "20", "100")


def make_entry(id_="q1", type_="fit_sizing", is_negative=False, relevant_ids=("a",), cited=("a",), best_photo_id=None) -> CragbEntry:
    return CragbEntry(
        id=id_,
        type=type_,
        is_negative=is_negative,
        question="Q?",
        relevant_ids=relevant_ids,
        reference_answer="Answer.",
        cited_doc_ids=cited,
        is_abstention=not relevant_ids and not cited,
        image_target=best_photo_id is not None,
        best_photo_id=best_photo_id,
    )


class TestValidateEntries:
    def test_well_formed_entries_pass(self):
        entries = [make_entry("q1", relevant_ids=("a", "b"), cited=("a",))]
        report = validate_entries(entries)
        assert report.ok is True

    def test_negative_question_with_empty_relevant_ids_is_fine(self):
        entries = [make_entry("q1_neg", is_negative=True, relevant_ids=(), cited=())]
        report = validate_entries(entries)
        assert report.ok is True

    def test_non_negative_empty_relevant_ids_is_hard_failure(self):
        entries = [make_entry("q1", is_negative=False, relevant_ids=(), cited=())]
        report = validate_entries(entries)
        assert report.ok is False
        assert report.missing_relevant_ids == ("q1",)

    def test_citation_outside_relevant_ids_is_hard_failure(self):
        entries = [make_entry("q1", relevant_ids=("a",), cited=("a", "ghost"))]
        report = validate_entries(entries)
        assert report.ok is False
        assert report.citation_not_subset == ("q1",)

    def test_photo_outside_relevant_ids_is_hard_failure(self):
        entries = [make_entry("q1", relevant_ids=("a",), cited=("a",), best_photo_id="ghost")]
        report = validate_entries(entries)
        assert report.ok is False
        assert report.photo_not_relevant == ("q1",)

    def test_photo_inside_relevant_ids_passes(self):
        entries = [make_entry("q1", relevant_ids=("a", "b"), cited=("a",), best_photo_id="b")]
        report = validate_entries(entries)
        assert report.ok is True
        assert report.n_with_photo == 1

    def test_counts_are_correct(self):
        entries = [
            make_entry("q1", is_negative=False),
            make_entry("q2", is_negative=True, relevant_ids=(), cited=()),
        ]
        report = validate_entries(entries)
        assert report.total == 2
        assert report.n_negative == 1


class TestComputeDatasheetStats:
    def test_basic_counts(self):
        entries = [
            make_entry("q1", type_="fit_sizing", relevant_ids=("a", "b"), cited=("a",)),
            make_entry("q2", type_="fit_sizing", is_negative=True, relevant_ids=(), cited=()),
            make_entry("q3", type_="colour_appearance", relevant_ids=("c",), cited=("c",), best_photo_id="c"),
        ]
        stats = compute_datasheet_stats(entries)
        assert stats["total"] == 3
        assert stats["by_type"] == {"colour_appearance": 1, "fit_sizing": 2}
        assert stats["negatives_by_type"] == {"fit_sizing": 1}
        assert stats["n_negative"] == 1
        assert stats["n_abstention"] == 1  # q2's empty relevant_ids/cited makes it an abstention by make_entry's helper
        assert stats["n_image_target"] == 1
        assert stats["n_with_photo"] == 1
        assert stats["image_coverage_pct"] == 100.0

    def test_relevant_id_range(self):
        entries = [make_entry("q1", relevant_ids=("a", "b", "c")), make_entry("q2", relevant_ids=("a",))]
        stats = compute_datasheet_stats(entries)
        assert stats["min_relevant_ids"] == 1
        assert stats["max_relevant_ids"] == 3

    def test_empty_entries_do_not_divide_by_zero(self):
        stats = compute_datasheet_stats([])
        assert stats["total"] == 0
        assert stats["image_coverage_pct"] == 0.0
        assert stats["mean_relevant_ids"] == 0.0


class TestRenderDatasheet:
    def test_contains_required_sections(self):
        entries = [make_entry("q1")]
        stats = compute_datasheet_stats(entries)
        md = render_datasheet(entries, stats, corpus_manifest={"seed": 42, "row_count": 200000, "sha256": "ab" * 32})
        for heading in ("Motivation", "Composition", "Collection process", "Known limitations", "Held-out status", "Provenance"):
            assert f"## {heading}" in md

    def test_handles_missing_corpus_manifest(self):
        entries = [make_entry("q1")]
        stats = compute_datasheet_stats(entries)
        md = render_datasheet(entries, stats, corpus_manifest=None)
        assert "not found" in md.lower()


class TestWriteFunctions:
    def test_write_cragb_jsonl_schema(self, tmp_path):
        entries = [make_entry("q1", relevant_ids=("a",), cited=("a",))]
        out_path = write_cragb_jsonl(entries, tmp_path / "cragb_v1.jsonl")
        row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        assert set(row) == {
            "id", "type", "is_negative", "question", "relevant_ids", "reference_answer",
            "cited_doc_ids", "is_abstention", "image_target", "best_photo_id",
        }

    def test_write_leakage_manifest_has_one_hash_per_entry(self, tmp_path):
        entries = [make_entry("q1"), make_entry("q2")]
        out_path = write_leakage_manifest(entries, tmp_path / "manifest.json")
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(manifest["question_hashes"]) == {"q1", "q2"}
        assert manifest["n_questions"] == 2

    def test_write_datasheet_creates_file(self, tmp_path):
        entries = [make_entry("q1")]
        out_path = write_datasheet(entries, tmp_path / "datasheet.md", corpus_manifest=None)
        assert out_path.is_file()


@pytest.mark.skipif(
    not resolve_path("benchmark/cragb_v1.jsonl").is_file(),
    reason="benchmark/cragb_v1.jsonl not present locally",
)
class TestRealCragbV1SelfConsistency:
    def test_schema_and_invariants(self):
        entries = []
        with resolve_path("benchmark/cragb_v1.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                entries.append(
                    CragbEntry(
                        id=obj["id"],
                        type=obj["type"],
                        is_negative=obj["is_negative"],
                        question=obj["question"],
                        relevant_ids=tuple(obj["relevant_ids"]),
                        reference_answer=obj["reference_answer"],
                        cited_doc_ids=tuple(obj["cited_doc_ids"]),
                        is_abstention=obj["is_abstention"],
                        image_target=obj["image_target"],
                        best_photo_id=obj["best_photo_id"],
                    )
                )
        report = validate_entries(entries)
        assert report.ok, report.messages
        assert report.total == 60
