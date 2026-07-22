"""Unit tests for cragb.bench.assemble's leakage guard (T2.10; PLAN.md §1.4 risk F).

This is the guard E8 (fine-tune data prep, second half of the project)
will call against every synthetic training-example prompt before
accepting it, so a CRAGB *evaluation* question can never end up as
*training* data. Covers: text normalization (so trivial
case/whitespace differences don't defeat the guard), hash determinism,
manifest building, and `check_no_leakage`'s detection — both that it
catches a real leak and that it doesn't false-positive on genuinely
different text. Also self-consistency-checks the real
`cragb_v1_leakage_manifest.json` against the real `cragb_v1.jsonl` if
both are present.
"""

from __future__ import annotations

import json

import pytest

from cragb.bench.assemble import (
    CragbEntry,
    build_leakage_manifest,
    check_no_leakage,
    compute_question_hash,
    normalize_question_text,
)
from cragb.utils.io import resolve_path


def make_entry(id_: str, question: str) -> CragbEntry:
    return CragbEntry(
        id=id_,
        type="fit_sizing",
        is_negative=False,
        question=question,
        relevant_ids=("a",),
        reference_answer="Answer [a].",
        cited_doc_ids=("a",),
        is_abstention=False,
        image_target=False,
        best_photo_id=None,
    )


class TestNormalizeQuestionText:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_question_text("  Does  This\tRun   Small?  ") == "does this run small?"

    def test_case_and_whitespace_variants_normalize_identically(self):
        a = normalize_question_text("Does this run small?")
        b = normalize_question_text("DOES   THIS RUN SMALL?")
        assert a == b


class TestComputeQuestionHash:
    def test_deterministic(self):
        assert compute_question_hash("Does this run small?") == compute_question_hash("Does this run small?")

    def test_normalization_insensitive(self):
        assert compute_question_hash("Does this run small?") == compute_question_hash("  DOES this run   small? ")

    def test_different_text_different_hash(self):
        assert compute_question_hash("Does this run small?") != compute_question_hash("Is the colour as pictured?")


class TestBuildLeakageManifest:
    def test_one_hash_per_entry(self):
        entries = [make_entry("q1", "Does this run small?"), make_entry("q2", "Is the colour as pictured?")]
        manifest = build_leakage_manifest(entries)
        assert set(manifest) == {"q1", "q2"}
        assert manifest["q1"] == compute_question_hash("Does this run small?")

    def test_identical_question_text_across_ids_still_both_recorded(self):
        # Pathological but shouldn't crash: two different question ids that
        # happen to share text still both get an entry (keyed by id, not text).
        entries = [make_entry("q1", "Same text?"), make_entry("q2", "Same text?")]
        manifest = build_leakage_manifest(entries)
        assert manifest["q1"] == manifest["q2"]
        assert len(manifest) == 2


class TestCheckNoLeakage:
    def test_no_overlap_passes(self):
        hashes = build_leakage_manifest([make_entry("q1", "Does this run small?")])
        report = check_no_leakage(hashes, ["Completely unrelated training prompt."])
        assert report.ok is True
        assert report.leaked_question_ids == ()

    def test_exact_verbatim_leak_is_detected(self):
        hashes = build_leakage_manifest([make_entry("q1", "Does this run small?")])
        report = check_no_leakage(hashes, ["Does this run small?"])
        assert report.ok is False
        assert report.leaked_question_ids == ("q1",)

    def test_leak_survives_case_and_whitespace_differences(self):
        hashes = build_leakage_manifest([make_entry("q1", "Does this run small?")])
        report = check_no_leakage(hashes, ["  DOES   this run small?  "])
        assert report.ok is False
        assert report.leaked_question_ids == ("q1",)

    def test_paraphrased_text_is_not_flagged(self):
        # By design: this guard catches verbatim/near-verbatim copies, not
        # semantic paraphrase detection (see module docstring).
        hashes = build_leakage_manifest([make_entry("q1", "Does this run small?")])
        report = check_no_leakage(hashes, ["Is this item true to size or does it run small?"])
        assert report.ok is True

    def test_multiple_candidates_only_matching_one_is_reported(self):
        hashes = build_leakage_manifest(
            [make_entry("q1", "Does this run small?"), make_entry("q2", "Is the colour as pictured?")]
        )
        report = check_no_leakage(hashes, ["Unrelated prompt.", "Is the colour as pictured?"])
        assert report.ok is False
        assert report.leaked_question_ids == ("q2",)
        assert report.n_checked == 2

    def test_empty_candidate_list_passes(self):
        hashes = build_leakage_manifest([make_entry("q1", "Does this run small?")])
        report = check_no_leakage(hashes, [])
        assert report.ok is True
        assert report.n_checked == 0


@pytest.mark.skipif(
    not (resolve_path("benchmark/cragb_v1.jsonl").is_file() and resolve_path("benchmark/cragb_v1_leakage_manifest.json").is_file()),
    reason="real cragb_v1.jsonl / leakage manifest not present locally",
)
class TestRealLeakageManifestSelfConsistency:
    def test_manifest_hashes_match_recomputed_hashes(self):
        manifest = json.loads(resolve_path("benchmark/cragb_v1_leakage_manifest.json").read_text(encoding="utf-8"))
        with resolve_path("benchmark/cragb_v1.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                assert manifest["question_hashes"][obj["id"]] == compute_question_hash(obj["question"])

    def test_the_benchmark_leaks_against_a_copy_of_itself(self):
        """Sanity check that the guard actually works: feeding it CRAGB's
        own question text back in must be detected as 100% leakage."""
        manifest = json.loads(resolve_path("benchmark/cragb_v1_leakage_manifest.json").read_text(encoding="utf-8"))
        questions = []
        with resolve_path("benchmark/cragb_v1.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line)["question"])

        report = check_no_leakage(manifest["question_hashes"], questions)
        assert report.ok is False
        assert len(report.leaked_question_ids) == len(questions)
