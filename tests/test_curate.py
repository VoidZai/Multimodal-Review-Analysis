"""Unit tests for cragb.bench.curate.

Covers: automated quality filtering, human-decision loading and its
completeness/consistency enforcement (every quality-passed draft must be
decided, every decision must reference a real draft), negative-question
authoring, corpus-driven image-target flagging, strict per-category
validation against a `TaxonomySpec`, and the full `curate()` pipeline
end-to-end on synthetic data. Also runs the real
`benchmark/curation_decisions_v1.yaml` against `configs/taxonomy.yaml`
for internal self-consistency (independent of the corpus/draft files,
which may not exist in a fresh checkout).
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from cragb.bench.curate import (
    CurationDecisions,
    DraftRecord,
    NegativeQuestion,
    apply_quality_filters,
    build_accepted_questions,
    build_negative_questions,
    curate,
    estimate_image_coverage,
    flag_image_targets,
    load_decisions,
    load_drafts,
    validate_curation,
    write_questions_jsonl,
)
from cragb.bench.taxonomy import TaxonomyCategory, TaxonomySpec

FIT = TaxonomyCategory("fit_sizing", "Fit desc.", target_count=2, negative_count=1)
COLOUR = TaxonomyCategory("colour_appearance", "Colour desc.", target_count=1, negative_count=1)


def make_spec(categories=(FIT, COLOUR), **overrides) -> TaxonomySpec:
    defaults = dict(
        categories=categories,
        expected_total=sum(c.target_count for c in categories),
        total_tolerance=1,
        min_coverage_pct=0.01,
        min_coverage_pct_hard_floor=0.0,
    )
    defaults.update(overrides)
    return TaxonomySpec(**defaults)


class TestLoadDrafts:
    def test_round_trips_jsonl(self, tmp_path):
        path = tmp_path / "drafts.jsonl"
        path.write_text(
            '{"id": "fit_sizing_000", "category": "fit_sizing", "question": "Does this run small?", "rationale": "sizing"}\n',
            encoding="utf-8",
        )
        drafts = load_drafts(path)
        assert len(drafts) == 1
        assert drafts[0] == DraftRecord("fit_sizing_000", "fit_sizing", "Does this run small?", "sizing")

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "drafts.jsonl"
        path.write_text(
            '{"id": "a", "category": "fit_sizing", "question": "Q1?", "rationale": ""}\n\n'
            '{"id": "b", "category": "fit_sizing", "question": "Q2?", "rationale": ""}\n',
            encoding="utf-8",
        )
        assert len(load_drafts(path)) == 2


class TestApplyQualityFilters:
    def test_too_short_is_rejected(self):
        drafts = [DraftRecord("a", "fit_sizing", "Small?", "")]
        passed, rejected = apply_quality_filters(drafts, min_words=3, max_words=30, require_question_mark=True)
        assert passed == []
        assert rejected[0].id == "a"
        assert "too short" in rejected[0].reason

    def test_too_long_is_rejected(self):
        long_question = " ".join(["word"] * 40) + "?"
        drafts = [DraftRecord("a", "fit_sizing", long_question, "")]
        passed, rejected = apply_quality_filters(drafts, min_words=3, max_words=30, require_question_mark=True)
        assert passed == []
        assert "too long" in rejected[0].reason

    def test_missing_question_mark_is_rejected(self):
        drafts = [DraftRecord("a", "fit_sizing", "Does this run small", "")]
        passed, rejected = apply_quality_filters(drafts, min_words=1, max_words=30, require_question_mark=True)
        assert passed == []
        assert "does not end with '?'" in rejected[0].reason

    def test_well_formed_draft_passes(self):
        drafts = [DraftRecord("a", "fit_sizing", "Does this run small on most buyers?", "")]
        passed, rejected = apply_quality_filters(drafts, min_words=3, max_words=30, require_question_mark=True)
        assert len(passed) == 1
        assert rejected == []


class TestLoadDecisions:
    def _write(self, tmp_path, content: dict) -> "Path":
        path = tmp_path / "decisions.yaml"
        path.write_text(yaml.dump(content), encoding="utf-8")
        return path

    def test_parses_accepted_rejected_and_negatives(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "accepted": [{"id": "a", "edited_question": "Edited?"}, {"id": "b"}],
                "rejected": [{"id": "c", "reason": "bad"}],
                "negative_questions": [
                    {"category": "fit_sizing", "question": "Exact spec?", "reason": "too precise"}
                ],
            },
        )
        decisions = load_decisions(path)
        assert decisions.accepted == {"a": "Edited?", "b": None}
        assert decisions.rejected == {"c": "bad"}
        assert decisions.negatives == (NegativeQuestion("fit_sizing", "Exact spec?", "too precise"),)

    def test_id_in_both_accepted_and_rejected_raises(self, tmp_path):
        path = self._write(
            tmp_path,
            {"accepted": [{"id": "a"}], "rejected": [{"id": "a", "reason": "bad"}]},
        )
        with pytest.raises(ValueError, match="both accepted and rejected"):
            load_decisions(path)

    def test_empty_file_yields_empty_decisions(self, tmp_path):
        path = tmp_path / "decisions.yaml"
        path.write_text("", encoding="utf-8")
        decisions = load_decisions(path)
        assert decisions.accepted == {}
        assert decisions.rejected == {}
        assert decisions.negatives == ()


class TestBuildAcceptedQuestions:
    def test_uses_edited_text_when_present(self):
        passed = [DraftRecord("a", "fit_sizing", "Original?", "")]
        decisions = CurationDecisions(accepted={"a": "Edited?"}, rejected={}, negatives=())
        questions = build_accepted_questions(passed, decisions)
        assert questions[0].question == "Edited?"
        assert questions[0].is_negative is False
        assert questions[0].image_target is False

    def test_keeps_original_text_when_no_edit(self):
        passed = [DraftRecord("a", "fit_sizing", "Original?", "")]
        decisions = CurationDecisions(accepted={"a": None}, rejected={}, negatives=())
        questions = build_accepted_questions(passed, decisions)
        assert questions[0].question == "Original?"

    def test_missing_decision_raises(self):
        passed = [DraftRecord("a", "fit_sizing", "Q?", ""), DraftRecord("b", "fit_sizing", "Q2?", "")]
        decisions = CurationDecisions(accepted={"a": None}, rejected={}, negatives=())
        with pytest.raises(ValueError, match="no accept/reject decision"):
            build_accepted_questions(passed, decisions)

    def test_decision_referencing_unknown_draft_raises(self):
        passed = [DraftRecord("a", "fit_sizing", "Q?", "")]
        decisions = CurationDecisions(accepted={"a": None, "ghost": None}, rejected={}, negatives=())
        with pytest.raises(ValueError, match="unknown or filtered-out"):
            build_accepted_questions(passed, decisions)

    def test_rejected_draft_excluded_from_output(self):
        passed = [DraftRecord("a", "fit_sizing", "Q?", ""), DraftRecord("b", "fit_sizing", "Q2?", "")]
        decisions = CurationDecisions(accepted={"a": None}, rejected={"b": "weak"}, negatives=())
        questions = build_accepted_questions(passed, decisions)
        assert [q.id for q in questions] == ["a"]


class TestBuildNegativeQuestions:
    def test_assigns_per_category_sequential_ids(self):
        negatives = (
            NegativeQuestion("fit_sizing", "Exact spec 1?", "r"),
            NegativeQuestion("fit_sizing", "Exact spec 2?", "r"),
            NegativeQuestion("colour_appearance", "Dye lot?", "r"),
        )
        questions = build_negative_questions(negatives)
        ids = [q.id for q in questions]
        assert ids == ["fit_sizing_neg_000", "fit_sizing_neg_001", "colour_appearance_neg_000"]
        assert all(q.is_negative for q in questions)


class TestEstimateImageCoverage:
    def test_computes_percentage_among_matched_reviews(self):
        df = pd.DataFrame(
            {
                "text": ["runs small", "runs small", "unrelated text entirely"],
                "has_image": [True, False, True],
            }
        )
        pct = estimate_image_coverage(df, FIT)
        # 2 rows match "small"/"runs small" (fit_sizing keywords), 1 of them has_image=True
        assert pct == pytest.approx(50.0)

    def test_no_matches_returns_zero(self):
        df = pd.DataFrame({"text": ["nothing relevant here"], "has_image": [True]})
        assert estimate_image_coverage(df, FIT) == 0.0


class TestFlagImageTargets:
    def test_flags_only_non_negative_questions_above_threshold(self):
        from cragb.bench.curate import CuratedQuestion

        questions = [
            CuratedQuestion("fit_sizing_000", "fit_sizing", "Q?", is_negative=False, image_target=False),
            CuratedQuestion("fit_sizing_neg_000", "fit_sizing", "Neg?", is_negative=True, image_target=False),
        ]
        df = pd.DataFrame({"text": ["runs small"] * 10, "has_image": [True] * 10})
        spec = make_spec()
        flagged, coverage = flag_image_targets(questions, df, spec, min_image_coverage_pct=5.0)

        by_id = {q.id: q for q in flagged}
        assert by_id["fit_sizing_000"].image_target is True
        assert by_id["fit_sizing_neg_000"].image_target is False  # negatives never flagged
        assert coverage["fit_sizing"] == 100.0

    def test_below_threshold_not_flagged(self):
        from cragb.bench.curate import CuratedQuestion

        questions = [CuratedQuestion("fit_sizing_000", "fit_sizing", "Q?", is_negative=False, image_target=False)]
        df = pd.DataFrame({"text": ["runs small"] * 10, "has_image": [False] * 10})
        spec = make_spec()
        flagged, _ = flag_image_targets(questions, df, spec, min_image_coverage_pct=5.0)
        assert flagged[0].image_target is False


class TestValidateCuration:
    def _make_questions(self, fit_count, fit_neg, colour_count, colour_neg):
        from cragb.bench.curate import CuratedQuestion

        questions = []
        for i in range(fit_count - fit_neg):
            questions.append(CuratedQuestion(f"fit_{i}", "fit_sizing", "Q?", False, False))
        for i in range(fit_neg):
            questions.append(CuratedQuestion(f"fit_neg_{i}", "fit_sizing", "Q?", True, False))
        for i in range(colour_count - colour_neg):
            questions.append(CuratedQuestion(f"colour_{i}", "colour_appearance", "Q?", False, False))
        for i in range(colour_neg):
            questions.append(CuratedQuestion(f"colour_neg_{i}", "colour_appearance", "Q?", True, False))
        return questions

    def test_exact_match_passes(self):
        spec = make_spec()  # fit target=2/neg=1, colour target=1/neg=1
        questions = self._make_questions(2, 1, 1, 1)
        report = validate_curation(questions, spec)
        assert report.ok
        assert report.total == 3

    def test_wrong_category_count_fails(self):
        spec = make_spec()
        questions = self._make_questions(3, 1, 1, 1)  # fit has 3, expected 2
        report = validate_curation(questions, spec)
        assert not report.ok
        assert any("fit_sizing" in m and "expected 2" in m for m in report.messages)

    def test_wrong_negative_count_fails(self):
        spec = make_spec()
        questions = self._make_questions(2, 0, 1, 1)  # fit has 0 negatives, expected 1
        report = validate_curation(questions, spec)
        assert not report.ok
        assert any("expected 1 negative" in m for m in report.messages)

    def test_unknown_category_raises(self):
        from cragb.bench.curate import CuratedQuestion

        spec = make_spec()
        with pytest.raises(ValueError, match="unknown category"):
            validate_curation([CuratedQuestion("x", "not_a_category", "Q?", False, False)], spec)


class TestCurateEndToEnd:
    def test_full_pipeline_on_synthetic_data(self):
        drafts = [
            DraftRecord("fit_sizing_000", "fit_sizing", "Does this run small on most buyers?", "sizing"),
            DraftRecord("fit_sizing_001", "fit_sizing", "Is the fit true to size overall?", "sizing"),
            DraftRecord("fit_sizing_002", "fit_sizing", "too", "junk"),  # fails min_words
            DraftRecord("colour_appearance_000", "colour_appearance", "Is the colour as pictured?", "colour"),
        ]
        decisions = CurationDecisions(
            accepted={"fit_sizing_000": None},
            rejected={"fit_sizing_001": "redundant", "colour_appearance_000": "weak phrasing"},
            negatives=(
                NegativeQuestion("fit_sizing", "Exact chest measurement in inches?", "too precise"),
                NegativeQuestion("colour_appearance", "Dye lot number?", "too precise"),
            ),
        )
        spec = make_spec()  # fit target=2/neg=1, colour target=1/neg=1
        corpus = pd.DataFrame(
            {
                "text": ["runs small", "true to size", "as pictured colour"],
                "has_image": [True, True, False],
            }
        )

        questions, report, coverage = curate(
            drafts=drafts,
            spec=spec,
            decisions=decisions,
            corpus=corpus,
            min_words=3,
            max_words=30,
            require_question_mark=True,
            min_image_coverage_pct=5.0,
        )

        assert report.ok, report.messages
        assert report.total == 3
        assert {q.id for q in questions} == {
            "fit_sizing_000",
            "fit_sizing_neg_000",
            "colour_appearance_neg_000",
        }
        accepted = next(q for q in questions if q.id == "fit_sizing_000")
        assert accepted.question == "Does this run small on most buyers?"
        negative = next(q for q in questions if q.id == "fit_sizing_neg_000")
        assert negative.is_negative is True
        assert negative.image_target is False


class TestWriteQuestionsJsonl:
    def test_writes_one_json_object_per_line(self, tmp_path):
        from cragb.bench.curate import CuratedQuestion

        questions = [
            CuratedQuestion("a", "fit_sizing", "Q1?", False, True),
            CuratedQuestion("b", "fit_sizing", "Q2?", True, False),
        ]
        out_path = tmp_path / "out.jsonl"
        write_questions_jsonl(questions, out_path)
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        import json

        first = json.loads(lines[0])
        assert first == {"id": "a", "category": "fit_sizing", "question": "Q1?", "is_negative": False, "image_target": True}


class TestRealDecisionsFileSelfConsistency:
    """Sanity-checks benchmark/curation_decisions_v1.yaml against configs/taxonomy.yaml.

    Uses only those two committed config/benchmark files (no corpus or
    draft file dependency), so this test is meaningful even before
    `data/processed/corpus_v1.parquet` exists in a fresh checkout.
    """

    def test_negative_quota_per_category_matches_taxonomy(self):
        from cragb.bench.taxonomy import load_taxonomy

        spec = load_taxonomy("configs/taxonomy.yaml")
        decisions = load_decisions("benchmark/curation_decisions_v1.yaml")

        negative_counts: dict[str, int] = {}
        for neg in decisions.negatives:
            negative_counts[neg.category] = negative_counts.get(neg.category, 0) + 1

        for category in spec.categories:
            assert negative_counts.get(category.name, 0) == category.negative_count, (
                f"{category.name}: decisions file has "
                f"{negative_counts.get(category.name, 0)} authored negatives, "
                f"taxonomy expects {category.negative_count}"
            )

    def test_accepted_count_plus_negatives_matches_target_per_category(self):
        from collections import Counter

        from cragb.bench.taxonomy import load_taxonomy

        spec = load_taxonomy("configs/taxonomy.yaml")
        decisions = load_decisions("benchmark/curation_decisions_v1.yaml")
        drafts = {d.id: d for d in load_drafts("benchmark/cragb_draft.jsonl")}

        accepted_by_category = Counter(drafts[i].category for i in decisions.accepted)
        negative_by_category = Counter(n.category for n in decisions.negatives)

        for category in spec.categories:
            total = accepted_by_category[category.name] + negative_by_category[category.name]
            assert total == category.target_count, (
                f"{category.name}: {accepted_by_category[category.name]} accepted + "
                f"{negative_by_category[category.name]} negatives = {total}, "
                f"expected {category.target_count}"
            )
