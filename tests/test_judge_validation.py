"""Unit tests for cragb.eval.judge_validation (T4b.6; M4b.md T4b.6).

`build_sample`/`export_worksheet`/`score_worksheet` need real transcript files on disk
(`load_arm_transcripts` reads from `cragb.eval.run_answer_generation.ARM_DEFAULT_OUT`) --
tests that exercise them monkeypatch `judge_validation.ARM_DEFAULT_OUT` to point at small
`tmp_path` fixtures instead of the real project data, so nothing here depends on
`results/tables/answer_gen_*.jsonl` actually existing. `parse_worksheet` and
`compute_agreement` need no I/O at all and are tested directly against hand-built text/
DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.bench.reference_answers import make_reference_answer
from cragb.eval.judge_validation import (
    build_paired_scores,
    build_sample,
    compute_agreement,
    export_worksheet,
    parse_worksheet,
    plot_judge_human_agreement,
    render_worksheet,
    score_worksheet,
)
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.closed_book_qa import write_transcripts_jsonl as write_closed_book_jsonl
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript
from cragb.generate.grounded_qa import write_transcripts_jsonl as write_grounded_qa_jsonl


def make_closed_book_transcript(qid: str, question: str, answer_text: str) -> ClosedBookTranscript:
    return ClosedBookTranscript(
        question_id=qid, question=question, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), abstained=False,
    )


def make_grounded_transcript(qid: str, question: str, answer_text: str, context_text: str) -> GroundedQATranscript:
    context = ContextBlock(text=context_text, doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question=question, context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), cited_photo_ids=(), abstained=False,
    )


@pytest.fixture
def two_arm_fixture(tmp_path, monkeypatch):
    """4 closed_book + 4 rag_small transcripts on disk, a matching judge_scores table,
    and references -- everything build_sample/export_worksheet/score_worksheet need,
    with ARM_DEFAULT_OUT monkeypatched to point at these tmp files.
    """
    closed_book_transcripts = [
        make_closed_book_transcript(f"cb_{i}", f"Closed-book question {i}?", f"Closed-book answer {i}.")
        for i in range(4)
    ]
    rag_small_transcripts = [
        make_grounded_transcript(f"rs_{i}", f"RAG question {i}?", f"RAG answer {i}.", f"[10{i}] context {i}.")
        for i in range(4)
    ]

    cb_path = write_closed_book_jsonl(closed_book_transcripts, tmp_path / "cb.jsonl")
    rs_path = write_grounded_qa_jsonl(rag_small_transcripts, tmp_path / "rs.jsonl")

    monkeypatch.setattr(
        "cragb.eval.judge_validation.ARM_DEFAULT_OUT",
        {"closed_book": str(cb_path), "rag_small": str(rs_path), "rag_large": str(rs_path)},
    )

    # Deliberately varied scores per row (not a single repeated value for any
    # criterion) -- a criterion that's constant across the whole fixture makes Cohen's
    # kappa mathematically undefined (both raters have zero variance), which is a real
    # edge case of the metric itself, not something these tests are meant to probe.
    rows = []
    for i in range(4):
        rows.append(
            {"arm": "closed_book", "question_id": f"cb_{i}", "correctness": 1 + (i % 3),
             "faithfulness": 3 + (i % 3), "completeness": 1 + (i % 2), "conciseness": 4 + (i % 2), "rationale": "r"}
        )
        rows.append(
            {"arm": "rag_small", "question_id": f"rs_{i}", "correctness": 2 + (i % 4),
             "faithfulness": 2 + (i % 4), "completeness": 3 + (i % 3), "conciseness": 3 + (i % 3), "rationale": "r"}
        )
    judge_scores = pd.DataFrame(rows)

    # Reference text deliberately does *not* embed the question id as a literal
    # substring -- tests check the worksheet never leaks question_id, and a fixture
    # where the reference text happens to contain it would make that check meaningless.
    references = {}
    for i in range(4):
        references[f"cb_{i}"] = make_reference_answer(f"cb_{i}", f"Trusted answer number {i} for the closed-book set.")
        references[f"rs_{i}"] = make_reference_answer(f"rs_{i}", f"Trusted answer number {i} for the RAG set.")

    return judge_scores, references


# --------------------------------------------------------------------------
# build_sample
# --------------------------------------------------------------------------


class TestBuildSample:
    def test_samples_evenly_across_arms(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        arms = [row.arm for row in rows]
        assert arms.count("closed_book") == 2
        assert arms.count("rag_small") == 2

    def test_same_seed_is_fully_reproducible(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        first = build_sample(judge_scores, references, sample_size=4, seed=7)
        second = build_sample(judge_scores, references, sample_size=4, seed=7)
        assert [(r.row_id, r.arm, r.question_id) for r in first] == [
            (r.row_id, r.arm, r.question_id) for r in second
        ]

    def test_row_ids_are_sequential_and_unique(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        assert [row.row_id for row in rows] == ["R01", "R02", "R03", "R04"]

    def test_closed_book_row_has_none_context(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        cb_row = next(row for row in rows if row.arm == "closed_book")
        assert cb_row.context_text is None

    def test_rag_row_carries_its_real_context(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        rag_row = next(row for row in rows if row.arm == "rag_small")
        assert rag_row.context_text is not None and "context" in rag_row.context_text

    def test_requesting_more_than_available_in_an_arm_raises(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        with pytest.raises(ValueError, match="only 4 row"):
            build_sample(judge_scores, references, sample_size=20, seed=1)

    def test_empty_judge_scores_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_sample(pd.DataFrame(columns=["arm", "question_id"]), {}, sample_size=4, seed=1)


# --------------------------------------------------------------------------
# render_worksheet / export_worksheet
# --------------------------------------------------------------------------


class TestRenderWorksheet:
    def test_never_shows_arm_question_id_or_judge_scores(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        text = render_worksheet(rows)

        for row in rows:
            assert row.question_id not in text
            assert row.arm not in text
        # The judge's own numeric scores must not leak into the worksheet either.
        assert "correctness: 1" not in text
        assert "correctness: 4" not in text

    def test_contains_blank_score_lines_for_every_row(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        text = render_worksheet(rows)
        for criterion in ("correctness", "faithfulness", "completeness", "conciseness"):
            assert text.count(f"- {criterion}: ") == len(rows)

    def test_contains_question_and_answer_content(self, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        rows = build_sample(judge_scores, references, sample_size=4, seed=1)
        text = render_worksheet(rows)
        for row in rows:
            assert row.question in text
            assert row.candidate_answer in text
            assert row.reference_answer in text


class TestExportWorksheet:
    def test_writes_worksheet_file_with_expected_row_count(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        out_path, rows = export_worksheet(judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md")

        assert out_path.is_file()
        assert len(rows) == 4
        assert out_path.read_text(encoding="utf-8").count("## R") == 4


def _write_references_jsonl(references, path):
    import json

    with open(path, "w", encoding="utf-8") as f:
        for ref in references.values():
            f.write(json.dumps({
                "question_id": ref.question_id, "answer": ref.answer,
                "cited_doc_ids": list(ref.cited_doc_ids), "is_abstention": ref.is_abstention,
            }))
            f.write("\n")


# --------------------------------------------------------------------------
# parse_worksheet
# --------------------------------------------------------------------------


FILLED_BLOCK = """## R01

**Question:** Do these run small?

**Context shown to the answerer:** No review context was available when this answer was written.

**Candidate answer:** Not enough information.

**Reference answer:** Runs small.

**Your scores (integer 1-5 each):**
- correctness: 3
- faithfulness: 5
- completeness: 2
- conciseness: 4
"""


class TestParseWorksheet:
    def test_parses_a_single_filled_row(self):
        result = parse_worksheet(FILLED_BLOCK)
        assert result == {"R01": {"correctness": 3, "faithfulness": 5, "completeness": 2, "conciseness": 4}}

    def test_parses_multiple_rows(self):
        text = FILLED_BLOCK + "\n" + FILLED_BLOCK.replace("R01", "R02").replace("correctness: 3", "correctness: 1")
        result = parse_worksheet(text)
        assert set(result) == {"R01", "R02"}
        assert result["R02"]["correctness"] == 1

    def test_blank_score_raises(self):
        text = FILLED_BLOCK.replace("correctness: 3", "correctness: ")
        with pytest.raises(ValueError, match="is blank"):
            parse_worksheet(text)

    def test_missing_criterion_line_raises(self):
        text = FILLED_BLOCK.replace("- correctness: 3\n", "")
        with pytest.raises(ValueError, match="no 'correctness' line"):
            parse_worksheet(text)

    def test_non_integer_score_raises(self):
        text = FILLED_BLOCK.replace("correctness: 3", "correctness: three")
        with pytest.raises(ValueError, match="not an integer"):
            parse_worksheet(text)

    def test_out_of_range_score_raises(self):
        text = FILLED_BLOCK.replace("correctness: 3", "correctness: 7")
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_worksheet(text)

    def test_zero_score_raises(self):
        text = FILLED_BLOCK.replace("correctness: 3", "correctness: 0")
        with pytest.raises(ValueError, match="not an integer in \\[1, 5\\]"):
            parse_worksheet(text)


# --------------------------------------------------------------------------
# compute_agreement
# --------------------------------------------------------------------------


class TestComputeAgreement:
    def test_identical_judge_and_human_scores_give_kappa_near_one(self):
        rows = []
        for i in range(10):
            score = (i % 5) + 1
            rows.append(
                {
                    "judge_correctness": score, "human_correctness": score,
                    "judge_faithfulness": score, "human_faithfulness": score,
                    "judge_completeness": score, "human_completeness": score,
                    "judge_conciseness": score, "human_conciseness": score,
                }
            )
        result = compute_agreement(pd.DataFrame(rows))
        assert (result["cohens_kappa"] >= 0.99).all()
        assert (result["pct_within_one_point"] == 1.0).all()

    def test_uncorrelated_scores_give_low_kappa(self):
        # Judge always says 5; human varies across the full range -- no agreement
        # beyond what quadratic weighting gives partial credit for, and definitely
        # not the near-1.0 the identical-scores case gets.
        judge_vals = [5] * 10
        human_vals = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        rows = [
            {
                "judge_correctness": j, "human_correctness": h,
                "judge_faithfulness": j, "human_faithfulness": h,
                "judge_completeness": j, "human_completeness": h,
                "judge_conciseness": j, "human_conciseness": h,
            }
            for j, h in zip(judge_vals, human_vals)
        ]
        result = compute_agreement(pd.DataFrame(rows))
        assert (result["cohens_kappa"] < 0.3).all()

    def test_empty_paired_table_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_agreement(pd.DataFrame(columns=["judge_correctness", "human_correctness"]))


# --------------------------------------------------------------------------
# score_worksheet
# --------------------------------------------------------------------------


class TestScoreWorksheet:
    def test_scores_a_correctly_filled_worksheet(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        worksheet_path, rows = export_worksheet(
            judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md"
        )

        # Fill in the worksheet with scores identical to the judge's own, so agreement
        # should come out perfect -- a clean end-to-end check of the whole pipeline.
        text = worksheet_path.read_text(encoding="utf-8")
        for row in rows:
            for criterion, value in row.judge_scores.items():
                text = text.replace(f"- {criterion}: \n", f"- {criterion}: {value}\n", 1)
        worksheet_path.write_text(text, encoding="utf-8")

        result = score_worksheet(worksheet_path, judge_scores_path, references_path, sample_size=4, seed=1)

        assert set(result["criterion"]) == {"correctness", "faithfulness", "completeness", "conciseness"}
        assert (result["cohens_kappa"] >= 0.99).all()
        assert (result["n"] == 4).all()

    def test_worksheet_not_matching_the_reseeded_sample_raises(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        export_worksheet(judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md")

        # A stale worksheet made under a different seed -- row ids won't line up.
        stale_worksheet = tmp_path / "stale.md"
        stale_worksheet.write_text(FILLED_BLOCK, encoding="utf-8")

        with pytest.raises(ValueError, match="do not match the re-derived sample"):
            score_worksheet(stale_worksheet, judge_scores_path, references_path, sample_size=4, seed=99)

    def test_incomplete_worksheet_raises_before_computing_anything(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        worksheet_path, _rows = export_worksheet(
            judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md"
        )
        # Left blank on purpose -- exactly the "refuses to run" case M4b.md's
        # validation checks require.
        with pytest.raises(ValueError, match="is blank"):
            score_worksheet(worksheet_path, judge_scores_path, references_path, sample_size=4, seed=1)


# --------------------------------------------------------------------------
# build_paired_scores (T4b.8: split out of score_worksheet so the agreement
# figure can use the raw per-row pairs, not just compute_agreement's summary)
# --------------------------------------------------------------------------


class TestBuildPairedScores:
    def test_returns_one_row_per_sample_with_judge_and_human_columns(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        worksheet_path, rows = export_worksheet(
            judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md"
        )
        text = worksheet_path.read_text(encoding="utf-8")
        for row in rows:
            for criterion, value in row.judge_scores.items():
                text = text.replace(f"- {criterion}: \n", f"- {criterion}: {value}\n", 1)
        worksheet_path.write_text(text, encoding="utf-8")

        paired = build_paired_scores(worksheet_path, judge_scores_path, references_path, sample_size=4, seed=1)

        assert len(paired) == 4
        expected_cols = {"row_id", "arm", "question_id"} | {
            f"{who}_{c}" for who in ("judge", "human") for c in ("correctness", "faithfulness", "completeness", "conciseness")
        }
        assert set(paired.columns) == expected_cols

    def test_score_worksheet_equals_compute_agreement_of_build_paired_scores(self, tmp_path, two_arm_fixture):
        # score_worksheet is now a thin wrapper -- lock down that it stays equivalent
        # to calling the two pieces separately, which is exactly what the T4b.8 figure
        # generation does.
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)

        worksheet_path, rows = export_worksheet(
            judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md"
        )
        text = worksheet_path.read_text(encoding="utf-8")
        for row in rows:
            for criterion, value in row.judge_scores.items():
                text = text.replace(f"- {criterion}: \n", f"- {criterion}: {value}\n", 1)
        worksheet_path.write_text(text, encoding="utf-8")

        via_wrapper = score_worksheet(worksheet_path, judge_scores_path, references_path, sample_size=4, seed=1)
        paired = build_paired_scores(worksheet_path, judge_scores_path, references_path, sample_size=4, seed=1)
        via_pieces = compute_agreement(paired)

        pd.testing.assert_frame_equal(via_wrapper, via_pieces)

    def test_raises_on_mismatched_worksheet_same_as_score_worksheet(self, tmp_path, two_arm_fixture):
        judge_scores, references = two_arm_fixture
        judge_scores_path = tmp_path / "judge_scores.csv"
        judge_scores.to_csv(judge_scores_path, index=False)
        references_path = tmp_path / "references.jsonl"
        _write_references_jsonl(references, references_path)
        export_worksheet(judge_scores_path, references_path, sample_size=4, seed=1, out_path=tmp_path / "worksheet.md")

        stale_worksheet = tmp_path / "stale.md"
        stale_worksheet.write_text(FILLED_BLOCK, encoding="utf-8")

        with pytest.raises(ValueError, match="do not match the re-derived sample"):
            build_paired_scores(stale_worksheet, judge_scores_path, references_path, sample_size=4, seed=99)


# --------------------------------------------------------------------------
# plot_judge_human_agreement
# --------------------------------------------------------------------------


class TestPlotJudgeHumanAgreement:
    def _paired_and_agreement(self):
        rows = []
        for i in range(8):
            rows.append(
                {
                    "row_id": f"R{i:02d}", "arm": "closed_book" if i % 2 else "rag_small", "question_id": f"q{i}",
                    "judge_correctness": 1 + (i % 5), "human_correctness": 1 + ((i + 1) % 5),
                    "judge_faithfulness": 5, "human_faithfulness": 5,
                    "judge_completeness": 1 + (i % 5), "human_completeness": 1 + ((i + 2) % 5),
                    "judge_conciseness": 5, "human_conciseness": 4 if i % 3 == 0 else 5,
                }
            )
        paired = pd.DataFrame(rows)
        agreement = compute_agreement(paired)
        return paired, agreement

    def test_writes_a_nonempty_png(self, tmp_path):
        paired, agreement = self._paired_and_agreement()
        out_path = plot_judge_human_agreement(paired, agreement, tmp_path / "agreement.png", seed=0)

        assert out_path.is_file()
        assert out_path.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path):
        paired, agreement = self._paired_and_agreement()
        out_path = plot_judge_human_agreement(paired, agreement, tmp_path / "nested" / "dir" / "agreement.png", seed=0)
        assert out_path.is_file()

    def test_same_seed_is_reproducible_byte_for_byte(self, tmp_path):
        paired, agreement = self._paired_and_agreement()
        first = plot_judge_human_agreement(paired, agreement, tmp_path / "a.png", seed=7)
        second = plot_judge_human_agreement(paired, agreement, tmp_path / "b.png", seed=7)
        assert first.read_bytes() == second.read_bytes()
