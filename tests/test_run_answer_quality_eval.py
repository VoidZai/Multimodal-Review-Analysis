"""Unit tests for cragb.eval.run_answer_quality_eval (T4b.7; M4b.md T4b.7).

`build_combined_scores`/`build_comparison_table`/`validate_comparison_table` are all
generic over any `EmbeddingModel`-shaped object (module docstring), so a small
deterministic `FakeModel` -- same pattern as `tests/test_metrics_answer.py` -- stands in
for a real `SentenceTransformer` throughout. `load_arm_transcripts` reads from
`cragb.eval.run_answer_generation.ARM_DEFAULT_OUT`; tests that need real transcripts
monkeypatch that module attribute to point at `tmp_path` fixtures instead of the real
project data. `main()` itself (real judge-scores CSV, real embedding model download,
real file writes) is intentionally not unit-tested here, the same convention every other
batch-driver module in this project follows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cragb.bench.reference_answers import make_reference_answer
from cragb.eval.run_answer_quality_eval import (
    METRICS,
    build_combined_scores,
    build_comparison_table,
    validate_comparison_table,
)
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.closed_book_qa import write_transcripts_jsonl as write_closed_book_jsonl
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript
from cragb.generate.grounded_qa import write_transcripts_jsonl as write_grounded_qa_jsonl


class FakeModel:
    """Deterministic stand-in for `SentenceTransformer.encode`, keyed by exact text."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        vecs = np.array([self._vectors[t] for t in texts], dtype=np.float32)
        if kwargs.get("normalize_embeddings", True):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


def make_closed_book_transcript(qid: str, answer_text: str) -> ClosedBookTranscript:
    return ClosedBookTranscript(
        question_id=qid, question=f"Question for {qid}?", raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), abstained=False,
    )


def make_grounded_transcript(qid: str, answer_text: str) -> GroundedQATranscript:
    context = ContextBlock(text="ctx", doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question=f"Question for {qid}?", context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), cited_photo_ids=(), abstained=False,
    )


@pytest.fixture
def two_arm_fixture(tmp_path, monkeypatch):
    """3 closed_book + 3 rag_small transcripts, matching judge scores, references, and
    a FakeModel that makes every arm's answer identical to its own reference (perfect
    similarity=1.0 for everyone) -- individual tests override what they need beyond that.
    """
    closed_book_transcripts = [make_closed_book_transcript(f"q{i}", f"cb answer {i}") for i in range(3)]
    rag_small_transcripts = [make_grounded_transcript(f"q{i}", f"rag answer {i}") for i in range(3)]

    cb_path = write_closed_book_jsonl(closed_book_transcripts, tmp_path / "cb.jsonl")
    rs_path = write_grounded_qa_jsonl(rag_small_transcripts, tmp_path / "rs.jsonl")
    monkeypatch.setattr(
        "cragb.eval.run_answer_quality_eval.ARM_DEFAULT_OUT",
        {"closed_book": str(cb_path), "rag_small": str(rs_path), "rag_large": str(rs_path)},
    )

    references = {f"q{i}": make_reference_answer(f"q{i}", f"reference {i}") for i in range(3)}

    judge_rows = []
    for i in range(3):
        judge_rows.append(
            {"arm": "closed_book", "question_id": f"q{i}", "correctness": 1, "faithfulness": 5,
             "completeness": 1, "conciseness": 5}
        )
        judge_rows.append(
            {"arm": "rag_small", "question_id": f"q{i}", "correctness": 4, "faithfulness": 5,
             "completeness": 4, "conciseness": 4}
        )
    judge_scores = pd.DataFrame(judge_rows)

    vectors = {"reference 0": [1, 0], "reference 1": [1, 0], "reference 2": [1, 0]}
    for i in range(3):
        vectors[f"cb answer {i}"] = [0, 1]  # orthogonal to its reference -- low similarity
        vectors[f"rag answer {i}"] = [1, 0]  # identical direction -- high similarity
    model = FakeModel(vectors)

    return references, judge_scores, model


# --------------------------------------------------------------------------
# build_combined_scores
# --------------------------------------------------------------------------


class TestBuildCombinedScores:
    def test_merges_similarity_and_judge_scores(self, two_arm_fixture):
        references, judge_scores, model = two_arm_fixture
        combined = build_combined_scores(("closed_book", "rag_small"), references, judge_scores, model)

        assert list(combined.columns) == ["arm", "question_id", *METRICS]
        assert len(combined) == 6  # 3 questions x 2 arms
        assert set(combined["arm"]) == {"closed_book", "rag_small"}

    def test_similarity_reflects_the_embedding_model(self, two_arm_fixture):
        references, judge_scores, model = two_arm_fixture
        combined = build_combined_scores(("closed_book", "rag_small"), references, judge_scores, model)

        cb_sim = combined.loc[combined["arm"] == "closed_book", "similarity"]
        rag_sim = combined.loc[combined["arm"] == "rag_small", "similarity"]
        assert (cb_sim < 0.1).all()  # orthogonal vectors -> ~0 similarity
        assert (rag_sim > 0.9).all()  # identical-direction vectors -> ~1.0 similarity

    def test_judge_columns_carried_through_unchanged(self, two_arm_fixture):
        references, judge_scores, model = two_arm_fixture
        combined = build_combined_scores(("closed_book", "rag_small"), references, judge_scores, model)

        row = combined[(combined["arm"] == "closed_book") & (combined["question_id"] == "q0")].iloc[0]
        assert row["correctness"] == 1
        assert row["faithfulness"] == 5

    def test_mismatched_similarity_and_judge_question_sets_raises(self, two_arm_fixture):
        references, judge_scores, model = two_arm_fixture
        # Drop one arm's judge row for one question -- similarity still has it.
        judge_scores = judge_scores[~((judge_scores["arm"] == "closed_book") & (judge_scores["question_id"] == "q0"))]

        with pytest.raises(ValueError, match="did not merge one-to-one"):
            build_combined_scores(("closed_book", "rag_small"), references, judge_scores, model)


# --------------------------------------------------------------------------
# build_comparison_table
# --------------------------------------------------------------------------


def make_combined_scores(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["arm", "question_id", *METRICS])


class TestBuildComparisonTable:
    def test_produces_one_row_per_metric_per_arm(self):
        rows = []
        for i in range(5):
            rows.append({"arm": "a", "question_id": f"q{i}", "similarity": 0.9, "correctness": 5,
                         "faithfulness": 5, "completeness": 5, "conciseness": 5})
            rows.append({"arm": "b", "question_id": f"q{i}", "similarity": 0.9, "correctness": 5,
                         "faithfulness": 5, "completeness": 5, "conciseness": 5})
        table = build_comparison_table(make_combined_scores(rows), "a", "b", n_boot=200, seed=1)

        assert len(table) == 2 * len(METRICS)
        assert set(table["metric"]) == set(METRICS)
        assert set(table["arm"]) == {"a", "b"}

    def test_flags_an_obvious_large_difference_as_significant(self):
        # RQ0's whole point: closed-book should score far lower on correctness than
        # RAG-small, consistently across every question -- a large, obvious,
        # consistent gap like this must come out as a small (significant) p-value.
        # This is the exact validation check M4b.md's task spec calls for.
        rows = []
        for i in range(20):
            rows.append({"arm": "closed_book", "question_id": f"q{i}", "similarity": 0.3, "correctness": 1,
                         "faithfulness": 5, "completeness": 1, "conciseness": 5})
            rows.append({"arm": "rag_small", "question_id": f"q{i}", "similarity": 0.8, "correctness": 5,
                         "faithfulness": 5, "completeness": 5, "conciseness": 4})
        table = build_comparison_table(make_combined_scores(rows), "closed_book", "rag_small", n_boot=500, seed=1)

        correctness_p = table.loc[table["metric"] == "correctness", "wilcoxon_p"].iloc[0]
        assert correctness_p < 0.01

        # Means should point the correct direction too, not just "significant".
        cb_mean = table.loc[(table["metric"] == "correctness") & (table["arm"] == "closed_book"), "mean"].iloc[0]
        rag_mean = table.loc[(table["metric"] == "correctness") & (table["arm"] == "rag_small"), "mean"].iloc[0]
        assert rag_mean > cb_mean

    def test_no_difference_gives_a_high_p_value(self):
        rows = []
        for i in range(10):
            rows.append({"arm": "a", "question_id": f"q{i}", "similarity": 0.7, "correctness": 3,
                         "faithfulness": 3, "completeness": 3, "conciseness": 3})
            rows.append({"arm": "b", "question_id": f"q{i}", "similarity": 0.7, "correctness": 3,
                         "faithfulness": 3, "completeness": 3, "conciseness": 3})
        table = build_comparison_table(make_combined_scores(rows), "a", "b", n_boot=200, seed=1)
        # Identical scores on every question -> paired_significance's own documented
        # "no evidence of a difference" shortcut (p=1.0).
        assert (table["wilcoxon_p"] == 1.0).all()

    def test_wilcoxon_p_is_identical_on_both_arms_rows_for_a_metric(self):
        rows = [
            {"arm": "a", "question_id": "q0", "similarity": 0.1, "correctness": 1, "faithfulness": 1,
             "completeness": 1, "conciseness": 1},
            {"arm": "a", "question_id": "q1", "similarity": 0.9, "correctness": 5, "faithfulness": 5,
             "completeness": 5, "conciseness": 5},
            {"arm": "b", "question_id": "q0", "similarity": 0.9, "correctness": 5, "faithfulness": 5,
             "completeness": 5, "conciseness": 5},
            {"arm": "b", "question_id": "q1", "similarity": 0.1, "correctness": 1, "faithfulness": 1,
             "completeness": 1, "conciseness": 1},
        ]
        table = build_comparison_table(make_combined_scores(rows), "a", "b", n_boot=100, seed=1)
        for metric in METRICS:
            p_values = table.loc[table["metric"] == metric, "wilcoxon_p"].unique()
            assert len(p_values) == 1

    def test_mismatched_question_sets_between_arms_raises(self):
        rows = [
            {"arm": "a", "question_id": "q0", "similarity": 0.5, "correctness": 3, "faithfulness": 3,
             "completeness": 3, "conciseness": 3},
            {"arm": "b", "question_id": "q1", "similarity": 0.5, "correctness": 3, "faithfulness": 3,
             "completeness": 3, "conciseness": 3},
        ]
        with pytest.raises(ValueError, match="do not cover the same question_id set"):
            build_comparison_table(make_combined_scores(rows), "a", "b")


# --------------------------------------------------------------------------
# validate_comparison_table
# --------------------------------------------------------------------------


class TestValidateComparisonTable:
    def test_passes_on_a_well_formed_table(self):
        table = pd.DataFrame(
            [{"metric": "correctness", "arm": "a", "n": 10, "mean": 3.0, "ci_lo": 2.5, "ci_hi": 3.5, "wilcoxon_p": 0.04}]
        )
        validate_comparison_table(table)  # no raise

    def test_raises_when_mean_is_below_ci_lo(self):
        table = pd.DataFrame(
            [{"metric": "correctness", "arm": "a", "n": 10, "mean": 2.0, "ci_lo": 2.5, "ci_hi": 3.5, "wilcoxon_p": 0.04}]
        )
        with pytest.raises(ValueError, match="outside its own bootstrap CI"):
            validate_comparison_table(table)

    def test_raises_when_mean_is_above_ci_hi(self):
        table = pd.DataFrame(
            [{"metric": "correctness", "arm": "a", "n": 10, "mean": 4.0, "ci_lo": 2.5, "ci_hi": 3.5, "wilcoxon_p": 0.04}]
        )
        with pytest.raises(ValueError, match="outside its own bootstrap CI"):
            validate_comparison_table(table)

    def test_raises_on_p_value_above_one(self):
        table = pd.DataFrame(
            [{"metric": "correctness", "arm": "a", "n": 10, "mean": 3.0, "ci_lo": 2.5, "ci_hi": 3.5, "wilcoxon_p": 1.5}]
        )
        with pytest.raises(ValueError, match="wilcoxon_p out of"):
            validate_comparison_table(table)

    def test_raises_on_negative_p_value(self):
        table = pd.DataFrame(
            [{"metric": "correctness", "arm": "a", "n": 10, "mean": 3.0, "ci_lo": 2.5, "ci_hi": 3.5, "wilcoxon_p": -0.1}]
        )
        with pytest.raises(ValueError, match="wilcoxon_p out of"):
            validate_comparison_table(table)
