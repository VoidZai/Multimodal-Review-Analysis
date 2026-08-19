"""Unit tests for cragb.eval.metrics_answer (T4b.3; M4b.md T4b.3).

None of these tests need `sentence-transformers`/`torch`/the `C:\\venv\\cragb` venv --
`embedding_similarity`/`score_arm` are generic over any `EmbeddingModel`-shaped object
(module docstring), so a small deterministic `FakeModel` stands in for a real
`SentenceTransformer` throughout. `load_model` itself (the one function that actually
imports `sentence-transformers`) is not exercised here; it is a thin, one-call wrapper
with nothing to unit test beyond "does it construct a `SentenceTransformer`", which would
require the real model and the venv to verify meaningfully.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cragb.bench.reference_answers import ABSTENTION_TEXT, make_reference_answer
from cragb.eval.metrics_answer import embedding_similarity, score_arm
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.context_builder import ContextBlock
from cragb.generate.grounded_qa import GroundedQATranscript


class FakeModel:
    """Deterministic stand-in for `SentenceTransformer.encode`, keyed by exact text.

    Raises `KeyError` on an unknown text rather than falling back to some default vector,
    so a test can never accidentally pass because of an unintended default.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.encode_call_count = 0
        self.encode_call_sizes: list[int] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> Any:
        self.encode_call_count += 1
        self.encode_call_sizes.append(len(texts))
        vecs = np.array([self._vectors[t] for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


def make_closed_book_transcript(qid: str, answer_text: str) -> ClosedBookTranscript:
    return ClosedBookTranscript(
        question_id=qid, question="q?", raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), abstained=False,
    )


def make_grounded_transcript(qid: str, answer_text: str) -> GroundedQATranscript:
    context = ContextBlock(text="ctx", doc_ids=("101",), photo_flags={"101": False})
    return GroundedQATranscript(
        question_id=qid, question="q?", context=context, raw_completion=answer_text,
        answer_text=answer_text, cited_doc_ids=(), cited_photo_ids=(), abstained=False,
    )


# --------------------------------------------------------------------------
# embedding_similarity
# --------------------------------------------------------------------------


class TestEmbeddingSimilarity:
    def test_identical_strings_similarity_is_one(self):
        model = FakeModel({"Runs small, size up.": [1.0, 2.0, 3.0]})
        sim = embedding_similarity("Runs small, size up.", "Runs small, size up.", model)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_texts_similarity_is_zero(self):
        model = FakeModel({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        sim = embedding_similarity("a", "b", model)
        assert sim == pytest.approx(0.0)

    def test_opposite_texts_similarity_is_negative_one(self):
        model = FakeModel({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
        sim = embedding_similarity("a", "b", model)
        assert sim == pytest.approx(-1.0)

    def test_result_is_within_valid_cosine_range(self):
        model = FakeModel({"a": [3.0, -4.0, 1.0], "b": [-2.0, 5.0, -1.0]})
        sim = embedding_similarity("a", "b", model)
        assert -1.0 <= sim <= 1.0

    def test_two_correct_abstentions_similarity_is_one(self):
        # A candidate that correctly abstains and a reference that is itself the
        # canonical abstention text are, textually, the exact same string -- the
        # similarity metric should agree that's a perfect match, same as any other
        # identical-string pair.
        model = FakeModel({ABSTENTION_TEXT: [1.0, 1.0, 1.0]})
        sim = embedding_similarity(ABSTENTION_TEXT, ABSTENTION_TEXT, model)
        assert sim == pytest.approx(1.0)

    def test_unnormalized_input_vectors_are_normalized_before_comparison(self):
        # Same direction, very different magnitudes -- cosine similarity must ignore
        # magnitude entirely (this is what normalize_embeddings=True buys).
        model = FakeModel({"a": [1.0, 0.0], "b": [50.0, 0.0]})
        sim = embedding_similarity("a", "b", model)
        assert sim == pytest.approx(1.0)


# --------------------------------------------------------------------------
# score_arm
# --------------------------------------------------------------------------


class TestScoreArm:
    def test_scores_each_transcript_against_its_reference_in_order(self):
        transcripts = [
            make_closed_book_transcript("q1", "Runs small."),
            make_closed_book_transcript("q2", "Fits true to size."),
        ]
        references = {
            "q1": make_reference_answer("q1", "Runs small [101]."),
            "q2": make_reference_answer("q2", "Completely unrelated statement."),
        }
        model = FakeModel(
            {
                "Runs small.": [1.0, 0.0],
                "Runs small [101].": [1.0, 0.0],
                "Fits true to size.": [1.0, 0.0],
                "Completely unrelated statement.": [0.0, 1.0],
            }
        )

        result = score_arm(transcripts, references, model)

        assert list(result["question_id"]) == ["q1", "q2"]
        assert result.loc[result["question_id"] == "q1", "similarity"].iloc[0] == pytest.approx(1.0)
        assert result.loc[result["question_id"] == "q2", "similarity"].iloc[0] == pytest.approx(0.0)

    def test_works_for_grounded_and_closed_book_transcripts_alike(self):
        transcripts = [
            make_grounded_transcript("q1", "Answer text."),
            make_closed_book_transcript("q2", "Answer text."),
        ]
        references = {
            "q1": make_reference_answer("q1", "Answer text."),
            "q2": make_reference_answer("q2", "Answer text."),
        }
        model = FakeModel({"Answer text.": [1.0, 2.0]})

        result = score_arm(transcripts, references, model)

        assert list(result["question_id"]) == ["q1", "q2"]
        assert result["similarity"].tolist() == pytest.approx([1.0, 1.0])

    def test_missing_reference_raises_key_error(self):
        transcripts = [make_closed_book_transcript("q1", "Answer.")]
        with pytest.raises(KeyError, match="q1"):
            score_arm(transcripts, references={}, model=FakeModel({}))

    def test_empty_transcripts_returns_empty_dataframe_with_expected_columns(self):
        result = score_arm([], references={}, model=FakeModel({}))
        assert list(result.columns) == ["question_id", "similarity"]
        assert len(result) == 0

    def test_batches_answers_and_references_in_two_encode_calls_total(self):
        # Not one encode() call per (answer, reference) pair -- module docstring's whole
        # reason for score_arm existing separately from embedding_similarity.
        transcripts = [
            make_closed_book_transcript("q1", "A1"),
            make_closed_book_transcript("q2", "A2"),
            make_closed_book_transcript("q3", "A3"),
        ]
        references = {
            "q1": make_reference_answer("q1", "R1"),
            "q2": make_reference_answer("q2", "R2"),
            "q3": make_reference_answer("q3", "R3"),
        }
        model = FakeModel(
            {"A1": [1, 0], "A2": [1, 0], "A3": [1, 0], "R1": [1, 0], "R2": [1, 0], "R3": [1, 0]}
        )

        score_arm(transcripts, references, model)

        assert model.encode_call_count == 2
        assert model.encode_call_sizes == [3, 3]  # 3 answers, then 3 references
