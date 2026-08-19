"""Answer-quality similarity metric: embedding-cosine to reference (T4b.3; PLAN.md §3 E5,
§8 G2, M4b.md T4b.3).

E5's answer-quality pipeline needs a reference-based similarity score alongside T4b.4's
rubric judge (PLAN.md §8: "similarity ... + rubric AI-judge"). This module is the
similarity half: how close is a generated answer's *embedding* to its CRAGB reference
answer's embedding.

**Reuses the dense-retrieval stack rather than adding `bert-score` as a new dependency.**
`cragb.retrieval.dense.DenseRetriever` already validated `sentence-transformers` +
`BAAI/bge-small-en-v1.5` on this project's hardware (PLAN.md §14.1) — the exact same
embedding call this module needs, just applied to two answer strings instead of a corpus
of reviews. Adding `bert-score` instead would mean a second heavy model download and a
second chance at the Windows MAX_PATH install failure §14.1 already documents once; there
is no reason to risk that twice for the same kind of measurement (semantic closeness of
two texts).

**Windows/venv note (PLAN.md §14.1):** `load_model` is the *only* thing in this module
that touches `sentence-transformers`/`torch`, and that import is deferred to inside the
function body — mirroring `cragb.eval.run_retrieval_eval.build_all_retrievers`'s
precedent for the same reason: everything else here (`embedding_similarity`, `score_arm`)
is generic over "any object with a `.encode(texts, ...) -> array-like` method", so this
module, and every test that doesn't call `load_model` itself, imports and runs cleanly in
the main environment with no venv needed at all. Only call `load_model` (or run a test
that does) via the short-path venv:

    C:\\venv\\cragb\\Scripts\\python.exe -m pytest tests/test_metrics_answer.py -q
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

from cragb.bench.reference_answers import ReferenceAnswer
from cragb.generate.closed_book_qa import ClosedBookTranscript
from cragb.generate.grounded_qa import GroundedQATranscript

logger = logging.getLogger(__name__)

# The same model cragb.retrieval.dense.DenseRetriever defaults to (PLAN.md §14.1) — kept
# as this module's own constant rather than imported from there, since the two are
# independent decisions that happen to currently agree (retrieval quality and answer
# similarity are different questions); redefining avoids coupling one to the other by
# accident if either is ever revisited on its own merits.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel(Protocol):
    """The only interface this module needs from a model: sentence-transformers'
    `.encode(texts, ...) -> array-like of shape (len(texts), dim)`.

    Declared as a `Protocol`, not a concrete type, so tests can pass a small
    deterministic stand-in instead of a real `SentenceTransformer` — no model
    download, no venv, no GPU/CPU encoding time.
    """

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> EmbeddingModel:
    """Load the sentence-transformers model used for answer-quality similarity scoring.

    Deferred import (see module docstring): only reachable if this function is actually
    called, so importing `cragb.eval.metrics_answer` itself never requires
    `sentence-transformers`/`torch` to be installed.

    Args:
        model_name: a sentence-transformers model id. Defaults to `DEFAULT_MODEL_NAME`.
        device: `"cuda"`/`"cpu"`, or `None` (default) to auto-detect CUDA and fall back
            to CPU — same auto-detection `cragb.retrieval.dense.DenseRetriever` uses.

    Returns:
        A loaded `SentenceTransformer` instance, ready for `embedding_similarity`/
        `score_arm`.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    resolved_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on device=%s for answer-similarity scoring", model_name, resolved_device)
    return SentenceTransformer(model_name, device=resolved_device)


def _embed(texts: list[str], model: EmbeddingModel, batch_size: int = 64) -> np.ndarray:
    """L2-normalized embeddings for `texts`, shape `(len(texts), dim)`, `float32`.

    `normalize_embeddings=True` unit-lengths every row, so a plain dot product between
    two rows *is* their cosine similarity — the same reasoning `cragb.retrieval.dense`'s
    docstring gives for why its FAISS index can use inner product directly instead of a
    separate cosine computation.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embedding_similarity(answer_text: str, reference_text: str, model: EmbeddingModel) -> float:
    """Cosine similarity between one answer and its reference answer.

    A single-pair convenience wrapper. Scoring many pairs at once should use `score_arm`
    instead — it batches every answer and every reference through `model.encode` in two
    calls total rather than one pair (i.e. two `encode` calls) at a time, which matters
    once T4b.5 scores 180 transcripts across three arms.

    Args:
        answer_text: a generated answer (e.g. `GroundedQATranscript.answer_text`).
        reference_text: the CRAGB reference answer to compare against (e.g.
            `ReferenceAnswer.answer`).
        model: a loaded embedding model (see `load_model`), or any object exposing the
            same `.encode` interface (`EmbeddingModel`).

    Returns:
        Cosine similarity in `[-1, 1]`.
    """
    embeddings = _embed([answer_text, reference_text], model, batch_size=2)
    return float(np.dot(embeddings[0], embeddings[1]))


def score_arm(
    transcripts: list[GroundedQATranscript] | list[ClosedBookTranscript],
    references: dict[str, ReferenceAnswer],
    model: EmbeddingModel,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score one arm's every transcript against its CRAGB reference answer.

    Args:
        transcripts: one arm's generated transcripts — either `GroundedQATranscript`
            (RAG arms) or `ClosedBookTranscript` (the closed-book arm); both carry the
            `question_id`/`answer_text` fields this needs, nothing else.
        references: from `cragb.bench.reference_answers.load_reference_answers`, keyed
            by `question_id`.
        model: a loaded embedding model (see `load_model`), or any `EmbeddingModel`
            stand-in.
        batch_size: passed through to `model.encode`.

    Returns:
        A `pd.DataFrame` with one row per transcript, in `transcripts` order:
        `question_id`, `similarity` (cosine, `[-1, 1]`).

    Raises:
        KeyError: if a transcript's `question_id` has no entry in `references` —
            scoring against a missing reference would silently produce a meaningless
            comparison, so this fails loudly instead, mirroring
            `cragb.eval.citation_validity.score_transcripts`'s convention for the same
            class of "missing ground truth" gap.
    """
    missing = [t.question_id for t in transcripts if t.question_id not in references]
    if missing:
        raise KeyError(f"No reference answer for question_id(s): {missing}")

    if not transcripts:
        return pd.DataFrame(columns=["question_id", "similarity"])

    answer_texts = [t.answer_text for t in transcripts]
    reference_texts = [references[t.question_id].answer for t in transcripts]

    # Two batched encode() calls total, not one per pair -- see docstring.
    answer_embeddings = _embed(answer_texts, model, batch_size=batch_size)
    reference_embeddings = _embed(reference_texts, model, batch_size=batch_size)

    # Both sides are L2-normalized, so a row-wise dot product is a row-wise cosine
    # similarity -- (answer_embeddings * reference_embeddings).sum(axis=1) computes all
    # len(transcripts) pairwise similarities in one vectorized pass.
    similarities = np.sum(answer_embeddings * reference_embeddings, axis=1)

    return pd.DataFrame(
        {
            "question_id": [t.question_id for t in transcripts],
            "similarity": similarities.tolist(),
        }
    )
