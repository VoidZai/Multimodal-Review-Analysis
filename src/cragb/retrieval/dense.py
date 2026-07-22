"""Dense embedding retriever (T2.5; PLAN.md §3 E3, §1.3, hardware note in M2.md).

The other half of RQ2's sparse-vs-dense comparison, behind the same
`Retriever` interface `BM25Retriever` (T2.4) implements. Uses
`sentence-transformers` to embed whole-review documents with a small
model (`bge-small-en-v1.5` by default, ~130M params) and a FAISS
**flat, exact** index — deliberately no ANN/IVF/PQ tuning (PLAN.md §12):
at `corpus_v1`'s scale (~200k rows), an exact index is fast enough and
removes ANN recall loss as a confound between BM25 and dense retrieval.

Embeddings are L2-normalized before indexing, and the FAISS index is
`IndexFlatIP` (inner product) rather than `IndexFlatL2`: inner product
over unit-length vectors is exactly cosine similarity, so "highest
inner-product score" and "highest cosine similarity" are the same
ranking — no separate cosine computation needed.

Encoding runs in batches (`batch_size`, default 64) per M2.md's hardware
note: this project is developed on a laptop RTX 3050 (4-6GB VRAM class),
lighter than the RTX 4060/8GB the proposal assumes, and batching bounds
peak memory regardless of corpus size. `device=None` (the default)
auto-detects CUDA and uses the GPU when available, falling back to CPU
otherwise — a model this small (~130M params) is not the retrieval
bottleneck either way (PLAN.md §1.4 risk G: fine-tuning is the actual
VRAM-constrained stage), but the GPU is a straightforward ~10-20x
encoding speedup when present, so there's no reason to leave it idle.
Pass `device="cpu"` explicitly to force CPU regardless of GPU
availability (e.g. to keep VRAM free for something else running
alongside indexing).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from cragb.retrieval.base import Retriever, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _resolve_device(device: str | None) -> str:
    """`device` as given, or an auto-detected `"cuda"`/`"cpu"` if `None`."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DenseRetriever(Retriever):
    """Cosine-similarity dense retriever via sentence-transformers + FAISS flat index."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 64
    device: str | None = None  # None = auto-detect CUDA, else CPU

    _model: SentenceTransformer | None = field(default=None, init=False, repr=False)
    _index: faiss.Index | None = field(default=None, init=False, repr=False)
    _doc_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _resolved_device: str = field(default="", init=False, repr=False)

    def _get_model(self) -> SentenceTransformer:
        # Lazy-loaded: constructing a DenseRetriever (e.g. to pass around
        # before deciding whether to use it) shouldn't force a model
        # download/load; only indexing or searching does.
        if self._model is None:
            self._resolved_device = _resolve_device(self.device)
            logger.info("DenseRetriever loading %s on device=%s", self.model_name, self._resolved_device)
            self._model = SentenceTransformer(self.model_name, device=self._resolved_device)
        return self._model

    def index(
        self,
        corpus: pd.DataFrame,
        text_col: str = "text",
        id_col: str | None = None,
    ) -> None:
        if corpus.empty:
            raise ValueError("Cannot index an empty corpus.")

        text = corpus[text_col].fillna("").astype(str).tolist()
        doc_ids = (
            corpus[id_col].astype(str).tolist()
            if id_col is not None
            else corpus.index.astype(str).tolist()
        )
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError(
                "Document ids are not unique; pass a unique `id_col` or "
                "reset the corpus index before indexing."
            )

        model = self._get_model()
        t0 = time.monotonic()
        embeddings = model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-length, so inner product == cosine similarity
            convert_to_numpy=True,
        ).astype(np.float32)
        elapsed_s = time.monotonic() - t0

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # exact inner-product search — no ANN (PLAN.md §12)
        index.add(embeddings)

        self._index = index
        self._doc_ids = doc_ids
        logger.info(
            "DenseRetriever (%s, device=%s) indexed %d documents (dim=%d) in %.2fs "
            "(%.1f docs/s); index size %.1f MB",
            self.model_name,
            self._resolved_device,
            len(doc_ids),
            dim,
            elapsed_s,
            len(doc_ids) / elapsed_s if elapsed_s > 0 else float("inf"),
            embeddings.nbytes / 1e6,
        )

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("DenseRetriever.search() called before .index().")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        model = self._get_model()
        query_embedding = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        k_eff = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k_eff)

        return [
            SearchResult(doc_id=self._doc_ids[doc_idx], score=float(score), rank=rank)
            for rank, (doc_idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if doc_idx != -1  # FAISS pads with -1 if the index has fewer than k_eff entries
        ]
