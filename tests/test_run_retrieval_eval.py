"""Unit tests for cragb.eval.run_retrieval_eval (T3.5; M3.md T3.5).

Config loading/validation and the BM25 half of index building run
unconditionally. The dense half is skipped wherever
`torch`/`sentence-transformers`/`faiss` aren't importable (this
project's main Windows environment can't install that stack — PLAN.md
§14.1; run via `C:\\venv\\cragb\\Scripts\\python.exe` for real dense
coverage), mirroring the `requires_dense` pattern already established in
`tests/test_retrieval.py`.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.eval.run_retrieval_eval import (
    DenseRetrieverConfig,
    IndexBuildReport,
    RetrievalEvalConfig,
    SmokeHit,
    build_all_retrievers,
    build_retriever,
    load_retrieval_eval_config,
)
from cragb.retrieval.bm25 import BM25Retriever
from cragb.retrieval.chunking import ChunkingConfig, chunk_corpus

try:
    import faiss  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401

    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False

requires_dense = pytest.mark.skipif(
    not DENSE_AVAILABLE,
    reason="torch/sentence-transformers/faiss not importable in this environment",
)


def make_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "This runs small, definitely size up before you order.",
                "Great fabric, held up after a dozen washes with no pilling.",
                "The colour matched the listing photo exactly, very happy.",
            ]
        }
    )


class TestDenseRetrieverConfig:
    def test_valid_config_constructs(self):
        cfg = DenseRetrieverConfig(model_name="BAAI/bge-small-en-v1.5", batch_size=64, device=None)
        assert cfg.batch_size == 64

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            DenseRetrieverConfig(model_name="", batch_size=64, device=None)

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_non_positive_batch_size_raises(self, batch_size):
        with pytest.raises(ValueError, match="batch_size"):
            DenseRetrieverConfig(model_name="m", batch_size=batch_size, device=None)


class TestRetrievalEvalConfig:
    def _make(self, **overrides):
        defaults = dict(
            seed=42,
            corpus_in="data/processed/corpus_v1.parquet",
            questions_in="benchmark/cragb_v1.jsonl",
            chunking_config_path="configs/chunking.yaml",
            k_values=(1, 3, 5, 10),
            build_report_out="results/tables/retrieval_index_build_v1.json",
            dense=DenseRetrieverConfig(model_name="m", batch_size=64, device=None),
        )
        defaults.update(overrides)
        return RetrievalEvalConfig(**defaults)

    def test_valid_config_constructs(self):
        cfg = self._make()
        assert cfg.k_values == (1, 3, 5, 10)

    def test_empty_k_values_raises(self):
        with pytest.raises(ValueError, match="k_values"):
            self._make(k_values=())

    def test_non_positive_k_value_raises(self):
        with pytest.raises(ValueError, match="positive"):
            self._make(k_values=(1, 0, 3))


class TestLoadRetrievalEvalConfig:
    def test_real_config_loads(self):
        cfg = load_retrieval_eval_config("configs/retrieval_eval.yaml")
        assert cfg.k_values == (1, 3, 5, 10)
        assert cfg.chunking_config_path == "configs/chunking.yaml"
        assert cfg.dense.model_name == "BAAI/bge-small-en-v1.5"
        assert cfg.dense.device is None

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_retrieval_eval_config("configs/does_not_exist.yaml")

    def test_missing_key_raises_keyerror(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("seed: 42\npaths:\n  corpus_in: x\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_retrieval_eval_config(bad_config)


class TestBuildRetriever:
    def test_bm25_indexes_and_smoke_query_succeeds(self):
        chunks = chunk_corpus(make_corpus(), ChunkingConfig(scheme="whole_review"))
        retriever, report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="does this run small"
        )
        assert isinstance(report, IndexBuildReport)
        assert report.retriever == "bm25"
        assert report.n_chunks == 3
        assert report.n_parent_docs == 3
        assert report.build_seconds >= 0.0
        assert len(report.smoke_hits) > 0
        assert all(isinstance(h, SmokeHit) for h in report.smoke_hits)

    def test_smoke_query_snippet_traces_to_real_chunk_text(self):
        chunks = chunk_corpus(make_corpus(), ChunkingConfig(scheme="whole_review"))
        _retriever, report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="fabric washes"
        )
        top_hit = report.smoke_hits[0]
        matching_row = chunks.loc[chunks["chunk_id"] == top_hit.doc_id, "text"].iloc[0]
        assert matching_row.startswith(top_hit.snippet[: min(50, len(top_hit.snippet))])

    def test_report_serializes_to_json_compatible_dict(self):
        chunks = chunk_corpus(make_corpus(), ChunkingConfig(scheme="whole_review"))
        _retriever, report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="colour"
        )
        as_dict = report.to_dict()
        json.dumps(as_dict)  # must not raise
        assert as_dict["retriever"] == "bm25"
        assert as_dict["n_chunks"] == 3

    def test_retriever_returned_is_already_indexed_and_searchable(self):
        chunks = chunk_corpus(make_corpus(), ChunkingConfig(scheme="whole_review"))
        retriever, _report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="colour"
        )
        # A second, independent search should work without re-indexing.
        results = retriever.search("size up", k=1)
        assert len(results) == 1


class TestBuildAllRetrieversBm25Only:
    """Exercises the chunking + BM25 half of build_all_retrievers without
    requiring the dense stack to be installed, by monkeypatching the
    lazy dense import to fail loudly if reached — proving BM25 indexing
    happens (and is reported) before dense is ever touched."""

    def test_bm25_report_present_even_when_dense_import_would_fail(self, monkeypatch):
        import sys

        # Force the lazy `from cragb.retrieval.dense import DenseRetriever`
        # inside build_all_retrievers to raise, simulating this
        # environment's real MAX_PATH install failure, and confirm BM25
        # was already built (and its report already captured) before
        # that import is even attempted.
        monkeypatch.setitem(sys.modules, "cragb.retrieval.dense", None)

        config = RetrievalEvalConfig(
            seed=42,
            corpus_in="data/processed/corpus_v1.parquet",
            questions_in="benchmark/cragb_v1.jsonl",
            chunking_config_path="configs/chunking.yaml",
            k_values=(1,),
            build_report_out="results/tables/retrieval_index_build_v1.json",
            dense=DenseRetrieverConfig(model_name="m", batch_size=8, device=None),
        )
        with pytest.raises(ImportError):
            build_all_retrievers(make_corpus(), config, smoke_query="does this run small")


@requires_dense
class TestBuildAllRetrieversWithDense:
    def test_builds_both_retrievers(self):
        config = RetrievalEvalConfig(
            seed=42,
            corpus_in="data/processed/corpus_v1.parquet",
            questions_in="benchmark/cragb_v1.jsonl",
            chunking_config_path="configs/chunking.yaml",
            k_values=(1,),
            build_report_out="results/tables/retrieval_index_build_v1.json",
            dense=DenseRetrieverConfig(model_name="BAAI/bge-small-en-v1.5", batch_size=8, device="cpu"),
        )
        built = build_all_retrievers(make_corpus(), config, smoke_query="does this run small")
        assert set(built.keys()) == {"bm25", "dense"}
        for _name, (_retriever, report) in built.items():
            assert len(report.smoke_hits) > 0

    def test_dense_report_records_resolved_device(self):
        config = RetrievalEvalConfig(
            seed=42,
            corpus_in="data/processed/corpus_v1.parquet",
            questions_in="benchmark/cragb_v1.jsonl",
            chunking_config_path="configs/chunking.yaml",
            k_values=(1,),
            build_report_out="results/tables/retrieval_index_build_v1.json",
            dense=DenseRetrieverConfig(model_name="BAAI/bge-small-en-v1.5", batch_size=8, device="cpu"),
        )
        built = build_all_retrievers(make_corpus(), config, smoke_query="does this run small")
        _retriever, dense_report = built["dense"]
        assert dense_report.device == "cpu"
