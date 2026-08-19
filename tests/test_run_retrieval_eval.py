"""Unit tests for cragb.eval.run_retrieval_eval (T3.5/T3.6/T3.7; M3.md T3.5/T3.6/T3.7).

Config loading/validation, the BM25 half of index building, and
`score_retriever`/`summarize_metrics`/`compute_significance` (T3.6) run
unconditionally on BM25-only fixtures. The dense-index and full
`run_rq2_eval` tests are skipped wherever `torch`/`sentence-transformers`/
`faiss` aren't importable (this project's main Windows environment can't
install that stack — PLAN.md §14.1; run via
`C:\\venv\\cragb\\Scripts\\python.exe` for real dense coverage), mirroring
the `requires_dense` pattern already established in `tests/test_retrieval.py`.
T3.7's `summarize_recall_by_type`/`h2_interaction_summary`/`plot_recall_per_type`
consume already-computed per-question DataFrames, so they need neither
retriever and run unconditionally too.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cragb.eval.cragb_questions import RetrievalQuestion
from cragb.eval.run_retrieval_eval import (
    DenseRetrieverConfig,
    IndexBuildReport,
    RetrievalEvalConfig,
    SmokeHit,
    build_all_retrievers,
    build_retriever,
    compute_significance,
    h2_interaction_summary,
    load_retrieval_eval_config,
    plot_recall_at_k,
    plot_recall_per_type,
    run_rq2_eval,
    score_retriever,
    summarize_metrics,
    summarize_recall_by_type,
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


def make_questions() -> list[RetrievalQuestion]:
    return [
        RetrievalQuestion(
            id="fit_q",
            type="fit_sizing",
            question="does this run small",
            is_negative=False,
            relevant_ids=frozenset({"0"}),
        ),
        RetrievalQuestion(
            id="fabric_q",
            type="fabric_quality",
            question="does the fabric hold up after washing",
            is_negative=False,
            relevant_ids=frozenset({"1"}),
        ),
    ]


def identity_chunk_to_parent(corpus: pd.DataFrame) -> dict[str, str]:
    chunks = chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))
    return dict(zip(chunks["chunk_id"], chunks["parent_doc_id"]))


class TestScoreRetriever:
    def test_output_has_one_row_per_question_per_k(self):
        corpus = make_corpus()
        chunks = chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))
        retriever, _report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="does this run small"
        )
        result = score_retriever(
            retriever, identity_chunk_to_parent(corpus), make_questions(), k_values=(1, 2)
        )
        assert len(result) == len(make_questions()) * 2
        assert set(result.columns) == {"question_id", "type", "k", "recall", "hit", "ndcg", "mrr"}

    def test_obviously_relevant_doc_found_at_k1(self):
        corpus = make_corpus()
        chunks = chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))
        retriever, _report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="does this run small"
        )
        result = score_retriever(
            retriever, identity_chunk_to_parent(corpus), make_questions(), k_values=(1,)
        )
        fit_row = result.loc[result["question_id"] == "fit_q"].iloc[0]
        assert fit_row["recall"] == 1.0
        assert fit_row["hit"] == 1.0

    def test_all_metric_scores_in_valid_range(self):
        corpus = make_corpus()
        chunks = chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))
        retriever, _report = build_retriever(
            "bm25", BM25Retriever(), chunks, smoke_query="does this run small"
        )
        result = score_retriever(
            retriever, identity_chunk_to_parent(corpus), make_questions(), k_values=(1, 2, 3)
        )
        for metric in ("recall", "hit", "ndcg", "mrr"):
            assert result[metric].between(0.0, 1.0).all()


class TestSummarizeMetrics:
    def _per_question(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q1", "q2"],
                "type": ["fit_sizing"] * 4,
                "k": [1, 1, 3, 3],
                "recall": [1.0, 0.0, 1.0, 0.5],
                "hit": [1.0, 0.0, 1.0, 1.0],
                "ndcg": [1.0, 0.0, 1.0, 0.6],
                "mrr": [1.0, 0.0, 1.0, 0.5],
            }
        )

    def test_columns_and_row_count(self):
        summary = summarize_metrics(
            self._per_question(), retriever="bm25", n_boot=500, rng=np.random.default_rng(0)
        )
        expected_cols = {"retriever", "k", "n_questions"}
        for metric in ("recall", "hit", "ndcg", "mrr"):
            expected_cols |= {f"{metric}_mean", f"{metric}_ci_lo", f"{metric}_ci_hi"}
        assert set(summary.columns) == expected_cols
        assert set(summary["k"]) == {1, 3}
        assert (summary["retriever"] == "bm25").all()

    def test_means_match_manual_average(self):
        summary = summarize_metrics(
            self._per_question(), retriever="bm25", n_boot=500, rng=np.random.default_rng(0)
        )
        row_k1 = summary.loc[summary["k"] == 1].iloc[0]
        assert row_k1["recall_mean"] == pytest.approx(0.5)
        assert row_k1["n_questions"] == 2


class TestComputeSignificance:
    def _per_question(self, recall_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "question_id": [f"q{i}" for i in range(len(recall_values))],
                "type": ["fit_sizing"] * len(recall_values),
                "k": [1] * len(recall_values),
                "recall": recall_values,
                "hit": recall_values,
                "ndcg": recall_values,
                "mrr": recall_values,
            }
        )

    def test_columns_and_row_count(self):
        bm25 = self._per_question([0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3, 0.85, 0.15])
        dense = self._per_question([0.1, 0.9, 0.2, 0.8, 0.05, 0.95, 0.3, 0.7, 0.15, 0.85])
        result = compute_significance(bm25, dense, k_values=(1,))
        assert set(result.columns) == {
            "k", "recall_wilcoxon_p", "hit_wilcoxon_p", "ndcg_wilcoxon_p", "mrr_wilcoxon_p",
        }
        assert len(result) == 1

    def test_p_values_within_valid_range(self):
        bm25 = self._per_question([0.9, 0.1, 0.8, 0.2, 0.95])
        dense = self._per_question([0.1, 0.9, 0.2, 0.8, 0.05])
        result = compute_significance(bm25, dense, k_values=(1,))
        for metric in ("recall", "hit", "ndcg", "mrr"):
            assert 0.0 <= result.loc[0, f"{metric}_wilcoxon_p"] <= 1.0

    def test_pairing_is_by_question_id_not_row_order(self):
        # Shuffle dense's row order relative to bm25's; a naive positional
        # pairing (rather than a question_id join) would silently compare
        # the wrong questions against each other.
        bm25 = self._per_question([1.0, 0.0, 1.0, 0.0])
        dense = self._per_question([1.0, 0.0, 1.0, 0.0])
        dense = dense.iloc[::-1].reset_index(drop=True)  # reverse row order
        result = compute_significance(bm25, dense, k_values=(1,))
        # identical scores once correctly paired by question_id -> p == 1.0
        assert result.loc[0, "recall_wilcoxon_p"] == 1.0

    def test_mismatched_question_sets_raise(self):
        bm25 = self._per_question([1.0, 0.0])
        dense = pd.DataFrame(
            {
                "question_id": ["q0", "q_different"],
                "type": ["fit_sizing"] * 2,
                "k": [1, 1],
                "recall": [1.0, 0.0],
                "hit": [1.0, 0.0],
                "ndcg": [1.0, 0.0],
                "mrr": [1.0, 0.0],
            }
        )
        with pytest.raises(ValueError, match="different questions"):
            compute_significance(bm25, dense, k_values=(1,))


class TestPlotRecallAtK:
    def _rq2_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "retriever": ["bm25", "bm25", "dense", "dense"],
                "k": [1, 3, 1, 3],
                "recall_mean": [0.2, 0.4, 0.25, 0.45],
                "recall_ci_lo": [0.15, 0.35, 0.20, 0.40],
                "recall_ci_hi": [0.25, 0.45, 0.30, 0.50],
            }
        )

    def test_writes_a_nonempty_png(self, tmp_path):
        out_path = tmp_path / "recall_at_k.png"
        plot_recall_at_k(self._rq2_table(), out_path)
        assert out_path.is_file()
        assert out_path.stat().st_size > 0


@requires_dense
class TestRunRq2Eval:
    def _config(self, k_values=(1, 2)):
        return RetrievalEvalConfig(
            seed=42,
            corpus_in="data/processed/corpus_v1.parquet",
            questions_in="benchmark/cragb_v1.jsonl",
            chunking_config_path="configs/chunking.yaml",
            k_values=k_values,
            build_report_out="results/tables/retrieval_index_build_v1.json",
            dense=DenseRetrieverConfig(model_name="BAAI/bge-small-en-v1.5", batch_size=8, device="cpu"),
        )

    def test_table_has_one_row_per_retriever_per_k(self):
        rq2_table, per_question, build_reports = run_rq2_eval(
            make_corpus(),
            self._config(),
            make_questions(),
            rng=np.random.default_rng(0),
            show_progress=False,
        )
        assert len(rq2_table) == 2 * 2  # 2 retrievers x 2 k values
        assert set(rq2_table["retriever"]) == {"bm25", "dense"}
        assert set(build_reports.keys()) == {"bm25", "dense"}
        assert set(per_question.keys()) == {"bm25", "dense"}

    def test_significance_columns_present_and_valid(self):
        rq2_table, _per_question, _build_reports = run_rq2_eval(
            make_corpus(),
            self._config(),
            make_questions(),
            rng=np.random.default_rng(0),
            show_progress=False,
        )
        for metric in ("recall", "hit", "ndcg", "mrr"):
            col = f"{metric}_wilcoxon_p"
            assert col in rq2_table.columns
            assert rq2_table[col].between(0.0, 1.0).all()

    def test_significance_is_identical_across_retriever_rows_at_same_k(self):
        rq2_table, _per_question, _build_reports = run_rq2_eval(
            make_corpus(),
            self._config(),
            make_questions(),
            rng=np.random.default_rng(0),
            show_progress=False,
        )
        for k, group in rq2_table.groupby("k"):
            assert group["recall_wilcoxon_p"].nunique() == 1


def make_per_question_by_type(recall_by_type_and_retriever: dict) -> dict[str, pd.DataFrame]:
    """Build synthetic {"bm25": df, "dense": df} per-question fixtures.

    `recall_by_type_and_retriever` example:
        {"fit_sizing": {"bm25": [1.0, 0.0, 1.0], "dense": [0.0, 0.0, 1.0]},
         "colour_appearance": {"bm25": [0.0, 0.0], "dense": [1.0, 1.0]}}
    Every question gets a unique id and k=5, matching what
    `score_retriever`'s real output looks like.
    """
    rows_by_retriever: dict[str, list[dict[str, object]]] = {"bm25": [], "dense": []}
    counter = 0
    for qtype, by_retriever in recall_by_type_and_retriever.items():
        n = len(next(iter(by_retriever.values())))
        for i in range(n):
            question_id = f"{qtype}_{i}_{counter}"
            counter += 1
            for retriever, values in by_retriever.items():
                rows_by_retriever[retriever].append(
                    {"question_id": question_id, "type": qtype, "k": 5, "recall": values[i]}
                )
    return {name: pd.DataFrame(rows) for name, rows in rows_by_retriever.items()}


class TestSummarizeRecallByType:
    def test_columns_and_one_row_per_retriever_per_type(self):
        per_question = make_per_question_by_type(
            {
                "fit_sizing": {"bm25": [1.0, 0.0, 1.0], "dense": [0.0, 0.0, 1.0]},
                "colour_appearance": {"bm25": [0.0, 0.0], "dense": [1.0, 1.0]},
            }
        )
        table = summarize_recall_by_type(per_question, k=5, n_boot=500, rng=np.random.default_rng(0))
        assert set(table.columns) == {
            "retriever", "type", "k", "recall_mean", "recall_ci_lo", "recall_ci_hi", "n_questions",
        }
        assert len(table) == 2 * 2  # 2 retrievers x 2 types
        assert (table["k"] == 5).all()

    def test_recall_means_match_manual_averages(self):
        per_question = make_per_question_by_type(
            {"fit_sizing": {"bm25": [1.0, 0.0, 1.0], "dense": [0.0, 0.0, 1.0]}}
        )
        table = summarize_recall_by_type(per_question, k=5, n_boot=500, rng=np.random.default_rng(0))
        bm25_row = table.loc[(table["retriever"] == "bm25") & (table["type"] == "fit_sizing")].iloc[0]
        dense_row = table.loc[(table["retriever"] == "dense") & (table["type"] == "fit_sizing")].iloc[0]
        assert bm25_row["recall_mean"] == pytest.approx(2 / 3)
        assert dense_row["recall_mean"] == pytest.approx(1 / 3)
        assert bm25_row["n_questions"] == 3

    def test_every_type_present_for_every_retriever(self):
        per_question = make_per_question_by_type(
            {
                "fit_sizing": {"bm25": [1.0], "dense": [0.0]},
                "colour_appearance": {"bm25": [0.0], "dense": [1.0]},
                "durability": {"bm25": [0.5], "dense": [0.5]},
            }
        )
        table = summarize_recall_by_type(per_question, k=5, n_boot=500, rng=np.random.default_rng(0))
        for retriever in ("bm25", "dense"):
            types_present = set(table.loc[table["retriever"] == retriever, "type"])
            assert types_present == {"fit_sizing", "colour_appearance", "durability"}

    def test_missing_k_raises(self):
        per_question = make_per_question_by_type({"fit_sizing": {"bm25": [1.0], "dense": [0.0]}})
        with pytest.raises(ValueError, match="k=10 not found"):
            summarize_recall_by_type(per_question, k=10, n_boot=500)


class TestH2InteractionSummary:
    def test_identifies_bm25_leader(self):
        by_type_table = pd.DataFrame(
            {
                "retriever": ["bm25", "dense"],
                "type": ["fit_sizing", "fit_sizing"],
                "k": [5, 5],
                "recall_mean": [0.8, 0.3],
                "recall_ci_lo": [0.7, 0.2],
                "recall_ci_hi": [0.9, 0.4],
                "n_questions": [10, 10],
            }
        )
        result = h2_interaction_summary(by_type_table)
        assert result.loc[result["type"] == "fit_sizing", "leader"].iloc[0] == "bm25"

    def test_identifies_dense_leader(self):
        by_type_table = pd.DataFrame(
            {
                "retriever": ["bm25", "dense"],
                "type": ["colour_appearance", "colour_appearance"],
                "k": [5, 5],
                "recall_mean": [0.2, 0.7],
                "recall_ci_lo": [0.1, 0.6],
                "recall_ci_hi": [0.3, 0.8],
                "n_questions": [10, 10],
            }
        )
        result = h2_interaction_summary(by_type_table)
        assert result.loc[result["type"] == "colour_appearance", "leader"].iloc[0] == "dense"

    def test_identifies_tie(self):
        by_type_table = pd.DataFrame(
            {
                "retriever": ["bm25", "dense"],
                "type": ["value", "value"],
                "k": [5, 5],
                "recall_mean": [0.5, 0.5],
                "recall_ci_lo": [0.4, 0.4],
                "recall_ci_hi": [0.6, 0.6],
                "n_questions": [10, 10],
            }
        )
        result = h2_interaction_summary(by_type_table)
        assert result.loc[result["type"] == "value", "leader"].iloc[0] == "tie"

    def test_output_columns(self):
        by_type_table = pd.DataFrame(
            {
                "retriever": ["bm25", "dense"],
                "type": ["value", "value"],
                "k": [5, 5],
                "recall_mean": [0.5, 0.6],
                "recall_ci_lo": [0.4, 0.5],
                "recall_ci_hi": [0.6, 0.7],
                "n_questions": [10, 10],
            }
        )
        result = h2_interaction_summary(by_type_table)
        assert list(result.columns) == ["type", "bm25_recall_mean", "dense_recall_mean", "leader"]


class TestPlotRecallPerType:
    def _by_type_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "retriever": ["bm25", "dense", "bm25", "dense"],
                "type": ["fit_sizing", "fit_sizing", "colour_appearance", "colour_appearance"],
                "k": [5, 5, 5, 5],
                "recall_mean": [0.3, 0.35, 0.4, 0.2],
                "recall_ci_lo": [0.2, 0.25, 0.3, 0.1],
                "recall_ci_hi": [0.4, 0.45, 0.5, 0.3],
                "n_questions": [10, 10, 8, 8],
            }
        )

    def test_writes_a_nonempty_png(self, tmp_path):
        out_path = tmp_path / "recall_per_type.png"
        plot_recall_per_type(self._by_type_table(), out_path, k=5)
        assert out_path.is_file()
        assert out_path.stat().st_size > 0
