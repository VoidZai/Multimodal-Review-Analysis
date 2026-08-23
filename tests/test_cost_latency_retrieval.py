"""Unit tests for cragb.eval.cost_latency_retrieval (T5.3; M5.md T5.3).

Config loading/validation and the BM25 half of `measure_retriever_cost_latency`/
`run_cost_latency_eval` run unconditionally on small synthetic fixtures. The dense half
of `run_cost_latency_eval` is skipped wherever `torch`/`sentence-transformers`/`faiss`
aren't importable, mirroring the `requires_dense` pattern already established in
`tests/test_retrieval.py` and `tests/test_run_retrieval_eval.py` (this project's main
Windows environment can't install that stack — PLAN.md §14.1; run via
`C:\\venv\\cragb\\Scripts\\python.exe` for real dense coverage).

Covers M5.md T5.3's own validation checks: every latency > 0 and `p50 <= p95`; `qps`
consistent with `1000 / latency_mean_ms` within rounding; `n_queries ==
n_repeats * len(questions)` (proves warm-up was excluded, not just executed);
`index_bytes` is positive.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.eval.cost_latency_retrieval import (
    CostLatencyConfig,
    RetrieverCostLatency,
    load_cost_latency_config,
    measure_retriever_cost_latency,
    run_cost_latency_eval,
)
from cragb.eval.run_retrieval_eval import DenseRetrieverConfig
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


def make_corpus(n: int = 8) -> pd.DataFrame:
    base = [
        "This runs small, definitely size up before you order.",
        "Great fabric, held up after a dozen washes with no pilling.",
        "The colour matched the listing photo exactly, very happy.",
        "Runs true to size and the material feels durable.",
        "Too tight in the shoulders, returned for a bigger size.",
        "Comfortable shoes, true to size, would buy again.",
        "The zipper broke after one wear, disappointed.",
        "Lovely dress, fits as expected, no complaints.",
    ]
    return pd.DataFrame({"text": (base * ((n // len(base)) + 1))[:n]})


def make_chunks(corpus: pd.DataFrame) -> pd.DataFrame:
    return chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))


def make_config(tmp_path, **overrides) -> CostLatencyConfig:
    defaults = dict(
        seed=42,
        corpus_in="unused",
        questions_in="unused",
        chunking_config_path="unused",
        out_path=str(tmp_path / "out.csv"),
        k=2,
        n_warmup=1,
        n_repeats=2,
        dense=DenseRetrieverConfig(model_name="BAAI/bge-small-en-v1.5", batch_size=8, device=None),
    )
    defaults.update(overrides)
    return CostLatencyConfig(**defaults)


class TestCostLatencyConfig:
    def test_valid_config_constructs(self, tmp_path):
        config = make_config(tmp_path)
        assert config.k == 2
        assert config.n_repeats == 2

    def test_non_positive_k_raises(self, tmp_path):
        with pytest.raises(ValueError, match="k must be positive"):
            make_config(tmp_path, k=0)

    def test_negative_warmup_raises(self, tmp_path):
        with pytest.raises(ValueError, match="n_warmup"):
            make_config(tmp_path, n_warmup=-1)

    def test_non_positive_repeats_raises(self, tmp_path):
        with pytest.raises(ValueError, match="n_repeats"):
            make_config(tmp_path, n_repeats=0)


class TestLoadCostLatencyConfig:
    def test_loads_the_committed_config(self):
        # configs/cost_latency.yaml itself must always parse and validate.
        config = load_cost_latency_config("configs/cost_latency.yaml")
        assert config.k > 0
        assert config.n_repeats >= 1
        assert config.dense.model_name

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_cost_latency_config("configs/does_not_exist_v1.yaml")


class TestMeasureRetrieverCostLatencyBm25:
    def test_returns_sane_latency_and_qps_stats(self):
        corpus = make_corpus()
        chunks = make_chunks(corpus)
        questions = ["does this run small", "is the fabric durable", "true to size"]

        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, questions, k=2, n_warmup=1, n_repeats=3
        )

        assert isinstance(result, RetrieverCostLatency)
        assert result.retriever == "bm25"
        assert result.n_chunks == len(chunks)
        assert result.build_seconds >= 0
        assert result.index_bytes > 0
        assert result.peak_rss_mb > 0
        # BM25 never touches CUDA.
        assert result.peak_vram_mb is None

    def test_n_queries_excludes_warmup(self):
        corpus = make_corpus()
        chunks = make_chunks(corpus)
        questions = ["does this run small", "is the fabric durable"]

        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, questions, k=2, n_warmup=5, n_repeats=3
        )

        # Only the n_repeats passes are counted/timed, regardless of n_warmup.
        assert result.n_queries == 3 * len(questions)

    def test_latency_percentiles_are_ordered(self):
        corpus = make_corpus()
        chunks = make_chunks(corpus)
        questions = ["does this run small", "is the fabric durable", "true to size"]

        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, questions, k=2, n_warmup=1, n_repeats=4
        )

        assert result.latency_p50_ms <= result.latency_p90_ms <= result.latency_p95_ms
        assert result.latency_p50_ms > 0

    def test_qps_matches_n_over_latency_mean_within_rounding(self):
        corpus = make_corpus()
        chunks = make_chunks(corpus)
        questions = ["does this run small", "is the fabric durable"]

        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, questions, k=2, n_warmup=1, n_repeats=5
        )

        # qps = n / sum(seconds) = 1 / mean(seconds), so 1000 / latency_mean_ms
        # should recover qps up to float rounding.
        implied_qps = 1000 / result.latency_mean_ms
        assert implied_qps == pytest.approx(result.qps, rel=1e-6)

    def test_to_dict_has_expected_keys(self):
        corpus = make_corpus()
        chunks = make_chunks(corpus)
        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, ["does this run small"], k=1, n_warmup=0, n_repeats=1
        )
        row = result.to_dict()
        assert set(row) == {
            "retriever",
            "n_chunks",
            "build_seconds",
            "index_bytes",
            "peak_rss_mb",
            "peak_vram_mb",
            "device",
            "n_queries",
            "qps",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_mean_ms",
        }


class TestRunCostLatencyEvalBm25Only:
    """Exercises everything up to (not including) the dense arm."""

    def test_chunking_and_question_loading_wire_together(self, tmp_path):
        corpus = make_corpus()
        questions_path = tmp_path / "questions.jsonl"
        questions_path.write_text(
            '{"id": "q0", "type": "fit_sizing", "question": "does this run small?", '
            '"is_negative": false, "relevant_ids": ["0"]}\n',
            encoding="utf-8",
        )
        chunking_path = tmp_path / "chunking.yaml"
        chunking_path.write_text("chunking:\n  scheme: whole_review\n", encoding="utf-8")

        config = make_config(
            tmp_path,
            questions_in=str(questions_path),
            chunking_config_path=str(chunking_path),
        )

        from cragb.eval.cragb_questions import load_retrieval_questions

        questions = [q.question for q in load_retrieval_questions(config.questions_in)]
        assert questions == ["does this run small?"]

        chunks = make_chunks(corpus)
        result = measure_retriever_cost_latency(
            "bm25", BM25Retriever(), chunks, questions, k=1, n_warmup=0, n_repeats=1
        )
        assert result.n_queries == 1


@requires_dense
class TestRunCostLatencyEvalWithDense:
    def test_two_rows_bm25_and_dense(self, tmp_path):
        corpus = make_corpus()
        questions_path = tmp_path / "questions.jsonl"
        questions_path.write_text(
            "\n".join(
                f'{{"id": "q{i}", "type": "t", "question": "does this run small?", '
                f'"is_negative": false, "relevant_ids": ["0"]}}'
                for i in range(2)
            ),
            encoding="utf-8",
        )
        chunking_path = tmp_path / "chunking.yaml"
        chunking_path.write_text("chunking:\n  scheme: whole_review\n", encoding="utf-8")
        corpus_path = tmp_path / "corpus.parquet"
        corpus.to_parquet(corpus_path)

        config = make_config(
            tmp_path,
            corpus_in=str(corpus_path),
            questions_in=str(questions_path),
            chunking_config_path=str(chunking_path),
            n_warmup=0,
            n_repeats=1,
        )

        table = run_cost_latency_eval(corpus, config)

        assert set(table["retriever"]) == {"bm25", "dense"}
        assert len(table) == 2
        dense_row = table[table["retriever"] == "dense"].iloc[0]
        # bge-small-en-v1.5 is ~130M params; an 8-review index must stay far
        # below the model's own weight size, or index_bytes leaked the model.
        assert dense_row["index_bytes"] < 10_000_000
        assert dense_row["device"] in {"cuda", "cpu"}
