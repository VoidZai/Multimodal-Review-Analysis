"""Unit tests for cragb.retrieval.chunking (T3.1; M3.md T3.1).

Covers: each scheme's own invariants (unique chunk ids, every chunk
traces to a real parent, no empty-text chunks, the fixed-token cap is
respected, whole-review count == corpus row count on a corpus with no
blank rows), `ChunkingConfig`'s validation, `load_chunking_config`
against the real `configs/chunking.yaml`, and `chunk_corpus`'s dispatch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.retrieval.chunking import (
    ALLOWED_SCHEMES,
    ChunkingConfig,
    chunk_corpus,
    chunk_fixed_token,
    chunk_sentence_window,
    chunk_whole_review,
    load_chunking_config,
)


def make_corpus(rows: list[str] | None = None) -> pd.DataFrame:
    rows = rows if rows is not None else [
        "This runs small. Size up if you are between sizes.",
        "Great fabric, held up after a dozen washes.",
        "Colour matched the photo exactly, very happy with it.",
    ]
    return pd.DataFrame({"text": rows})


class TestChunkWholeReview:
    def test_one_chunk_per_review(self):
        corpus = make_corpus()
        chunks = chunk_whole_review(corpus)
        assert len(chunks) == len(corpus)

    def test_chunk_id_equals_parent_doc_id(self):
        corpus = make_corpus()
        chunks = chunk_whole_review(corpus)
        assert (chunks["chunk_id"] == chunks["parent_doc_id"]).all()

    def test_chunk_ids_unique(self):
        chunks = chunk_whole_review(make_corpus())
        assert chunks["chunk_id"].is_unique

    def test_no_empty_text_chunks(self):
        chunks = chunk_whole_review(make_corpus())
        assert (chunks["text"].str.len() > 0).all()

    def test_default_id_uses_dataframe_index(self):
        corpus = make_corpus().set_index(pd.Index([10, 20, 30]))
        chunks = chunk_whole_review(corpus)
        assert set(chunks["parent_doc_id"]) == {"10", "20", "30"}

    def test_custom_id_col_used_as_parent_id(self):
        corpus = make_corpus()
        corpus["review_id"] = ["r1", "r2", "r3"]
        chunks = chunk_whole_review(corpus, id_col="review_id")
        assert set(chunks["parent_doc_id"]) == {"r1", "r2", "r3"}

    def test_blank_rows_are_dropped_not_emitted_empty(self):
        corpus = make_corpus(["real text here", "   ", ""])
        chunks = chunk_whole_review(corpus)
        assert len(chunks) == 1

    def test_missing_text_treated_as_empty_and_dropped(self):
        corpus = pd.DataFrame({"text": ["real text here", None]})
        chunks = chunk_whole_review(corpus)
        assert len(chunks) == 1

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_whole_review(pd.DataFrame({"text": []}))

    def test_all_blank_corpus_raises(self):
        with pytest.raises(ValueError, match="No non-empty text"):
            chunk_whole_review(make_corpus(["", "   "]))

    def test_duplicate_parent_ids_raise(self):
        corpus = make_corpus()
        corpus["review_id"] = ["r1", "r1", "r2"]
        with pytest.raises(ValueError, match="not unique"):
            chunk_whole_review(corpus, id_col="review_id")


class TestChunkFixedToken:
    def test_respects_token_cap(self):
        long_review = " ".join(f"word{i}" for i in range(300))
        corpus = make_corpus([long_review])
        chunks = chunk_fixed_token(corpus, token_size=128)
        for text in chunks["text"]:
            assert len(text.split()) <= 128

    def test_short_review_produces_one_chunk(self):
        corpus = make_corpus(["only five short words here"])
        chunks = chunk_fixed_token(corpus, token_size=256)
        assert len(chunks) == 1

    def test_long_review_splits_into_multiple_chunks(self):
        long_review = " ".join(f"word{i}" for i in range(300))
        corpus = make_corpus([long_review])
        chunks = chunk_fixed_token(corpus, token_size=128)
        assert len(chunks) == 3  # ceil(300 / 128)

    def test_every_chunk_traces_to_a_real_parent(self):
        corpus = make_corpus()
        corpus["review_id"] = ["r1", "r2", "r3"]
        chunks = chunk_fixed_token(corpus, id_col="review_id", token_size=4)
        assert set(chunks["parent_doc_id"]).issubset({"r1", "r2", "r3"})

    def test_chunk_ids_unique_across_multi_chunk_reviews(self):
        long_review = " ".join(f"word{i}" for i in range(300))
        corpus = make_corpus([long_review, long_review])
        chunks = chunk_fixed_token(corpus, token_size=100)
        assert chunks["chunk_id"].is_unique

    def test_chunk_id_format_is_parent_double_colon_index(self):
        long_review = " ".join(f"word{i}" for i in range(300))
        corpus = make_corpus([long_review]).set_index(pd.Index(["r1"]))
        chunks = chunk_fixed_token(corpus, token_size=128)
        assert list(chunks["chunk_id"]) == ["r1::0", "r1::1", "r1::2"]

    def test_no_empty_text_chunks(self):
        corpus = make_corpus()
        chunks = chunk_fixed_token(corpus, token_size=4)
        assert (chunks["text"].str.len() > 0).all()

    def test_non_positive_token_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            chunk_fixed_token(make_corpus(), token_size=0)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_fixed_token(pd.DataFrame({"text": []}), token_size=128)


class TestChunkSentenceWindow:
    def test_short_review_produces_one_chunk(self):
        corpus = make_corpus(["One sentence only."])
        chunks = chunk_sentence_window(corpus, window_size=3)
        assert len(chunks) == 1

    def test_review_with_more_sentences_than_window_splits(self):
        review = "One. Two. Three. Four. Five. Six. Seven."
        corpus = make_corpus([review])
        chunks = chunk_sentence_window(corpus, window_size=3)
        assert len(chunks) == 3  # ceil(7 / 3)

    def test_window_never_exceeds_configured_sentence_count(self):
        review = "One. Two. Three. Four. Five."
        corpus = make_corpus([review])
        chunks = chunk_sentence_window(corpus, window_size=2)
        for text in chunks["text"]:
            # count sentence-ending punctuation as a proxy for sentence count
            assert text.count(".") <= 2

    def test_every_chunk_traces_to_a_real_parent(self):
        corpus = make_corpus()
        corpus["review_id"] = ["r1", "r2", "r3"]
        chunks = chunk_sentence_window(corpus, id_col="review_id", window_size=1)
        assert set(chunks["parent_doc_id"]).issubset({"r1", "r2", "r3"})

    def test_chunk_ids_unique(self):
        review = "One. Two. Three. Four. Five. Six. Seven."
        corpus = make_corpus([review, review])
        chunks = chunk_sentence_window(corpus, window_size=2)
        assert chunks["chunk_id"].is_unique

    def test_no_empty_text_chunks(self):
        chunks = chunk_sentence_window(make_corpus(), window_size=1)
        assert (chunks["text"].str.len() > 0).all()

    def test_text_with_no_terminal_punctuation_is_one_sentence(self):
        corpus = make_corpus(["no terminal punctuation here just words"])
        chunks = chunk_sentence_window(corpus, window_size=3)
        assert len(chunks) == 1

    def test_non_positive_window_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            chunk_sentence_window(make_corpus(), window_size=0)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_sentence_window(pd.DataFrame({"text": []}), window_size=3)


class TestChunkingConfig:
    def test_valid_config_constructs(self):
        cfg = ChunkingConfig(scheme="whole_review")
        assert cfg.scheme == "whole_review"
        assert cfg.fixed_token_size == 256
        assert cfg.sentence_window_size == 3

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking scheme"):
            ChunkingConfig(scheme="not_a_real_scheme")

    @pytest.mark.parametrize("size", [0, -1])
    def test_non_positive_fixed_token_size_raises(self, size):
        with pytest.raises(ValueError, match="fixed_token_size"):
            ChunkingConfig(scheme="fixed_token", fixed_token_size=size)

    @pytest.mark.parametrize("size", [0, -1])
    def test_non_positive_sentence_window_size_raises(self, size):
        with pytest.raises(ValueError, match="sentence_window_size"):
            ChunkingConfig(scheme="sentence_window", sentence_window_size=size)


class TestLoadChunkingConfig:
    def test_real_config_loads(self):
        cfg = load_chunking_config("configs/chunking.yaml")
        assert cfg.scheme in ALLOWED_SCHEMES
        assert cfg.fixed_token_size > 0
        assert cfg.sentence_window_size > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_chunking_config("configs/does_not_exist.yaml")

    def test_missing_chunking_block_raises(self, tmp_path):
        bad_config = tmp_path / "bad_chunking.yaml"
        bad_config.write_text("seed: 42\npaths:\n  corpus_in: x\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_chunking_config(bad_config)


class TestChunkCorpus:
    def test_dispatches_to_whole_review(self):
        corpus = make_corpus()
        result = chunk_corpus(corpus, ChunkingConfig(scheme="whole_review"))
        pd.testing.assert_frame_equal(result, chunk_whole_review(corpus))

    def test_dispatches_to_fixed_token(self):
        corpus = make_corpus()
        cfg = ChunkingConfig(scheme="fixed_token", fixed_token_size=4)
        result = chunk_corpus(corpus, cfg)
        pd.testing.assert_frame_equal(result, chunk_fixed_token(corpus, token_size=4))

    def test_dispatches_to_sentence_window(self):
        corpus = make_corpus()
        cfg = ChunkingConfig(scheme="sentence_window", sentence_window_size=2)
        result = chunk_corpus(corpus, cfg)
        pd.testing.assert_frame_equal(result, chunk_sentence_window(corpus, window_size=2))

    def test_output_schema_is_always_the_same_three_columns(self):
        corpus = make_corpus()
        for scheme_kwargs in (
            {"scheme": "whole_review"},
            {"scheme": "fixed_token", "fixed_token_size": 4},
            {"scheme": "sentence_window", "sentence_window_size": 2},
        ):
            result = chunk_corpus(corpus, ChunkingConfig(**scheme_kwargs))
            assert list(result.columns) == ["chunk_id", "parent_doc_id", "text"]
