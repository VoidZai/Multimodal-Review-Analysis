"""Unit tests for cragb.data.clean.

Each filter is tested in isolation on small synthetic DataFrames covering
the edge cases that matter for a filter that will run over tens of
millions of real rows: exact boundaries (min_chars), whitespace-only vs.
truly-empty text, NaN handling, and detector failure on garbage input.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cragb.data.clean import dedup, drop_empty_text, filter_language, normalize_whitespace


class TestNormalizeWhitespace:
    def test_collapses_internal_whitespace(self):
        df = pd.DataFrame({"text": ["a   b\n\nc\td"]})
        out = normalize_whitespace(df, columns=["text"])
        assert out["text"].iloc[0] == "a b c d"

    def test_strips_leading_and_trailing(self):
        df = pd.DataFrame({"text": ["   padded   "]})
        out = normalize_whitespace(df, columns=["text"])
        assert out["text"].iloc[0] == "padded"

    def test_leaves_non_string_values_untouched(self):
        df = pd.DataFrame({"text": [None, 42, "  ok  "]})
        out = normalize_whitespace(df, columns=["text"])
        assert out["text"].iloc[0] is None
        assert out["text"].iloc[1] == 42
        assert out["text"].iloc[2] == "ok"

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"text": ["  a  "]})
        normalize_whitespace(df, columns=["text"])
        assert df["text"].iloc[0] == "  a  "

    def test_only_touches_named_columns(self):
        df = pd.DataFrame({"text": ["  a  "], "title": ["  b  "]})
        out = normalize_whitespace(df, columns=["text"])
        assert out["text"].iloc[0] == "a"
        assert out["title"].iloc[0] == "  b  "


class TestDropEmptyText:
    def test_drops_below_threshold(self):
        df = pd.DataFrame({"text": ["ab", "abcdef"]})
        out = drop_empty_text(df, text_col="text", min_chars=5)
        assert list(out["text"]) == ["abcdef"]

    def test_boundary_is_inclusive(self):
        df = pd.DataFrame({"text": ["abcde", "abcd"]})
        out = drop_empty_text(df, text_col="text", min_chars=5)
        assert list(out["text"]) == ["abcde"]

    def test_whitespace_only_text_is_dropped(self):
        df = pd.DataFrame({"text": ["     ", "real content"]})
        out = drop_empty_text(df, text_col="text", min_chars=1)
        assert list(out["text"]) == ["real content"]

    def test_nan_is_dropped(self):
        df = pd.DataFrame({"text": [float("nan"), "keep me"]})
        out = drop_empty_text(df, text_col="text", min_chars=1)
        assert list(out["text"]) == ["keep me"]

    def test_preserves_original_index(self):
        df = pd.DataFrame({"text": ["short", "long enough text"]}, index=[10, 20])
        out = drop_empty_text(df, text_col="text", min_chars=10)
        assert list(out.index) == [20]


class TestDedup:
    def test_drops_exact_duplicates_on_subset(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "parent_asin": ["p1", "p1", "p1"],
                "text": ["same", "same", "same"],
            }
        )
        out = dedup(df, subset=["user_id", "parent_asin", "text"])
        assert len(out) == 2

    def test_keeps_first_occurrence(self):
        df = pd.DataFrame({"k": ["a", "a"], "v": [1, 2]})
        out = dedup(df, subset=["k"])
        assert list(out["v"]) == [1]

    def test_differing_subset_column_is_not_a_duplicate(self):
        df = pd.DataFrame({"user_id": ["u1", "u1"], "text": ["same", "same"], "parent_asin": ["p1", "p2"]})
        out = dedup(df, subset=["user_id", "parent_asin", "text"])
        assert len(out) == 2

    def test_no_duplicates_is_a_no_op(self):
        df = pd.DataFrame({"k": ["a", "b", "c"]})
        out = dedup(df, subset=["k"])
        assert len(out) == 3


class TestFilterLanguage:
    def test_keeps_english(self):
        df = pd.DataFrame(
            {"text": ["This sweater fits true to size and the fabric feels great for the price."]}
        )
        out = filter_language(df, text_col="text", keep_languages=["en"])
        assert len(out) == 1

    def test_drops_non_english(self):
        df = pd.DataFrame(
            {
                "text": [
                    "Este vestido es hermoso y la calidad de la tela es excelente para el precio que pagué.",
                ]
            }
        )
        out = filter_language(df, text_col="text", keep_languages=["en"])
        assert len(out) == 0

    def test_drops_undetectable_garbage(self):
        # Too short/symbolic for langdetect to classify -> raises
        # LangDetectException internally; must be dropped, not kept.
        df = pd.DataFrame({"text": ["!!!", "123", ""]})
        out = filter_language(df, text_col="text", keep_languages=["en"])
        assert len(out) == 0

    def test_drops_nan(self):
        df = pd.DataFrame({"text": [float("nan")]})
        out = filter_language(df, text_col="text", keep_languages=["en"])
        assert len(out) == 0

    def test_mixed_batch_keeps_only_matching_language(self):
        df = pd.DataFrame(
            {
                "text": [
                    "The shoes run about half a size small so I would size up if you are between sizes.",
                    "La qualité de ce tissu est vraiment décevante et je ne le recommande pas du tout.",
                ]
            }
        )
        out = filter_language(df, text_col="text", keep_languages=["en"])
        assert len(out) == 1
        assert out["text"].iloc[0].startswith("The shoes")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
