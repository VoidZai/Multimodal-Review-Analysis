"""Unit tests for cragb.eval.run_multimodal_pilot (T6.5; M6.md T6.5).

Zero network access anywhere in this file: `run_pilot`/`judge_pair` take an
injected `chat_fn`, and `summarize_cost` takes a plain list of
`CompletionResult`s built by hand -- no `PhotoStore` network calls, no
`GeminiClient` construction. The one place real photo bytes are needed
(`PhotoStore.to_data_part`, inside `judge_pair`), a real `PhotoStore` is
fetched once via a monkeypatched `_session.get` (the same fake-transport
pattern `test_photo_store.py`/`test_photo_link.py`/`test_vision_judge.py`
all use).

Covers, per M6.md T6.5's validation checks: win-rate on a hand-built
fixture matches the hand-computed value; ties are excluded from the
numerator but included in the denominator; the bootstrap CI brackets the
point estimate; an all-tie fixture yields `win_rate=0.0` and a p-value that
is *not* significant (one-sided, `alternative="greater"`); `n_pairs` in the
win-rate table equals the row count of the written verdicts file equals the
number of input pairs (T6.3's funnel invariant, carried through T6.5); and
the cost table's total matches a hand computation that includes Gemini's
hidden thinking tokens in the billed output count.
"""

from __future__ import annotations

import io
import json
from string import Template

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cragb.eval.run_multimodal_pilot import (
    UsageRecorder,
    load_pairs,
    run_pilot,
    summarize_cost,
    summarize_winrate,
    write_verdicts_jsonl,
)
from cragb.generate.api_clients import CompletionResult
from cragb.eval.cost_model import ModelPricing
from cragb.multimodal.photo_store import PhotoStore

TEMPLATE = Template(
    "Question: $question\n\nPhoto A:\n[[PHOTO_A]]\n\nPhoto B:\n[[PHOTO_B]]\n\nRespond with JSON."
)


def _jpeg_bytes(color: str = "teal") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), color=color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def make_store(tmp_path, monkeypatch) -> PhotoStore:
    store = PhotoStore(photos_dir=str(tmp_path / "photos"), request_delay_s=0.0)
    monkeypatch.setattr(store._session, "get", lambda url, **kw: FakeResponse(_jpeg_bytes()))
    return store


def make_pairs_df(n: int) -> pd.DataFrame:
    types = ["fit_sizing", "colour_appearance"]
    return pd.DataFrame(
        [
            {
                "question_id": f"q{i}",
                "type": types[i % len(types)],
                "question": f"question {i}?",
                "surfaced_photo_id": None,  # resolved on demand by the store's fetch
                "surfaced_doc_id": str(i),
                "control_photo_id": None,
                "control_doc_id": str(100 + i),
            }
            for i in range(n)
        ]
    )


def make_verdicts_df(outcomes: list[str], order_agreement: list[bool] | None = None, types: list[str] | None = None) -> pd.DataFrame:
    n = len(outcomes)
    order_agreement = order_agreement if order_agreement is not None else [o != "tie" for o in outcomes]
    types = types if types is not None else ["fit_sizing"] * n
    return pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(n)],
            "type": types,
            "outcome": outcomes,
            "order_agreement": order_agreement,
            "winner_surfaced_as_a": ["A"] * n,
            "confidence_surfaced_as_a": [3] * n,
            "rationale_surfaced_as_a": ["x"] * n,
            "winner_surfaced_as_b": ["B"] * n,
            "confidence_surfaced_as_b": [3] * n,
            "rationale_surfaced_as_b": ["y"] * n,
        }
    )


# --------------------------------------------------------------------------
# load_pairs / write_verdicts_jsonl
# --------------------------------------------------------------------------


class TestLoadPairs:
    def test_loads_rows_in_order(self, tmp_path):
        path = tmp_path / "mm_pairs_v1.jsonl"
        rows = [
            {"question_id": "q1", "type": "fit_sizing", "question": "q1?", "surfaced_photo_id": "s1", "control_photo_id": "c1"},
            {"question_id": "q2", "type": "colour_appearance", "question": "q2?", "surfaced_photo_id": "s2", "control_photo_id": "c2"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        df = load_pairs(path)

        assert list(df["question_id"]) == ["q1", "q2"]
        assert list(df["surfaced_photo_id"]) == ["s1", "s2"]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "mm_pairs_v1.jsonl"
        path.write_text("\n" + json.dumps({"question_id": "q1", "type": "t", "question": "?", "surfaced_photo_id": "s", "control_photo_id": "c"}) + "\n\n", encoding="utf-8")
        assert len(load_pairs(path)) == 1


class TestWriteVerdictsJsonl:
    def test_round_trips(self, tmp_path):
        verdicts = make_verdicts_df(["surfaced_win", "tie"])
        out_path = tmp_path / "mm_verdicts_v1.jsonl"

        write_verdicts_jsonl(verdicts, out_path)

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["outcome"] == "surfaced_win"
        assert json.loads(lines[1])["outcome"] == "tie"


# --------------------------------------------------------------------------
# UsageRecorder
# --------------------------------------------------------------------------


class TestUsageRecorder:
    def test_records_calls_and_returns_text(self):
        def fake_usage_fn(parts):
            return CompletionResult(
                text="hello", prompt_tokens=10, completion_tokens=5, latency_s=0.1, cached=False, model="m"
            )

        recorder = UsageRecorder(usage_fn=fake_usage_fn)
        text = recorder([{"type": "text", "text": "hi"}])

        assert text == "hello"
        assert len(recorder.calls) == 1
        assert recorder.calls[0].prompt_tokens == 10


# --------------------------------------------------------------------------
# run_pilot: n_pairs invariant chain (T6.3's funnel -> mm_verdicts_v1.jsonl)
# --------------------------------------------------------------------------


class TestRunPilot:
    def test_one_verdict_row_per_input_pair_same_order(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        # Pre-cache two distinct photos per pair so surfaced/control ids exist.
        pairs = []
        for i in range(4):
            surfaced = store.fetch_photo(f"https://x/surfaced-{i}.jpg")
            control = store.fetch_photo(f"https://x/control-{i}.jpg")
            pairs.append(
                {
                    "question_id": f"q{i}",
                    "type": "fit_sizing" if i % 2 == 0 else "colour_appearance",
                    "question": f"question {i}?",
                    "surfaced_photo_id": surfaced.photo_id,
                    "control_photo_id": control.photo_id,
                }
            )
        pairs_df = pd.DataFrame(pairs)

        chat_fn = lambda parts: '{"winner": "A", "confidence": 4, "rationale": "ok"}'  # noqa: E731

        verdicts = run_pilot(pairs_df, store, TEMPLATE, chat_fn)

        assert len(verdicts) == len(pairs_df) == 4
        assert list(verdicts["question_id"]) == list(pairs_df["question_id"])
        assert set(verdicts["outcome"]) <= {"surfaced_win", "control_win", "tie"}


# --------------------------------------------------------------------------
# summarize_winrate
# --------------------------------------------------------------------------


class TestSummarizeWinrate:
    def test_matches_hand_computed_winrate(self):
        # 2 surfaced wins, 1 control win, 1 tie -> win_rate = 2/4 = 0.5.
        verdicts = make_verdicts_df(["surfaced_win", "surfaced_win", "control_win", "tie"])

        table = summarize_winrate(verdicts, rng=np.random.default_rng(0))
        overall = table[table["group"] == "overall"].iloc[0]

        assert overall["n_pairs"] == 4
        assert overall["n_surfaced_win"] == 2
        assert overall["n_control_win"] == 1
        assert overall["n_tie"] == 1
        assert overall["win_rate"] == pytest.approx(0.5)
        assert overall["tie_rate"] == pytest.approx(0.25)

    def test_ties_excluded_from_numerator_included_in_denominator(self):
        # 1 surfaced win, 3 ties -> win_rate = 1/4, NOT 1/1 or 1/3.
        verdicts = make_verdicts_df(["surfaced_win", "tie", "tie", "tie"])

        table = summarize_winrate(verdicts, rng=np.random.default_rng(0))
        overall = table[table["group"] == "overall"].iloc[0]

        assert overall["n_pairs"] == 4
        assert overall["win_rate"] == pytest.approx(0.25)

    def test_ci_brackets_the_point_estimate(self):
        verdicts = make_verdicts_df(["surfaced_win", "surfaced_win", "control_win", "tie", "surfaced_win"])
        table = summarize_winrate(verdicts, rng=np.random.default_rng(1))
        for _, row in table.iterrows():
            assert row["ci_lo"] <= row["win_rate"] <= row["ci_hi"]

    def test_all_tie_yields_zero_winrate_and_not_significant(self):
        verdicts = make_verdicts_df(["tie", "tie", "tie", "tie", "tie"], order_agreement=[False] * 5)

        table = summarize_winrate(verdicts, rng=np.random.default_rng(0))
        overall = table[table["group"] == "overall"].iloc[0]

        assert overall["win_rate"] == 0.0
        assert overall["ci_lo"] == 0.0 and overall["ci_hi"] == 0.0
        # One-sided (alternative="greater"): 0 wins is strong evidence
        # AGAINST winning more than chance, not evidence for it -- p must
        # be large (not significant), never small.
        assert overall["p_value_vs_0.5_greater"] > 0.5

    def test_per_type_breakdown_present_and_correctly_grouped(self):
        verdicts = make_verdicts_df(
            ["surfaced_win", "control_win", "surfaced_win", "tie"],
            types=["fit_sizing", "fit_sizing", "colour_appearance", "colour_appearance"],
        )

        table = summarize_winrate(verdicts, rng=np.random.default_rng(0))

        groups = set(table["group"])
        assert groups == {"overall", "fit_sizing", "colour_appearance"}
        fit_row = table[table["group"] == "fit_sizing"].iloc[0]
        assert fit_row["n_pairs"] == 2
        assert fit_row["n_surfaced_win"] == 1
        colour_row = table[table["group"] == "colour_appearance"].iloc[0]
        assert colour_row["n_pairs"] == 2
        assert colour_row["n_surfaced_win"] == 1

    def test_raises_on_empty_verdicts(self):
        empty = make_verdicts_df([])
        with pytest.raises(ValueError, match="non-empty"):
            summarize_winrate(empty)

    def test_same_seed_reproduces_identical_ci(self):
        verdicts = make_verdicts_df(["surfaced_win", "control_win", "tie", "surfaced_win", "surfaced_win"])
        first = summarize_winrate(verdicts, rng=np.random.default_rng(7))
        second = summarize_winrate(verdicts, rng=np.random.default_rng(7))
        pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------
# summarize_cost
# --------------------------------------------------------------------------


class TestSummarizeCost:
    def _pricing(self) -> dict[str, ModelPricing]:
        return {
            "gemini-3.6-flash": ModelPricing(
                input_usd_per_1m=0.75, output_usd_per_1m=3.75, snapshot_date="2026-08-24", source_url="https://x"
            )
        }

    def test_matches_hand_computed_total_including_thinking_tokens(self):
        calls = [
            CompletionResult(
                text="a", prompt_tokens=1000, completion_tokens=50, latency_s=1.0, cached=False,
                model="gemini-3.6-flash", thinking_tokens=200,
            ),
            CompletionResult(
                text="b", prompt_tokens=2000, completion_tokens=30, latency_s=1.0, cached=False,
                model="gemini-3.6-flash", thinking_tokens=100,
            ),
        ]
        pricing = self._pricing()

        table = summarize_cost(calls, "gemini-3.6-flash", pricing, wall_clock_s=10.0)
        row = table.iloc[0]

        # Hand computation: cost = prompt/1e6*0.75 + (completion+thinking)/1e6*3.75
        expected_1 = (1000 / 1e6) * 0.75 + (250 / 1e6) * 3.75
        expected_2 = (2000 / 1e6) * 0.75 + (130 / 1e6) * 3.75
        assert row["total_usd"] == pytest.approx(expected_1 + expected_2)
        assert row["n_calls"] == 2
        assert row["n_pairs"] == 1
        assert row["total_prompt_tokens"] == 3000
        assert row["total_completion_tokens"] == 80
        assert row["total_thinking_tokens"] == 300
        assert row["wall_clock_s"] == 10.0
        assert row["calls_per_second"] == pytest.approx(0.2)

    def test_missing_thinking_tokens_treated_as_zero(self):
        calls = [
            CompletionResult(
                text="a", prompt_tokens=100, completion_tokens=10, latency_s=1.0, cached=False,
                model="gemini-3.6-flash", thinking_tokens=None,
            )
        ]
        table = summarize_cost(calls, "gemini-3.6-flash", self._pricing(), wall_clock_s=1.0)
        expected = (100 / 1e6) * 0.75 + (10 / 1e6) * 3.75
        assert table.iloc[0]["total_usd"] == pytest.approx(expected)

    def test_raises_on_empty_calls(self):
        with pytest.raises(ValueError, match="non-empty"):
            summarize_cost([], "gemini-3.6-flash", self._pricing(), wall_clock_s=1.0)

    def test_zero_wall_clock_gives_nan_calls_per_second(self):
        calls = [
            CompletionResult(
                text="a", prompt_tokens=1, completion_tokens=1, latency_s=0.0, cached=True,
                model="gemini-3.6-flash", thinking_tokens=None,
            )
        ]
        table = summarize_cost(calls, "gemini-3.6-flash", self._pricing(), wall_clock_s=0.0)
        assert np.isnan(table.iloc[0]["calls_per_second"])


# --------------------------------------------------------------------------
# End-to-end invariant: n_pairs (winrate CSV) == len(verdicts jsonl) == input pairs
# --------------------------------------------------------------------------


class TestEndToEndInvariant:
    def test_n_pairs_consistent_across_pairs_verdicts_and_winrate(self, tmp_path, monkeypatch):
        store = make_store(tmp_path, monkeypatch)
        n = 6
        pairs = []
        for i in range(n):
            surfaced = store.fetch_photo(f"https://x/s-{i}.jpg")
            control = store.fetch_photo(f"https://x/c-{i}.jpg")
            pairs.append(
                {
                    "question_id": f"q{i}",
                    "type": "fit_sizing",
                    "question": f"question {i}?",
                    "surfaced_photo_id": surfaced.photo_id,
                    "control_photo_id": control.photo_id,
                }
            )
        pairs_df = pd.DataFrame(pairs)
        # A funnel value T6.3 would have printed for this synthetic set --
        # in a real run this comes from mm_coverage_v1.csv's usable_pairs stage.
        t63_usable_pairs_count = n

        chat_fn = lambda parts: '{"winner": "tie", "confidence": 2, "rationale": "ok"}'  # noqa: E731
        verdicts = run_pilot(pairs_df, store, TEMPLATE, chat_fn)

        verdicts_path = tmp_path / "mm_verdicts_v1.jsonl"
        write_verdicts_jsonl(verdicts, verdicts_path)
        n_verdict_rows = len(verdicts_path.read_text(encoding="utf-8").splitlines())

        winrate = summarize_winrate(verdicts, rng=np.random.default_rng(0))
        overall_n_pairs = int(winrate[winrate["group"] == "overall"].iloc[0]["n_pairs"])

        assert overall_n_pairs == n_verdict_rows == t63_usable_pairs_count == len(pairs_df)
