"""Unit tests for cragb.eval.cost_model (T5.4; M5.md T5.4).

`TestCostUsd`/`TestArmCostTable` cover the pure cost arithmetic against
hand-computed numbers, with no network or cache involved. `TestBuildMessagesForRow`/
`TestReissueTranscriptsForTokens` cover token recovery via a fake `usage_fn`
(mirroring this project's established `chat_fn`-injection testability pattern,
e.g. `cragb.generate.grounded_qa`), so no real `GroqClient`/disk cache is needed
either.

Covers M5.md T5.4's own validation checks: a hand-computed cost at a known rate;
`total_usd == mean_usd_per_query * n` within float tolerance; the closed-book arm
has materially fewer prompt tokens than the RAG arms; RAG-small and RAG-large have
near-identical prompt-token counts (the RQ1 control).
"""

from __future__ import annotations

from string import Template

import numpy as np
import pandas as pd
import pytest

from cragb.eval.cost_model import (
    ModelPricing,
    arm_cost_table,
    build_messages_for_row,
    cost_usd,
    load_pricing_config,
    reissue_transcripts_for_tokens,
)
from cragb.generate.api_clients import CompletionResult


def make_pricing(**overrides) -> dict[str, ModelPricing]:
    defaults = {
        "openai/gpt-oss-20b": ModelPricing(
            input_usd_per_1m=0.075, output_usd_per_1m=0.30, snapshot_date="2026-08-23", source_url="x"
        ),
        "openai/gpt-oss-120b": ModelPricing(
            input_usd_per_1m=0.15, output_usd_per_1m=0.60, snapshot_date="2026-08-23", source_url="x"
        ),
    }
    defaults.update(overrides)
    return defaults


class TestModelPricing:
    def test_valid_pricing_constructs(self):
        p = ModelPricing(input_usd_per_1m=0.1, output_usd_per_1m=0.5, snapshot_date="2026-08-23", source_url="x")
        assert p.input_usd_per_1m == 0.1

    def test_negative_input_rate_raises(self):
        with pytest.raises(ValueError, match="input_usd_per_1m"):
            ModelPricing(input_usd_per_1m=-0.1, output_usd_per_1m=0.5, snapshot_date="2026-08-23", source_url="x")

    def test_negative_output_rate_raises(self):
        with pytest.raises(ValueError, match="output_usd_per_1m"):
            ModelPricing(input_usd_per_1m=0.1, output_usd_per_1m=-0.5, snapshot_date="2026-08-23", source_url="x")


class TestLoadPricingConfig:
    def test_loads_the_committed_config(self):
        pricing = load_pricing_config("configs/pricing.yaml")
        assert "openai/gpt-oss-20b" in pricing
        assert "openai/gpt-oss-120b" in pricing
        assert "qwen/qwen3.6-27b" in pricing
        assert pricing["openai/gpt-oss-20b"].input_usd_per_1m > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pricing_config("configs/does_not_exist_v1.yaml")


class TestCostUsd:
    def test_hand_computed_cost_at_a_known_rate(self):
        pricing = make_pricing()
        # 1000 in @ $0.075/1M + 500 out @ $0.30/1M
        expected = (1000 / 1e6) * 0.075 + (500 / 1e6) * 0.30
        assert cost_usd(1000, 500, "openai/gpt-oss-20b", pricing) == pytest.approx(expected)

    def test_zero_tokens_costs_zero(self):
        pricing = make_pricing()
        assert cost_usd(0, 0, "openai/gpt-oss-20b", pricing) == 0.0

    def test_more_expensive_model_costs_more_for_same_tokens(self):
        pricing = make_pricing()
        small = cost_usd(1000, 500, "openai/gpt-oss-20b", pricing)
        large = cost_usd(1000, 500, "openai/gpt-oss-120b", pricing)
        assert large > small

    def test_unknown_model_raises_keyerror(self):
        pricing = make_pricing()
        with pytest.raises(KeyError, match="pricing.yaml"):
            cost_usd(100, 100, "unknown/model", pricing)


def make_call_rows(
    arm: str, model: str, n: int, prompt_tokens: int, completion_tokens: int, is_estimated: bool = False
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": [arm] * n,
            "question_id": [f"{arm}_{i}" for i in range(n)],
            "model": [model] * n,
            "prompt_tokens": [prompt_tokens] * n,
            "completion_tokens": [completion_tokens] * n,
            "is_estimated": [is_estimated] * n,
        }
    )


class TestArmCostTable:
    def test_empty_input_raises(self):
        pricing = make_pricing()
        with pytest.raises(ValueError, match="non-empty"):
            arm_cost_table(pd.DataFrame(columns=["arm", "model", "prompt_tokens", "completion_tokens", "is_estimated"]), pricing)

    def test_total_usd_matches_mean_times_n(self):
        pricing = make_pricing()
        rows = make_call_rows("rag_small", "openai/gpt-oss-20b", n=5, prompt_tokens=800, completion_tokens=100)
        table = arm_cost_table(rows, pricing, rng=np.random.default_rng(0))

        row = table.iloc[0]
        assert row["total_usd"] == pytest.approx(row["mean_usd_per_query"] * row["n_questions"])

    def test_one_row_per_arm_with_expected_columns(self):
        pricing = make_pricing()
        rows = pd.concat(
            [
                make_call_rows("closed_book", "openai/gpt-oss-20b", n=3, prompt_tokens=50, completion_tokens=80),
                make_call_rows("rag_small", "openai/gpt-oss-20b", n=3, prompt_tokens=800, completion_tokens=100),
                make_call_rows("rag_large", "openai/gpt-oss-120b", n=3, prompt_tokens=800, completion_tokens=100),
            ],
            ignore_index=True,
        )
        table = arm_cost_table(rows, pricing, rng=np.random.default_rng(0))

        assert set(table["arm"]) == {"closed_book", "rag_small", "rag_large"}
        assert len(table) == 3
        assert set(table.columns) == {
            "arm",
            "model",
            "n_questions",
            "mean_prompt_tokens",
            "mean_completion_tokens",
            "mean_usd_per_query",
            "usd_per_query_ci_lo",
            "usd_per_query_ci_hi",
            "total_usd",
            "is_estimated",
        }

    def test_closed_book_has_far_fewer_prompt_tokens_than_rag_arms(self):
        # RQ0's control: the only thing that should differ between closed-book and
        # RAG-small is presence of context. If closed-book's prompt tokens weren't
        # materially smaller, the arms got crossed somewhere upstream.
        pricing = make_pricing()
        rows = pd.concat(
            [
                make_call_rows("closed_book", "openai/gpt-oss-20b", n=10, prompt_tokens=40, completion_tokens=80),
                make_call_rows("rag_small", "openai/gpt-oss-20b", n=10, prompt_tokens=850, completion_tokens=100),
            ],
            ignore_index=True,
        )
        table = arm_cost_table(rows, pricing, rng=np.random.default_rng(0)).set_index("arm")
        assert table.loc["closed_book", "mean_prompt_tokens"] < table.loc["rag_small", "mean_prompt_tokens"] / 2

    def test_rag_small_and_rag_large_have_near_identical_prompt_tokens(self):
        # RQ1's control: same retriever/k/prompt, only the model differs.
        pricing = make_pricing()
        rows = pd.concat(
            [
                make_call_rows("rag_small", "openai/gpt-oss-20b", n=10, prompt_tokens=850, completion_tokens=100),
                make_call_rows("rag_large", "openai/gpt-oss-120b", n=10, prompt_tokens=852, completion_tokens=110),
            ],
            ignore_index=True,
        )
        table = arm_cost_table(rows, pricing, rng=np.random.default_rng(0)).set_index("arm")
        assert table.loc["rag_small", "mean_prompt_tokens"] == pytest.approx(
            table.loc["rag_large", "mean_prompt_tokens"], rel=0.01
        )

    def test_multiple_models_within_one_arm_raises(self):
        pricing = make_pricing()
        rows = pd.DataFrame(
            {
                "arm": ["rag_small", "rag_small"],
                "question_id": ["q0", "q1"],
                "model": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
                "prompt_tokens": [800, 800],
                "completion_tokens": [100, 100],
                "is_estimated": [False, False],
            }
        )
        with pytest.raises(ValueError, match="multiple distinct models"):
            arm_cost_table(rows, pricing)

    def test_is_estimated_true_if_any_row_in_arm_is_estimated(self):
        pricing = make_pricing()
        rows = pd.DataFrame(
            {
                "arm": ["rag_small", "rag_small"],
                "question_id": ["q0", "q1"],
                "model": ["openai/gpt-oss-20b", "openai/gpt-oss-20b"],
                "prompt_tokens": [800, 800],
                "completion_tokens": [100, 100],
                "is_estimated": [False, True],
            }
        )
        table = arm_cost_table(rows, pricing, rng=np.random.default_rng(0))
        assert bool(table.iloc[0]["is_estimated"]) is True

    def test_seeded_rng_gives_reproducible_ci(self):
        pricing = make_pricing()
        rows = pd.DataFrame(
            {
                "arm": ["rag_small"] * 6,
                "question_id": [f"q{i}" for i in range(6)],
                "model": ["openai/gpt-oss-20b"] * 6,
                "prompt_tokens": [700, 750, 800, 850, 900, 950],
                "completion_tokens": [80, 90, 100, 110, 120, 130],
                "is_estimated": [False] * 6,
            }
        )
        table_a = arm_cost_table(rows, pricing, rng=np.random.default_rng(7))
        table_b = arm_cost_table(rows, pricing, rng=np.random.default_rng(7))
        assert table_a["usd_per_query_ci_lo"].iloc[0] == table_b["usd_per_query_ci_lo"].iloc[0]


class FakeGroundedRow(dict):
    pass


def make_grounded_row(question_id: str, question: str, context_text: str) -> dict:
    return {"question_id": question_id, "question": question, "context_text": context_text}


def make_closed_book_row(question_id: str, question: str) -> dict:
    return {"question_id": question_id, "question": question}


GROUNDED_TEMPLATE = Template("Q: $question\nContext:\n$context_block")
CLOSED_BOOK_TEMPLATE = Template("Q: $question (no context)")


class TestBuildMessagesForRow:
    def test_row_with_context_text_uses_grounded_render(self):
        row = make_grounded_row("q0", "Does this run small?", "[1] runs small\n[2] true to size")
        messages = build_messages_for_row(row, GROUNDED_TEMPLATE)
        assert messages == [
            {
                "role": "user",
                "content": "Q: Does this run small?\nContext:\n[1] runs small\n[2] true to size",
            }
        ]

    def test_row_without_context_text_uses_closed_book_render(self):
        row = make_closed_book_row("q0", "Does this run small?")
        messages = build_messages_for_row(row, CLOSED_BOOK_TEMPLATE)
        assert messages == [{"role": "user", "content": "Q: Does this run small? (no context)"}]


class TestReissueTranscriptsForTokens:
    def test_measured_usage_is_used_when_available(self):
        rows = [make_grounded_row("q0", "Q?", "ctx")]

        def usage_fn(messages):
            return CompletionResult(
                text="An answer.",
                prompt_tokens=42,
                completion_tokens=7,
                latency_s=None,
                cached=True,
                model="openai/gpt-oss-20b",
            )

        result = reissue_transcripts_for_tokens("rag_small", rows, GROUNDED_TEMPLATE, usage_fn)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["prompt_tokens"] == 42
        assert row["completion_tokens"] == 7
        assert row["is_estimated"] == False  # noqa: E712 (explicit bool check on a pandas scalar)
        assert row["arm"] == "rag_small"
        assert row["question_id"] == "q0"
        assert row["model"] == "openai/gpt-oss-20b"

    def test_missing_usage_falls_back_to_char_estimate(self):
        rows = [make_grounded_row("q0", "Q?", "ctx")]

        def usage_fn(messages):
            return CompletionResult(
                text="12345678",  # 8 chars -> 2 tokens at 4 chars/token
                prompt_tokens=None,
                completion_tokens=None,
                latency_s=None,
                cached=True,
                model="openai/gpt-oss-20b",
            )

        result = reissue_transcripts_for_tokens("rag_small", rows, GROUNDED_TEMPLATE, usage_fn)

        row = result.iloc[0]
        assert row["is_estimated"] == True  # noqa: E712
        assert row["completion_tokens"] == 2
        # prompt = "Q: Q?\nContext:\nctx" (18 chars) -> round(18/4) = 4
        assert row["prompt_tokens"] == 4

    def test_estimate_never_reports_zero_tokens(self):
        rows = [make_closed_book_row("q0", "")]

        def usage_fn(messages):
            return CompletionResult(
                text="", prompt_tokens=None, completion_tokens=None, latency_s=None, cached=False, model="m"
            )

        result = reissue_transcripts_for_tokens("closed_book", rows, CLOSED_BOOK_TEMPLATE, usage_fn)
        assert result.iloc[0]["completion_tokens"] >= 1
        assert result.iloc[0]["prompt_tokens"] >= 1

    def test_processes_every_row_in_order(self):
        rows = [make_grounded_row(f"q{i}", f"Question {i}?", f"ctx{i}") for i in range(4)]
        calls = []

        def usage_fn(messages):
            calls.append(messages[0]["content"])
            return CompletionResult(
                text="ok", prompt_tokens=10, completion_tokens=2, latency_s=None, cached=True, model="m"
            )

        result = reissue_transcripts_for_tokens("rag_small", rows, GROUNDED_TEMPLATE, usage_fn)
        assert list(result["question_id"]) == ["q0", "q1", "q2", "q3"]
        assert len(calls) == 4
