"""Unit tests for cragb.finetune.local_client (T7.7; PLAN.md §10, M7.md T7.7).

No real GPU load or model download here (that's a manual verification step, per M7.md
T7.7's own "How I verify it worked" -- interface tests only). `LocalHFClient._run_generation`
is monkeypatched directly, mirroring `tests/test_api_clients.py`'s pattern of stubbing
`GroqClient._session.post` -- both replace the one method that would otherwise touch a
real resource (network / GPU), leaving every other method's real logic (caching, usage
telemetry, call logging) exercised for real.

Covers: `complete`/`complete_with_usage`'s cache-hit/miss behaviour and `bypass_cache`
semantics (mirroring `GroqClient`'s own test suite closely, since that parity is the
entire point of this client existing), that `LocalHFClient` and `GroqClient` are
interchangeable through a shared harness function (the validation check M7.md T7.7 names
explicitly), and `probe_model`/`select_base_model`'s VRAM-budget classification logic
(via a monkeypatched `torch` proxy, no CUDA required to run these).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cragb.eval.cost_model import UsageFn
from cragb.finetune.local_client import (
    DEFAULT_CANDIDATE_MODELS,
    LocalHFClient,
    ModelProbeResult,
    probe_models,
    select_base_model,
)
from cragb.generate.api_clients import CompletionResult, GroqClient


def make_local_client(tmp_path, **overrides) -> LocalHFClient:
    kwargs = dict(
        model="fake/local-model",
        cache_dir=str(tmp_path / "cache"),
        call_log_path=str(tmp_path / "api_calls_v1.jsonl"),
    )
    kwargs.update(overrides)
    return LocalHFClient(**kwargs)


def stub_generation(client: LocalHFClient, responses: list[tuple[str, int, int]]):
    """Replace `client._run_generation` with a stub returning `responses` in call order."""
    calls: list[list[dict]] = []

    def fake_run_generation(messages, max_new_tokens=None, min_new_tokens=None):
        calls.append(messages)
        return responses[len(calls) - 1]

    client._run_generation = fake_run_generation
    return calls


# --------------------------------------------------------------------------
# complete()
# --------------------------------------------------------------------------


class TestComplete:
    def test_returns_generated_text(self, tmp_path):
        client = make_local_client(tmp_path)
        stub_generation(client, [("The answer.", 10, 5)])
        assert client.complete([{"role": "user", "content": "hi"}]) == "The answer."

    def test_second_identical_call_is_served_from_cache(self, tmp_path):
        client = make_local_client(tmp_path)
        calls = stub_generation(client, [("first", 10, 5), ("second", 10, 5)])
        first = client.complete([{"role": "user", "content": "hi"}])
        second = client.complete([{"role": "user", "content": "hi"}])
        assert first == second == "first"
        assert len(calls) == 1  # stub only invoked once -- second call was a cache hit

    def test_different_messages_are_different_cache_keys(self, tmp_path):
        client = make_local_client(tmp_path)
        calls = stub_generation(client, [("a", 10, 5), ("b", 10, 5)])
        client.complete([{"role": "user", "content": "one"}])
        client.complete([{"role": "user", "content": "two"}])
        assert len(calls) == 2

    def test_different_model_is_a_different_cache_key(self, tmp_path):
        client_a = make_local_client(tmp_path, model="model-a")
        client_b = make_local_client(tmp_path, model="model-b")
        stub_generation(client_a, [("from a", 10, 5)])
        stub_generation(client_b, [("from b", 10, 5)])
        result_a = client_a.complete([{"role": "user", "content": "same question"}])
        result_b = client_b.complete([{"role": "user", "content": "same question"}])
        assert result_a == "from a"
        assert result_b == "from b"

    def test_different_load_in_4bit_is_a_different_cache_key(self, tmp_path):
        client_4bit = make_local_client(tmp_path, load_in_4bit=True)
        client_fp16 = make_local_client(tmp_path, load_in_4bit=False)
        stub_generation(client_4bit, [("4bit answer", 10, 5)])
        stub_generation(client_fp16, [("fp16 answer", 10, 5)])
        result_4bit = client_4bit.complete([{"role": "user", "content": "same question"}])
        result_fp16 = client_fp16.complete([{"role": "user", "content": "same question"}])
        assert result_4bit == "4bit answer"
        assert result_fp16 == "fp16 answer"


# --------------------------------------------------------------------------
# complete_with_usage()
# --------------------------------------------------------------------------


class TestCompleteWithUsage:
    def test_fresh_call_returns_exact_token_counts_from_the_tokenizer(self, tmp_path):
        client = make_local_client(tmp_path)
        stub_generation(client, [("The answer.", 42, 17)])
        result = client.complete_with_usage([{"role": "user", "content": "hi"}])
        assert result.text == "The answer."
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 17
        assert result.cached is False
        assert result.latency_s is not None
        assert result.model == "fake/local-model"

    def test_cache_hit_reports_cached_true_and_recovers_token_counts(self, tmp_path):
        client = make_local_client(tmp_path)
        stub_generation(client, [("The answer.", 42, 17)])
        client.complete_with_usage([{"role": "user", "content": "hi"}])
        second = client.complete_with_usage([{"role": "user", "content": "hi"}])
        assert second.cached is True
        assert second.prompt_tokens == 42
        assert second.completion_tokens == 17
        assert second.latency_s is None  # a hit's replay time is never reported as real

    def test_complete_and_complete_with_usage_share_one_cache_entry(self, tmp_path):
        client = make_local_client(tmp_path)
        calls = stub_generation(client, [("shared answer", 10, 5)])
        client.complete([{"role": "user", "content": "hi"}])
        result = client.complete_with_usage([{"role": "user", "content": "hi"}])
        assert result.cached is True
        assert result.text == "shared answer"
        assert len(calls) == 1

    def test_bypass_cache_skips_both_read_and_write(self, tmp_path):
        client = make_local_client(tmp_path)
        calls = stub_generation(client, [("a", 10, 5), ("b", 10, 5), ("c", 10, 5)])
        first = client.complete_with_usage([{"role": "user", "content": "hi"}], bypass_cache=True)
        second = client.complete_with_usage([{"role": "user", "content": "hi"}], bypass_cache=True)
        assert first.cached is False
        assert second.cached is False
        assert len(calls) == 2  # never served from cache, so the stub ran twice
        # A normal (non-bypass) call afterwards is also a fresh call -- bypass_cache
        # never *wrote* to the cache either.
        third = client.complete_with_usage([{"role": "user", "content": "hi"}])
        assert third.cached is False
        assert len(calls) == 3

    def test_every_call_appends_a_row_to_the_call_log(self, tmp_path):
        client = make_local_client(tmp_path)
        stub_generation(client, [("a", 10, 5), ("b", 10, 5)])
        client.complete_with_usage([{"role": "user", "content": "one"}])
        client.complete_with_usage([{"role": "user", "content": "two"}])
        lines = (tmp_path / "api_calls_v1.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert set(row) == {"timestamp", "model", "prompt_tokens", "completion_tokens", "latency_s", "cached"}


# --------------------------------------------------------------------------
# Interchangeability with GroqClient (the M7.md T7.7 validation check)
# --------------------------------------------------------------------------


def run_through_shared_harness(usage_fn: UsageFn) -> CompletionResult:
    """A harness written against nothing but the `UsageFn` shape -- exactly what
    T7.8's baseline-eval harness (built against `GroqClient`) needs to also accept
    `LocalHFClient` with a config swap and nothing else.
    """
    return usage_fn([{"role": "user", "content": "Does this run true to size?"}])


class TestInterchangeableWithGroqClient:
    def test_both_clients_return_a_completion_result_with_the_same_field_set(self, tmp_path):
        local_client = make_local_client(tmp_path)
        stub_generation(local_client, [("Local answer.", 10, 5)])

        groq_client = GroqClient(
            model="openai/gpt-oss-20b",
            cache_dir=str(tmp_path / "groq_cache"),
            call_log_path=str(tmp_path / "groq_calls.jsonl"),
        )

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"content": "Groq answer."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                    "model": "openai/gpt-oss-20b",
                }

        groq_client._session.post = lambda *a, **kw: FakeResponse()
        import os

        os.environ["GROQ_API_KEY"] = "test-key"

        local_result = run_through_shared_harness(local_client.complete_with_usage)
        groq_result = run_through_shared_harness(groq_client.complete_with_usage)

        assert type(local_result) is type(groq_result) is CompletionResult
        assert {f.name for f in local_result.__dataclass_fields__.values()} == {
            f.name for f in groq_result.__dataclass_fields__.values()
        }
        assert local_result.text == "Local answer."
        assert groq_result.text == "Groq answer."
        assert local_result.prompt_tokens is not None
        assert groq_result.prompt_tokens is not None


# --------------------------------------------------------------------------
# probe_model / probe_models / select_base_model
# --------------------------------------------------------------------------


class FakeCudaModule:
    """A minimal stand-in for `torch.cuda`, driven by a scripted sequence of
    `max_memory_allocated`/`memory_allocated` return values so `probe_model`'s VRAM
    bookkeeping can be tested without a real GPU.
    """

    def __init__(self, load_peak_mb: float, resident_mb: float, generation_peak_mb: float):
        self._load_peak = load_peak_mb * 1024**2
        self._resident = resident_mb * 1024**2
        self._generation_peak = generation_peak_mb * 1024**2
        self._reset_count = 0
        self.is_available = lambda: True

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self):
        self._reset_count += 1

    def max_memory_allocated(self):
        # First reset -> load phase; second reset -> generation phase.
        return self._load_peak if self._reset_count <= 1 else self._generation_peak

    def memory_allocated(self):
        return self._resident


class TestProbeModelClassification:
    def test_generation_within_yes_fraction_is_yes(self, monkeypatch, tmp_path):
        import cragb.finetune.local_client as module

        fake_torch = _make_fake_torch(FakeCudaModule(load_peak_mb=1500, resident_mb=1400, generation_peak_mb=1800))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        client_stub_calls = []

        def fake_run_generation(self, messages, max_new_tokens=None, min_new_tokens=None):
            client_stub_calls.append(1)
            return "generated text", 10, 2048

        monkeypatch.setattr(module.LocalHFClient, "load", lambda self: None)
        monkeypatch.setattr(module.LocalHFClient, "_run_generation", fake_run_generation)
        monkeypatch.setattr(module.LocalHFClient, "unload", lambda self: None)

        result = module.probe_model("fake-model", gpu_total_vram_mb=4096, generation_tokens=2048)
        assert result.fits == "yes"
        assert result.generation_peak_vram_mb == 1800
        assert result.error is None
        assert client_stub_calls == [1]

    def test_generation_between_yes_fraction_and_total_is_marginal(self, monkeypatch):
        import cragb.finetune.local_client as module

        fake_torch = _make_fake_torch(FakeCudaModule(load_peak_mb=3000, resident_mb=2900, generation_peak_mb=3700))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        monkeypatch.setattr(module.LocalHFClient, "load", lambda self: None)
        monkeypatch.setattr(
            module.LocalHFClient,
            "_run_generation",
            lambda self, messages, max_new_tokens=None, min_new_tokens=None: ("x", 10, 2048),
        )
        monkeypatch.setattr(module.LocalHFClient, "unload", lambda self: None)

        result = module.probe_model("fake-model", gpu_total_vram_mb=4096, generation_tokens=2048)
        assert result.fits == "marginal"

    def test_generation_at_or_above_total_is_no(self, monkeypatch):
        import cragb.finetune.local_client as module

        fake_torch = _make_fake_torch(FakeCudaModule(load_peak_mb=3900, resident_mb=3800, generation_peak_mb=4200))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        monkeypatch.setattr(module.LocalHFClient, "load", lambda self: None)
        monkeypatch.setattr(
            module.LocalHFClient,
            "_run_generation",
            lambda self, messages, max_new_tokens=None, min_new_tokens=None: ("x", 10, 2048),
        )
        monkeypatch.setattr(module.LocalHFClient, "unload", lambda self: None)

        result = module.probe_model("fake-model", gpu_total_vram_mb=4096, generation_tokens=2048)
        assert result.fits == "no"

    def test_an_exception_during_probing_is_recorded_not_raised(self, monkeypatch):
        import cragb.finetune.local_client as module

        fake_torch = _make_fake_torch(FakeCudaModule(load_peak_mb=100, resident_mb=100, generation_peak_mb=100))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        def raising_load(self):
            raise RuntimeError("gated repo, no access")

        monkeypatch.setattr(module.LocalHFClient, "load", raising_load)
        monkeypatch.setattr(module.LocalHFClient, "unload", lambda self: None)

        result = module.probe_model("meta-llama/Llama-3.2-3B-Instruct", gpu_total_vram_mb=4096)
        assert result.fits == "no"
        assert result.error is not None
        assert "gated repo" in result.error

    def test_no_cuda_available_returns_a_no_result_without_raising(self, monkeypatch):
        import cragb.finetune.local_client as module

        class NoCuda:
            is_available = staticmethod(lambda: False)

        fake_torch = type("FakeTorch", (), {"cuda": NoCuda()})()
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        result = module.probe_model("fake-model", gpu_total_vram_mb=4096)
        assert result.fits == "no"
        assert "CUDA not available" in result.error


def _make_fake_torch(cuda_module: FakeCudaModule):
    return type("FakeTorch", (), {"cuda": cuda_module})()


class TestProbeModels:
    def test_returns_one_row_per_model_in_order(self, monkeypatch):
        import cragb.finetune.local_client as module

        def fake_probe_model(model_name, **kwargs):
            return ModelProbeResult(
                model=model_name, load_peak_vram_mb=100, resident_vram_mb=90,
                generation_peak_vram_mb=150, tokens_per_second=20.0, fits="yes", error=None,
            )

        monkeypatch.setattr(module, "probe_model", fake_probe_model)
        df = module.probe_models(("a", "b", "c"), gpu_total_vram_mb=4096)
        assert list(df["model"]) == ["a", "b", "c"]
        assert set(df.columns) == {
            "model", "load_peak_vram_mb", "resident_vram_mb",
            "generation_peak_vram_mb", "tokens_per_second", "fits", "error",
        }


class TestSelectBaseModel:
    def test_picks_the_only_fitting_model(self):
        df = pd.DataFrame(
            [
                {"model": "small", "fits": "yes", "generation_peak_vram_mb": 1000},
                {"model": "big", "fits": "no", "generation_peak_vram_mb": None},
            ]
        )
        assert select_base_model(df) == "small"

    def test_prefers_larger_generation_vram_among_yes_rows(self):
        # A proxy for "more capacity used, i.e. a bigger model" among multiple
        # comfortably-fitting candidates.
        df = pd.DataFrame(
            [
                {"model": "1.5b", "fits": "yes", "generation_peak_vram_mb": 1200},
                {"model": "3b", "fits": "yes", "generation_peak_vram_mb": 2400},
            ]
        )
        assert select_base_model(df) == "3b"

    def test_marginal_is_never_preferred_over_yes(self):
        df = pd.DataFrame(
            [
                {"model": "marginal_but_bigger", "fits": "marginal", "generation_peak_vram_mb": 3900},
                {"model": "fits_cleanly", "fits": "yes", "generation_peak_vram_mb": 1200},
            ]
        )
        assert select_base_model(df) == "fits_cleanly"

    def test_no_fitting_model_returns_none(self):
        df = pd.DataFrame([{"model": "too_big", "fits": "no", "generation_peak_vram_mb": None}])
        assert select_base_model(df) is None


# --------------------------------------------------------------------------
# DEFAULT_CANDIDATE_MODELS sanity
# --------------------------------------------------------------------------


def test_default_candidate_models_match_planmd_10_shortlist():
    assert DEFAULT_CANDIDATE_MODELS == (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    )
