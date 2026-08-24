"""Unit tests for cragb.finetune.train_lora (T7.9; PLAN.md §10, M7.md T7.9).

No real model, GPU, or tokenizer download here -- per M7.md T7.9's own "How I verify it
worked" (watch the loss curve fall live, read the extrapolated epoch time from a real
run), the actual training/VRAM numbers are a manual verification step, not something a
unit test should fabricate. What *is* tested here, thoroughly, with a small fake
tokenizer that mimics a real chat template's prefix property: `build_training_tensors`'
label masking (the spec's own emphasis -- "write it first"), `assert_label_mask`'s
detection of both a broken mask and a fully-truncated-away completion, the prompt-length
statistics/`max_seq_len` derivation, and `load_examples_for_length_calibration`'s
train-then-fallback logic. `probe_qlora_config`'s OOM/error-handling *contract* (not its
real numbers) is exercised via a monkeypatched `torch` module, mirroring
`tests/test_local_client.py`'s identical approach for `cragb.finetune.local_client
.probe_model`.
"""

from __future__ import annotations

import pytest

from cragb.finetune.schema import TrainingExample
from cragb.finetune.train_lora import (
    QloraProbeResult,
    TARGET_MODULES,
    assert_label_mask,
    build_training_tensors,
    compute_sequence_length_stats,
    derive_max_seq_len_candidates,
    load_examples_for_length_calibration,
    select_probe_example,
)


class FakeTokenizer:
    """Mimics a real chat template's key property: the user-turn-only rendering (with
    `add_generation_prompt=True`) is a genuine string prefix of the full `[user,
    assistant]` rendering -- confirmed live against Qwen2.5's real chat template at T7.9
    build time (see module docstring). Tokenizes by character for a fully deterministic,
    easy-to-reason-about token count.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = f"PROMPT:{messages[0]['content']}||ANSWER:"
        if len(messages) > 1:
            text += messages[1]["content"]
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class NonPrefixTokenizer:
    """A deliberately broken chat template -- the user-turn-only rendering diverges from
    the full rendering instead of being a prefix of it (e.g. a template that inserts a
    different closing tag depending on whether a generation prompt was requested).
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if len(messages) > 1:
            return f"PROMPT:{messages[0]['content']}||ANSWER:{messages[1]['content']}"
        return f"PROMPT:{messages[0]['content']}||GENERATE_NOW"  # diverges at "||"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def make_example(
    example_id: str = "e1",
    question: str = "Does this run small?",
    answer: str = "Yes, it runs small [1].",
    category: str = "fit_sizing",
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        category=category,
        source_doc_ids=("1",),
        source_parent_asins=("P1",),
        question=question,
        context_text="[1] has_photo: no\nsome review context",
        answer=answer,
        cited_doc_ids=("1",),
        is_abstention=False,
        provenance={"method": "test"},
    )


# --------------------------------------------------------------------------
# build_training_tensors / assert_label_mask
# --------------------------------------------------------------------------


class TestBuildTrainingTensors:
    def test_prompt_span_is_fully_masked(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=100_000)
        assert all(label == -100 for label in batch["labels"][: batch["prompt_len"]])

    def test_completion_span_is_not_masked(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=100_000)
        assert all(label != -100 for label in batch["labels"][batch["prompt_len"] :])

    def test_completion_labels_equal_completion_input_ids(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=100_000)
        p = batch["prompt_len"]
        assert batch["labels"][p:] == batch["input_ids"][p:]

    def test_truncation_respects_max_seq_len(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=50)
        assert len(batch["input_ids"]) <= 50
        assert len(batch["labels"]) <= 50

    def test_non_prefix_chat_template_raises(self):
        with pytest.raises(ValueError, match="not a token prefix"):
            build_training_tensors(make_example(), NonPrefixTokenizer(), max_seq_len=100_000)

    def test_different_questions_give_different_prompt_lengths(self):
        short = build_training_tensors(make_example(question="Q?"), FakeTokenizer(), max_seq_len=100_000)
        long = build_training_tensors(
            make_example(question="Q" * 200 + "?"), FakeTokenizer(), max_seq_len=100_000
        )
        assert long["prompt_len"] > short["prompt_len"]


class TestAssertLabelMask:
    def test_valid_mask_does_not_raise(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=100_000)
        assert_label_mask(batch)  # no raise

    def test_broken_mask_raises(self):
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=100_000)
        # Corrupt one prompt-span label to simulate a masking bug.
        batch["labels"][0] = batch["input_ids"][0]
        with pytest.raises(ValueError, match="non-masked label"):
            assert_label_mask(batch)

    def test_fully_truncated_completion_raises(self):
        # max_seq_len so small the whole sequence is prompt -- nothing left to train on.
        batch = build_training_tensors(make_example(), FakeTokenizer(), max_seq_len=3)
        with pytest.raises(ValueError, match="truncated the completion away"):
            assert_label_mask(batch)


# --------------------------------------------------------------------------
# compute_sequence_length_stats / derive_max_seq_len_candidates
# --------------------------------------------------------------------------


class TestComputeSequenceLengthStats:
    def test_raises_on_empty_examples(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_sequence_length_stats([], FakeTokenizer())

    def test_n_matches_input_count(self):
        examples = [make_example(f"e{i}") for i in range(5)]
        stats = compute_sequence_length_stats(examples, FakeTokenizer())
        assert stats["n"] == 5

    def test_longer_questions_increase_max(self):
        examples = [make_example("e1", question="Q?"), make_example("e2", question="Q" * 500 + "?")]
        stats = compute_sequence_length_stats(examples, FakeTokenizer())
        assert stats["max"] > stats["p50"]

    def test_uniform_lengths_give_equal_percentiles(self):
        examples = [make_example(f"e{i}", question="same question?") for i in range(10)]
        stats = compute_sequence_length_stats(examples, FakeTokenizer())
        assert stats["p50"] == stats["p95"] == stats["max"]


class TestDeriveMaxSeqLenCandidates:
    def test_single_candidate_when_p95_equals_max_rounded(self):
        candidates = derive_max_seq_len_candidates({"p95": 100, "max": 105})
        assert len(candidates) == 1  # both round up to 128

    def test_two_candidates_when_p95_and_max_round_differently(self):
        candidates = derive_max_seq_len_candidates({"p95": 100, "max": 2000})
        assert len(candidates) == 2
        assert candidates[0] < candidates[1]

    def test_candidates_are_rounded_up_to_128(self):
        candidates = derive_max_seq_len_candidates({"p95": 100, "max": 100})
        assert candidates == [128]

    def test_exact_multiple_is_unchanged(self):
        candidates = derive_max_seq_len_candidates({"p95": 256, "max": 256})
        assert candidates == [256]


# --------------------------------------------------------------------------
# select_probe_example
# --------------------------------------------------------------------------


class TestSelectProbeExample:
    def test_returns_the_longest_rendering_example_when_unconstrained(self):
        short = make_example("short", question="Q?")
        long = make_example("long", question="Q" * 1000 + "?")
        chosen = select_probe_example([short, long], FakeTokenizer())
        assert chosen.example_id == "long"

    def test_single_example_returns_itself(self):
        only = make_example("only")
        assert select_probe_example([only], FakeTokenizer()) is only

    def test_with_max_seq_len_picks_longest_that_still_fits(self):
        # The regression this test locks down: picking the absolute-longest example
        # regardless of max_seq_len would truncate its completion away entirely when
        # tested against a *smaller* max_seq_len candidate (confirmed live at T7.9 build
        # time -- see select_probe_example's own docstring).
        short = make_example("short", question="Q?")
        medium = make_example("medium", question="Q" * 50 + "?")
        long = make_example("long", question="Q" * 1000 + "?")
        tokenizer = FakeTokenizer()

        long_len = len(tokenizer.apply_chat_template([{"role": "user", "content": long.question}]))
        # A max_seq_len well below "long"'s rendered length, but above "medium"'s.
        constrained = select_probe_example([short, medium, long], tokenizer, max_seq_len=long_len - 1)
        assert constrained.example_id in {"short", "medium"}
        assert constrained.example_id != "long"

    def test_falls_back_to_shortest_when_nothing_fits(self):
        short = make_example("short", question="Q?")
        long = make_example("long", question="Q" * 1000 + "?")
        chosen = select_probe_example([short, long], FakeTokenizer(), max_seq_len=1)
        assert chosen.example_id == "short"  # the least-bad option, not a crash


# --------------------------------------------------------------------------
# load_examples_for_length_calibration
# --------------------------------------------------------------------------


class TestLoadExamplesForLengthCalibration:
    def _write_examples(self, path, examples):
        from cragb.finetune.schema import write_training_examples_jsonl

        write_training_examples_jsonl(examples, path)

    def test_uses_train_when_non_empty(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        fallback_path = tmp_path / "filtered_pairs.jsonl"
        self._write_examples(train_path, [make_example("t1")])
        self._write_examples(fallback_path, [make_example("f1")])

        cfg = {"paths": {"train_in": str(train_path), "filtered_pairs_fallback_in": str(fallback_path)}}
        examples, source = load_examples_for_length_calibration(cfg)
        assert source == "train"
        assert [e.example_id for e in examples] == ["t1"]

    def test_falls_back_when_train_is_empty(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        fallback_path = tmp_path / "filtered_pairs.jsonl"
        train_path.write_text("", encoding="utf-8")
        self._write_examples(fallback_path, [make_example("f1")])

        cfg = {"paths": {"train_in": str(train_path), "filtered_pairs_fallback_in": str(fallback_path)}}
        examples, source = load_examples_for_length_calibration(cfg)
        assert source == "filtered_pairs_fallback"
        assert [e.example_id for e in examples] == ["f1"]

    def test_raises_when_both_are_empty(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        fallback_path = tmp_path / "filtered_pairs.jsonl"
        train_path.write_text("", encoding="utf-8")
        fallback_path.write_text("", encoding="utf-8")

        cfg = {"paths": {"train_in": str(train_path), "filtered_pairs_fallback_in": str(fallback_path)}}
        with pytest.raises(ValueError, match="nothing to calibrate"):
            load_examples_for_length_calibration(cfg)

    def test_raises_when_train_missing_and_fallback_missing(self, tmp_path):
        cfg = {
            "paths": {
                "train_in": str(tmp_path / "does_not_exist_train.jsonl"),
                "filtered_pairs_fallback_in": str(tmp_path / "does_not_exist_fallback.jsonl"),
            }
        }
        with pytest.raises(FileNotFoundError):
            load_examples_for_length_calibration(cfg)


# --------------------------------------------------------------------------
# TARGET_MODULES sanity
# --------------------------------------------------------------------------


def test_target_modules_covers_attention_and_mlp():
    attention = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}
    assert attention.issubset(set(TARGET_MODULES))
    assert mlp.issubset(set(TARGET_MODULES))


# --------------------------------------------------------------------------
# probe_qlora_config: error-handling contract (fake torch, no real GPU)
# --------------------------------------------------------------------------


class FakeCudaModule:
    def __init__(self):
        self.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        self._peak_mb = 512.0

    def is_available(self):
        return True

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return self._peak_mb * 1024**2


def make_fake_torch():
    cuda = FakeCudaModule()
    return type("FakeTorch", (), {"cuda": cuda, "float16": "float16"})(), cuda


def make_fake_pretrained_tokenizer_class():
    """A fake `AutoTokenizer` whose `.from_pretrained(...)` returns an object with a
    real `pad_token_id`/`eos_token` -- just enough surface for `probe_qlora_config`'s
    tokenizer setup lines to succeed, so execution reaches the (also faked)
    `AutoModelForCausalLM.from_pretrained` call where these tests want the actual
    failure to originate.
    """
    fake_tokenizer = type("FakeTokenizerInstance", (), {"pad_token_id": 0, "eos_token": "<eos>"})()
    return type("T", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})


def make_fake_peft():
    """A `peft` stand-in that never gets far enough to matter -- both tests using this
    fail inside the (also faked) `transformers.AutoModelForCausalLM.from_pretrained`
    call, which happens before `peft`'s own functions are ever invoked. It only needs
    to exist so `from peft import ...` succeeds.
    """
    return type(
        "FakePeft",
        (),
        {
            "LoraConfig": staticmethod(lambda **kw: object()),
            "get_peft_model": staticmethod(lambda model, config: model),
            "prepare_model_for_kbit_training": staticmethod(lambda model, **kw: model),
        },
    )()


class TestProbeQloraConfigErrorHandling:
    def test_no_cuda_returns_a_result_without_raising(self, monkeypatch):
        import cragb.finetune.train_lora as module

        class NoCuda:
            def is_available(self):
                return False

        fake_torch = type("FakeTorch", (), {"cuda": NoCuda()})()
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        result = module.probe_qlora_config(
            "fake-model", rank=8, max_seq_len=512, probe_example=make_example(), n_train_examples=10,
            probe_steps=5, gradient_accumulation_steps=2, learning_rate=2e-4, lora_alpha=16, lora_dropout=0.05,
        )
        assert isinstance(result, QloraProbeResult)
        assert result.oom is False
        assert "CUDA not available" in result.error

    def test_oom_during_model_load_is_caught_and_recorded(self, monkeypatch):
        import cragb.finetune.train_lora as module

        fake_torch, fake_cuda = make_fake_torch()
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        monkeypatch.setitem(__import__("sys").modules, "peft", make_fake_peft())

        class RaisingAutoModel:
            @staticmethod
            def from_pretrained(*a, **kw):
                raise fake_cuda.OutOfMemoryError("CUDA out of memory")

        fake_transformers = type(
            "FakeTransformers",
            (),
            {
                "AutoModelForCausalLM": RaisingAutoModel,
                "AutoTokenizer": make_fake_pretrained_tokenizer_class(),
                "BitsAndBytesConfig": staticmethod(lambda **kw: object()),
            },
        )()
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

        result = module.probe_qlora_config(
            "fake-model", rank=8, max_seq_len=512, probe_example=make_example(), n_train_examples=10,
            probe_steps=5, gradient_accumulation_steps=2, learning_rate=2e-4, lora_alpha=16, lora_dropout=0.05,
        )
        assert result.oom is True
        assert result.peak_vram_mb == pytest.approx(512.0)
        assert result.seconds_per_step is None

    def test_non_oom_exception_is_caught_and_recorded_not_raised(self, monkeypatch):
        import cragb.finetune.train_lora as module

        fake_torch, _ = make_fake_torch()
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        monkeypatch.setitem(__import__("sys").modules, "peft", make_fake_peft())

        class RaisingAutoModel:
            @staticmethod
            def from_pretrained(*a, **kw):
                raise RuntimeError("gated repo, no access")

        fake_transformers = type(
            "FakeTransformers",
            (),
            {
                "AutoModelForCausalLM": RaisingAutoModel,
                "AutoTokenizer": make_fake_pretrained_tokenizer_class(),
                "BitsAndBytesConfig": staticmethod(lambda **kw: object()),
            },
        )()
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

        result = module.probe_qlora_config(
            "meta-llama/Llama-3.2-3B-Instruct", rank=8, max_seq_len=512, probe_example=make_example(),
            n_train_examples=10, probe_steps=5, gradient_accumulation_steps=2,
            learning_rate=2e-4, lora_alpha=16, lora_dropout=0.05,
        )
        assert result.oom is False
        assert "gated repo" in result.error
