"""QLoRA feasibility probe (T7.9; PLAN.md §10, M7.md T7.9).

Answers the two questions PLAN.md §10's go/no-go rule ("proceed only if a pilot LoRA
shows >= X improvement... within *T* hours of GPU time") cannot be written without: does a
QLoRA training step at a realistic sequence length actually fit in 4 GB, and how long does
one epoch take? T7.7 already measured *inference* VRAM for the base-model shortlist, but a
training step is a categorically heavier load -- gradients, optimizer state, and
activation memory that pure inference never touches -- so that earlier measurement cannot
answer this module's question; it has to be measured again, for real, here.

**Written now, run only as a probe.** Twenty single-example-overfit steps per
(model, LoRA rank, max_seq_len) configuration -- never a real training run. Every
hyperparameter this probe needs *beyond* what's required to make the sanity check
converge (`configs/finetune_train.yaml`'s `placeholder_hyperparameters:` block) is
explicitly marked "set in M8 from T7.8's failure modes", per PLAN.md §10's own stated
position: locking a LoRA config before T7.8's baseline showed where the untuned model
actually fails would be false precision.

**Why a hand-rolled training loop, not `trl.SFTTrainer`.** `trl` is installed (per T7.7's
environment work) and is the right tool for M8's actual training run -- but this probe
needs finer-grained control than `SFTTrainer` conveniently exposes: an exact assertion on
the label tensor's mask before any step runs, per-micro-step VRAM/timing captured
directly around a manually-constructed forward/backward pass, and an OOM on any one
configuration caught and recorded as a data point rather than propagating up through
several layers of `Trainer` machinery. Mirrors this project's established preference for
a direct implementation over a heavier framework when the framework's conveniences don't
clearly help (T2.2's `requests`-over-`groq`-SDK reasoning, restated here for `trl`).

**Label masking uses the tokenizer's own chat template twice, not a hand-counted offset**
(`build_training_tensors`): once over the full `[user, assistant]` turn pair
(`cragb.finetune.schema.to_chat_messages`, the *exact* pair T7.1 already fixed for
train/inference parity) and once over just the user turn with `add_generation_prompt=True`
-- the second is asserted to be a literal string/token prefix of the first (confirmed live
against Qwen2.5's own chat template at T7.9 build time; a tokenizer where this doesn't
hold would need a different masking strategy, and this module would rather raise than
silently mask the wrong span). Every prompt token becomes `-100` (PyTorch's
"ignore this token in the loss" sentinel); only completion tokens contribute to the loss
`model(...).loss` returns.

Usage:
    python -m cragb.finetune.train_lora --config configs/finetune_train.yaml
    python -m cragb.finetune.train_lora --probe-steps 5 --models Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cragb.finetune.schema import TrainingExample, load_training_examples_jsonl, to_chat_messages
from cragb.utils.io import load_config, resolve_path
from cragb.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

# Attention + MLP projections -- the standard QLoRA target-module set, and the exact
# names both Qwen2.5's and Llama-3.2's architectures use (confirmed against the real
# Qwen2.5-1.5B-Instruct module names at T7.9 build time). A different architecture
# family would need this list revisited, same as any other architecture-specific
# small curated constant in this project (e.g. cragb.bench.taxonomy.CATEGORY_KEYWORDS).
TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

_SEQ_LEN_ROUND_TO = 128


# --------------------------------------------------------------------------
# Prompt-length calibration
# --------------------------------------------------------------------------


def load_examples_for_length_calibration(cfg: dict) -> tuple[list[TrainingExample], str]:
    """`train.jsonl`, or -- if it's currently empty -- `filtered_pairs_v1.jsonl`, logged
    either way.

    T7.6's real train/val/probe split currently has zero examples in `train.jsonl` (the
    tiny dataset available before T7.3's full generation sweep runs was entirely consumed
    by the probe's own ~40/~40 target); this probe still needs *some* realistic sample of
    rendered-prompt lengths to calibrate `max_seq_len`, so it falls back rather than
    failing outright -- but always says so, per this module's "an empty/degenerate input
    is a finding, not a silent substitution" discipline (mirrors T7.6's embedding-backstop
    graceful degradation).

    Returns:
        `(examples, source_label)` -- `source_label` is `"train"` or
        `"filtered_pairs_fallback"`, carried into `ft_prompt_length_stats_v1.csv` so a
        reader can tell which case produced a given row.

    Raises:
        ValueError: if both `train.jsonl` and the fallback are empty -- there is nothing
            left to calibrate against.
    """
    train_examples = load_training_examples_jsonl(cfg["paths"]["train_in"])
    if train_examples:
        return train_examples, "train"

    logger.warning(
        "%s has zero examples -- falling back to %s for prompt-length calibration "
        "(see this function's docstring for why).",
        cfg["paths"]["train_in"],
        cfg["paths"]["filtered_pairs_fallback_in"],
    )
    fallback_examples = load_training_examples_jsonl(cfg["paths"]["filtered_pairs_fallback_in"])
    if not fallback_examples:
        raise ValueError(
            f"Both {cfg['paths']['train_in']!r} and "
            f"{cfg['paths']['filtered_pairs_fallback_in']!r} are empty -- nothing to "
            "calibrate max_seq_len against."
        )
    return fallback_examples, "filtered_pairs_fallback"


def compute_sequence_length_stats(examples: list[TrainingExample], tokenizer) -> dict:
    """p50/p95/max token length of every example's *full* rendered training sequence
    (prompt + completion, chat-templated) -- not the prompt alone, since both halves
    must fit within `max_seq_len` for training to see an untruncated completion.

    Args:
        examples: training examples to measure.
        tokenizer: a loaded HuggingFace tokenizer (token counts are architecture-specific,
            so this must be the *target* model's own tokenizer, not a stand-in).

    Returns:
        `{"n": int, "p50": int, "p95": int, "max": int}`.

    Raises:
        ValueError: if `examples` is empty.
    """
    if not examples:
        raise ValueError("examples must not be empty")

    lengths = []
    for example in examples:
        messages = to_chat_messages(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    arr = np.asarray(lengths, dtype=float)
    return {
        "n": len(lengths),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def _round_up(n: int, multiple: int = _SEQ_LEN_ROUND_TO) -> int:
    return int(np.ceil(n / multiple) * multiple)


def derive_max_seq_len_candidates(stats: dict) -> list[int]:
    """`max_seq_len` candidates from a `compute_sequence_length_stats` result: the p95
    length rounded up to the nearest `_SEQ_LEN_ROUND_TO` (the "realistic" budget most
    training examples fit inside) and, if it differs, the max length rounded up the same
    way (the "no truncation at all" budget -- worth its own VRAM measurement precisely
    because it's the more expensive one).

    Returns:
        `[p95_rounded]` if `p95_rounded == max_rounded`, else `[p95_rounded, max_rounded]`,
        smaller first.
    """
    p95_rounded = _round_up(stats["p95"])
    max_rounded = _round_up(stats["max"])
    return [p95_rounded] if p95_rounded == max_rounded else [p95_rounded, max_rounded]


def select_probe_example(
    examples: list[TrainingExample], tokenizer, max_seq_len: int | None = None
) -> TrainingExample:
    """The longest-rendering example in `examples` that still fits within `max_seq_len`
    -- a deliberate worst-case choice for a *feasibility* probe: "does the worst case
    that's actually supposed to fit, fit" is the question that matters, not "does a
    typical case fit". `max_seq_len=None` (the unconstrained case) returns the single
    longest example overall.

    **Must be called per `max_seq_len` candidate, not once and reused.** The absolute
    longest example in a real dataset is, by construction, close to (or above) the
    *largest* `max_seq_len` candidate `derive_max_seq_len_candidates` returns -- testing
    that same example against a *smaller* candidate (e.g. the p95-derived one) would
    truncate its completion away entirely and spuriously fail
    `assert_label_mask`'s "nothing would train" check, confounding a genuine
    doesn't-fit-in-VRAM result with a config-of-the-probe-itself mistake (confirmed live
    at T7.9 build time -- this is exactly what happened before this parameter existed).

    Args:
        examples: candidate pool.
        tokenizer: the target model's tokenizer.
        max_seq_len: only consider examples whose full rendered length is at most this
            many tokens; `None` for no constraint.

    Returns:
        The chosen example. If `max_seq_len` is given and *no* example fits within it,
        returns the single shortest example instead (logged) -- it will still be
        truncated, but truncating the shortest available example is the least-bad option
        when every example is too long for this particular candidate.
    """
    def rendered_length(example: TrainingExample) -> int:
        text = tokenizer.apply_chat_template(to_chat_messages(example), tokenize=False, add_generation_prompt=False)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    if max_seq_len is None:
        return max(examples, key=rendered_length)

    fitting = [e for e in examples if rendered_length(e) <= max_seq_len]
    if not fitting:
        logger.warning(
            "Every example's full rendered length exceeds max_seq_len=%d; falling back "
            "to the shortest available example (it will still be truncated).",
            max_seq_len,
        )
        return min(examples, key=rendered_length)
    return max(fitting, key=rendered_length)


# --------------------------------------------------------------------------
# Label-masked tokenization
# --------------------------------------------------------------------------


def build_training_tensors(example: TrainingExample, tokenizer, max_seq_len: int) -> dict:
    """Tokenize `example` into `input_ids`/`labels`, masking the prompt span to `-100`.

    Args:
        example: the training example to tokenize.
        tokenizer: a loaded HuggingFace tokenizer with a chat template.
        max_seq_len: truncate the full (prompt + completion) sequence to at most this
            many tokens -- a completion that gets fully truncated away by this is a real,
            reportable degenerate case, not silently ignored (see `assert_label_mask`).

    Returns:
        `{"input_ids": list[int], "labels": list[int], "prompt_len": int}`. `labels[i] ==
        -100` for `i < prompt_len`; `labels[i] == input_ids[i]` otherwise.

    Raises:
        ValueError: if the user-turn-only rendering (with `add_generation_prompt=True`)
            is not a token-level prefix of the full `[user, assistant]` rendering for this
            tokenizer -- the precondition this masking strategy depends on (confirmed live
            against Qwen2.5's chat template at T7.9 build time; a tokenizer where this
            doesn't hold needs a different masking approach, not a silently-wrong one).
    """
    messages = to_chat_messages(example)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_seq_len]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"][:max_seq_len]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{example.example_id}: the user-turn-only chat-template rendering is not a "
            "token prefix of the full [user, assistant] rendering for this tokenizer -- "
            "label masking would mask the wrong span."
        )
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels, "prompt_len": prompt_len}


def assert_label_mask(batch: dict) -> None:
    """Verify `build_training_tensors`' output actually masks the prompt and leaves the
    completion trainable -- the exact check M7.md T7.9 names as its own "How I verify it
    worked" step, made assertable rather than eyeballed.

    Raises:
        ValueError: if the prompt span isn't fully masked, or if the completion span is
            (a max_seq_len so small the completion was truncated away entirely --
            training on this example would silently teach nothing).
    """
    prompt_len = batch["prompt_len"]
    labels = batch["labels"]

    if any(label != -100 for label in labels[:prompt_len]):
        bad = [i for i, label in enumerate(labels[:prompt_len]) if label != -100]
        raise ValueError(f"Prompt span [0:{prompt_len}) contains non-masked label(s) at index/indices {bad}.")
    if prompt_len >= len(labels):
        raise ValueError(
            f"prompt_len ({prompt_len}) >= total sequence length ({len(labels)}) -- "
            "max_seq_len truncated the completion away entirely; nothing would train."
        )


# --------------------------------------------------------------------------
# QLoRA probe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QloraProbeResult:
    """One row of `ft_qlora_probe_v1.csv`."""

    model: str
    rank: int
    max_seq_len: int
    peak_vram_mb: float | None
    seconds_per_step: float | None
    oom: bool
    initial_loss: float | None
    final_loss: float | None
    overfit_confirmed: bool | None
    n_train_examples: int
    extrapolated_minutes_per_epoch: float | None
    error: str | None


def probe_qlora_config(
    model_name: str,
    rank: int,
    max_seq_len: int,
    probe_example: TrainingExample,
    n_train_examples: int,
    *,
    probe_steps: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lora_alpha: int,
    lora_dropout: float,
) -> QloraProbeResult:
    """Load `model_name` in 4-bit, apply LoRA at `rank`, and run `probe_steps` micro-steps
    of forward+backward+(periodic) optimizer.step() -- all on the *same* `probe_example`,
    repeated -- measuring peak VRAM and per-micro-step time, then release everything.

    Args:
        model_name: HuggingFace model id.
        rank: LoRA rank (`peft.LoraConfig.r`).
        max_seq_len: truncation length for `build_training_tensors`.
        probe_example: the single example every micro-step trains on (see
            `select_probe_example` -- a deliberate worst-case choice).
        n_train_examples: the real train-set size, for the epoch-time extrapolation
            (`n_train_examples * seconds_per_step / 60`, since micro batch size is
            always 1 -- one micro-step per training example per epoch).
        probe_steps: number of micro-steps to run.
        gradient_accumulation_steps: `optimizer.step()` fires every this-many micro-steps.
        learning_rate, lora_alpha, lora_dropout: `configs/finetune_train.yaml`'s
            `placeholder_hyperparameters` -- see that file's own comments for why these
            are marked provisional.

    Returns:
        A `QloraProbeResult`. On CUDA OOM, `oom=True` and VRAM/timing/loss fields are
        `None` except whatever `torch.cuda.max_memory_allocated()` reports even from a
        failed attempt; on any other failure (a gated/inaccessible repo, a tokenizer
        chat-template mismatch, ...) `oom=False` and `error` carries the message -- either
        way this is a recorded data point, not a crashed run (mirrors
        `cragb.finetune.local_client.probe_model`'s identical contract).
    """
    import torch

    if not torch.cuda.is_available():
        return QloraProbeResult(
            model=model_name, rank=rank, max_seq_len=max_seq_len, peak_vram_mb=None,
            seconds_per_step=None, oom=False, initial_loss=None, final_loss=None,
            overfit_confirmed=None, n_train_examples=n_train_examples,
            extrapolated_minutes_per_epoch=None, error="CUDA not available on this machine",
        )

    model = None
    optimizer = None
    try:
        # peft/transformers/bitsandbytes imported *inside* the try block, not above --
        # a missing or broken install of any of them (this module's own venv-only
        # dependency, PLAN.md §14.1) must be a recorded probe failure like any other,
        # not an uncaught ImportError that takes the whole multi-configuration sweep
        # down with it.
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.train()

        batch = build_training_tensors(probe_example, tokenizer, max_seq_len)
        assert_label_mask(batch)

        device = next(model.parameters()).device
        input_ids = torch.tensor([batch["input_ids"]], device=device)
        labels = torch.tensor([batch["labels"]], device=device)
        attention_mask = torch.ones_like(input_ids)

        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=learning_rate)

        losses: list[float] = []
        step_times: list[float] = []
        for step in range(probe_steps):
            t0 = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            (loss / gradient_accumulation_steps).backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach().item()))

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        # First micro-step includes one-off CUDA/kernel warmup (T7.7's live finding for
        # inference; the same applies to a first backward pass) -- discarded from the
        # per-step average, never averaged in, matching cragb.utils.timing's stated
        # warm-up convention.
        timed = step_times[1:] if len(step_times) > 1 else step_times
        seconds_per_step = float(np.mean(timed))

        initial_loss, final_loss = losses[0], losses[-1]
        overfit_confirmed = final_loss < initial_loss * 0.1

        extrapolated_minutes_per_epoch = (n_train_examples * seconds_per_step) / 60

        return QloraProbeResult(
            model=model_name, rank=rank, max_seq_len=max_seq_len, peak_vram_mb=peak_vram_mb,
            seconds_per_step=seconds_per_step, oom=False, initial_loss=initial_loss,
            final_loss=final_loss, overfit_confirmed=overfit_confirmed,
            n_train_examples=n_train_examples,
            extrapolated_minutes_per_epoch=extrapolated_minutes_per_epoch, error=None,
        )
    except torch.cuda.OutOfMemoryError as e:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.warning("OOM for model=%s rank=%d max_seq_len=%d: %s", model_name, rank, max_seq_len, e)
        return QloraProbeResult(
            model=model_name, rank=rank, max_seq_len=max_seq_len, peak_vram_mb=peak_vram_mb,
            seconds_per_step=None, oom=True, initial_loss=None, final_loss=None,
            overfit_confirmed=None, n_train_examples=n_train_examples,
            extrapolated_minutes_per_epoch=None, error=str(e),
        )
    except Exception as e:  # noqa: BLE001 -- a probe failure is a measurement, not a crash
        logger.warning("Probe failed for model=%s rank=%d max_seq_len=%d: %s", model_name, rank, max_seq_len, e, exc_info=True)
        return QloraProbeResult(
            model=model_name, rank=rank, max_seq_len=max_seq_len, peak_vram_mb=None,
            seconds_per_step=None, oom=False, initial_loss=None, final_loss=None,
            overfit_confirmed=None, n_train_examples=n_train_examples,
            extrapolated_minutes_per_epoch=None, error=str(e),
        )
    finally:
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- best-effort cleanup only
            pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_train.yaml", help="Path to the training-probe config YAML.")
    parser.add_argument("--probe-steps", type=int, default=None, help="Override config's probe.probe_steps.")
    parser.add_argument("--models", nargs="+", default=None, help="Override config's sweep.models.")
    parser.add_argument("--ranks", type=int, nargs="+", default=None, help="Override config's sweep.lora_ranks.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    examples, source_label = load_examples_for_length_calibration(cfg)
    logger.info("Calibrating prompt length from %d example(s) (source=%s).", len(examples), source_label)

    models = args.models or cfg["sweep"]["models"]
    ranks = args.ranks or cfg["sweep"]["lora_ranks"]
    probe_steps = args.probe_steps or cfg["probe"]["probe_steps"]
    grad_accum = cfg["probe"]["gradient_accumulation_steps"]

    placeholders = cfg["placeholder_hyperparameters"]
    learning_rate = float(placeholders["learning_rate"])
    lora_dropout = float(placeholders["lora_dropout"])
    lora_alpha_multiplier = int(placeholders["lora_alpha_multiplier"])

    from transformers import AutoTokenizer

    length_stats_rows = []
    probe_rows = []
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        stats = compute_sequence_length_stats(examples, tokenizer)
        length_stats_rows.append({"model": model_name, "source": source_label, **stats})
        candidates = derive_max_seq_len_candidates(stats)
        # One probe example per max_seq_len candidate, not one reused across all of
        # them -- see select_probe_example's docstring for why that would spuriously
        # fail the smaller candidate's label-mask check.
        probe_examples = {msl: select_probe_example(examples, tokenizer, msl) for msl in candidates}
        logger.info(
            "model=%s: n=%d p50=%d p95=%d max=%d -> max_seq_len candidates=%s",
            model_name, stats["n"], stats["p50"], stats["p95"], stats["max"], candidates,
        )

        for rank in ranks:
            for max_seq_len in candidates:
                probe_example = probe_examples[max_seq_len]
                logger.info("Probing model=%s rank=%d max_seq_len=%d (%d steps)...", model_name, rank, max_seq_len, probe_steps)
                result = probe_qlora_config(
                    model_name, rank, max_seq_len, probe_example, len(examples),
                    probe_steps=probe_steps,
                    gradient_accumulation_steps=grad_accum,
                    learning_rate=learning_rate,
                    lora_alpha=rank * lora_alpha_multiplier,
                    lora_dropout=lora_dropout,
                )
                probe_rows.append(result)
                if result.oom:
                    logger.warning("  OOM (peak %.0f MiB)", result.peak_vram_mb or -1)
                elif result.error:
                    logger.warning("  failed: %s", result.error)
                else:
                    logger.info(
                        "  peak_vram=%.0f MiB, %.2f s/step, loss %.3f -> %.3f (overfit=%s), ~%.1f min/epoch",
                        result.peak_vram_mb, result.seconds_per_step, result.initial_loss,
                        result.final_loss, result.overfit_confirmed, result.extrapolated_minutes_per_epoch,
                    )

    length_stats_df = pd.DataFrame(length_stats_rows)
    length_stats_path = resolve_path(cfg["paths"]["prompt_length_stats_out"])
    length_stats_path.parent.mkdir(parents=True, exist_ok=True)
    length_stats_df.to_csv(length_stats_path, index=False)
    logger.info("Wrote prompt-length stats to %s", length_stats_path)

    probe_df = pd.DataFrame(
        [
            {
                "model": r.model, "rank": r.rank, "max_seq_len": r.max_seq_len,
                "peak_vram_mb": r.peak_vram_mb, "seconds_per_step": r.seconds_per_step,
                "oom": r.oom, "initial_loss": r.initial_loss, "final_loss": r.final_loss,
                "overfit_confirmed": r.overfit_confirmed, "n_train_examples": r.n_train_examples,
                "extrapolated_minutes_per_epoch": r.extrapolated_minutes_per_epoch, "error": r.error,
            }
            for r in probe_rows
        ]
    )
    probe_path = resolve_path(cfg["paths"]["probe_out"])
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(probe_path, index=False)
    logger.info("Wrote QLoRA probe table to %s", probe_path)

    fitting = [r for r in probe_rows if not r.oom and r.error is None]
    if not fitting:
        logger.warning(
            "Every configuration OOM'd or failed -- per M7.md T7.9, this is the fallback "
            "trigger: T7.10's retriever-fine-tune branch becomes the primary plan."
        )
    else:
        # A row that ran without an OOM/error but never cleanly overfit its single probe
        # example (overfit_confirmed=False) hasn't actually demonstrated the label
        # masking + gradient flow work correctly at that sequence length within the
        # probed step budget -- "completed" and "verified working" are different claims,
        # and only the latter should be recommended. Prefer confirmed-overfit rows; fall
        # back to any non-error row only if none confirmed (reported as such, not hidden).
        confirmed = [r for r in fitting if r.overfit_confirmed]
        candidates_for_largest = confirmed if confirmed else fitting
        if not confirmed:
            logger.warning(
                "No configuration cleanly overfit its probe example within the step "
                "budget -- recommending from the unconfirmed set; consider more probe "
                "steps or a higher learning rate before trusting these epoch-time numbers."
            )
        largest = max(candidates_for_largest, key=lambda r: r.max_seq_len * r.rank)
        logger.info(
            "Largest %sfitting configuration: model=%s rank=%d max_seq_len=%d "
            "(peak %.0f MiB, %.1f s/step, ~%.1f min/epoch over %d example(s), overfit_confirmed=%s).",
            "confirmed-" if confirmed else "",
            largest.model, largest.rank, largest.max_seq_len, largest.peak_vram_mb,
            largest.seconds_per_step, largest.extrapolated_minutes_per_epoch,
            largest.n_train_examples, largest.overfit_confirmed,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
