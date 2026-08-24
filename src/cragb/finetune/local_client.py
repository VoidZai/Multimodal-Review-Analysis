"""Local 4-bit HF inference client + measured base-model VRAM probe (T7.7; PLAN.md §10, M7.md T7.7).

Gives the project the one capability it lacks: running a model *locally*, on this
machine's own GPU, behind the exact same two-method shape every other provider client in
this project already exposes (`GroqClient`, T2.2/T5.2; `GeminiClient`, T6.2). That parity
is the whole point — `LocalHFClient` is a drop-in swap for T7.8's baseline-eval harness
(built against `GroqClient`), not a parallel code path it has to special-case.

**PLAN.md §10 assumed "small local, 4-bit, fits 8 GB."** This project's actual dev
machine is an RTX 3050 Laptop with **4096 MiB** VRAM (`environment.yml`'s T2.5 comment
already records this; confirmed again here via `nvidia-smi` at T7.7 build time). Rather
than continue building on that 8 GB assumption, `probe_models`/`main` measure the real
number for each shortlisted base model and let §10's choice follow the measurement, not
the other way around — see `results/tables/ft_model_probe_v1.csv` and PLAN.md §14.7 for
what the measurement actually found.

**4-bit quantization (`BitsAndBytesConfig`, nf4 + double quant, fp16 compute dtype) is
CUDA-only** — `bitsandbytes` has no meaningful 4-bit CPU kernel, so `LocalHFClient`
disables it automatically (with a logged warning, not a silent behaviour change) whenever
`device` resolves to `"cpu"`, falling back to a plain fp16/fp32 load instead. This is the
"fall back to fp16... or CPU-offload" contingency PLAN.md §10/M7.md T7.7 names in advance
for the case a 4-bit load doesn't work on this hardware at all.

**Token counts are exact, not estimated.** Every other provider client in this project
(`GroqClient`, `GeminiClient`) reports token counts the *API* returns, and
`cragb.eval.cost_model` has a documented `len(text) / 4` fallback (`is_estimated`) for
calls made before that telemetry existed. A local model needs no such fallback: the
tokenizer that produced the prompt and the one that will decode the completion are the
exact same object in the exact same process, so `prompt_tokens`/`completion_tokens` here
are always a precise count, never an estimate.

Usage:
    client = LocalHFClient(model="Qwen/Qwen2.5-1.5B-Instruct")
    text = client.complete([{"role": "user", "content": "Hello"}])

    python -m cragb.finetune.local_client probe --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cragb.generate.api_cache import DiskCache
from cragb.generate.api_clients import CompletionResult
from cragb.utils.io import resolve_path
from cragb.utils.timing import Timer

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
)

# generation_peak_vram_mb below this fraction of the GPU's total VRAM is "yes"; below
# 100% but above the "yes" line is "marginal" (survives this one generation, but leaves
# too little headroom for a training step's optimizer state/gradients/activations
# alongside it, which is a categorically heavier VRAM load than pure inference); at or
# above the GPU's total VRAM (an OOM) is "no". This is only ever applied to the
# *inference* probe here -- T7.9's own QLoRA feasibility probe is the actual measurement
# for whether a *training* step fits, not this heuristic.
_FITS_YES_FRACTION = 0.85


# --------------------------------------------------------------------------
# LocalHFClient
# --------------------------------------------------------------------------


@dataclass
class LocalHFClient:
    """Chat-completions client for a local HuggingFace causal LM, 4-bit quantized by default.

    Mirrors `GroqClient`'s field shape and exposes the same two methods (`complete`,
    `complete_with_usage`), the same `messages` shape
    (`list[{"role": ..., "content": ...}]`), and the same disk-cache/`bypass_cache`
    semantics (T5.5) -- so `cragb.eval.cost_model`'s accounting and any harness built
    against `GroqClient` (T7.8's baseline eval) work against this client with a config
    swap and nothing else.

    Args:
        model: a HuggingFace model id or local path (e.g.
            `"Qwen/Qwen2.5-1.5B-Instruct"`).
        device: `"cuda"` or `"cpu"`; `None` (default) auto-detects CUDA, matching
            `cragb.retrieval.dense.DenseRetriever`'s own `_resolve_device` convention.
        temperature: sampling temperature. `<= 0` uses greedy decoding
            (`do_sample=False`) rather than passing a non-positive temperature to
            `generate()`, which HuggingFace rejects.
        max_tokens: max new tokens per completion (mirrors `GroqClient.max_tokens`'s
            name, not `transformers`' own `max_new_tokens`, for interface parity).
        load_in_4bit: use `BitsAndBytesConfig` (nf4 + double quant, fp16 compute) when
            `device` resolves to `"cuda"`. Silently has no effect on CPU (bitsandbytes
            4-bit is CUDA-only) -- see module docstring.
        cache_dir: directory for the disk-cached responses. Shares `GroqClient`'s
            default (`results/cache/api`); the cache key includes `model` and
            `load_in_4bit`, so a different model or quantization setting is always a
            cache miss, never a false hit against another config's cached text.
        call_log_path: append-only JSONL log of every `complete_with_usage` call, same
            row shape and same default path as `GroqClient`/`GeminiClient`, so
            downstream cost/latency accounting sees every provider in one place.
    """

    model: str
    device: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1200
    load_in_4bit: bool = True
    cache_dir: str = "results/cache/api"
    call_log_path: str = "results/cache/api_calls_v1.jsonl"

    _cache: DiskCache = field(init=False, repr=False)
    _model_obj: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _resolved_device: str = field(default="", init=False, repr=False)
    _effective_4bit: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache = DiskCache(self.cache_dir)

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Force the model + tokenizer to load now, rather than on first `complete()` call.

        Idempotent -- a second call is a no-op. Exists so a caller (e.g. `probe_models`)
        can measure load-time VRAM/latency separately from generation-time VRAM/latency,
        and so `complete()`/`complete_with_usage()` never pay a surprise multi-second
        model-load cost inside what looks like a single inference call.
        """
        if self._model_obj is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._resolved_device = self._resolve_device()
        self._effective_4bit = self.load_in_4bit and self._resolved_device == "cuda"
        if self.load_in_4bit and not self._effective_4bit:
            logger.warning(
                "load_in_4bit=True but resolved device is %r (bitsandbytes 4-bit is "
                "CUDA-only); loading %s at full precision instead.",
                self._resolved_device,
                self.model,
            )

        logger.info(
            "LocalHFClient loading %s (4bit=%s) on device=%s",
            self.model,
            self._effective_4bit,
            self._resolved_device,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        torch_dtype = None
        if self._effective_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self._resolved_device == "cuda":
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto" if self._resolved_device == "cuda" else None,
        )
        if self._resolved_device == "cpu":
            model = model.to("cpu")
        model.eval()

        self._model_obj = model
        self._tokenizer = tokenizer

    def unload(self) -> None:
        """Release the loaded model/tokenizer and free CUDA memory.

        Not needed for normal single-model use (the process exiting frees everything
        anyway) -- exists for `probe_models`, which loads several candidate models
        in one process and must not let one's VRAM linger into the next one's
        measurement.
        """
        if self._model_obj is None:
            return
        del self._model_obj
        del self._tokenizer
        self._model_obj = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _run_generation(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """Render `messages` through the chat template, generate, and return `(text,
        prompt_tokens, completion_tokens)` -- both token counts read directly off the
        tokenizer's own tensors, never estimated.
        """
        import torch

        self.load()
        model, tokenizer = self._model_obj, self._tokenizer

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        if min_new_tokens is not None:
            generate_kwargs["min_new_tokens"] = min_new_tokens

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        completion_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(completion_ids.shape[0])
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text, prompt_tokens, completion_tokens

    def _build_payload(self, messages: list[dict[str, str]]) -> dict:
        """The sole source of the disk-cache key -- includes `model`/`load_in_4bit` so a
        different model or quantization setting is always a cache miss (mirrors
        `GroqClient._build_payload`'s role for the same reason).
        """
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "load_in_4bit": self.load_in_4bit,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the completion text for `messages`.

        Args:
            messages: OpenAI-style chat messages, e.g. `[{"role": "user", "content": "..."}]`.

        Returns:
            The completion text.
        """
        payload = self._build_payload(messages)

        def _call() -> str:
            text, _, _ = self._run_generation(messages)
            return text

        return self._cache.call(payload, _call)

    def complete_with_usage(
        self, messages: list[dict[str, str]], bypass_cache: bool = False
    ) -> CompletionResult:
        """Like `complete`, but also returns exact token counts, latency, and cache status.

        Args:
            messages: OpenAI-style chat messages.
            bypass_cache: if `True`, always runs generation and never touches the disk
                cache (read or write) -- mirrors `GroqClient.complete_with_usage`'s flag,
                same reasoning: a cache hit's near-zero replay time would make a latency
                measurement meaningless.

        Returns:
            A `CompletionResult` (shared type -- same shape as `GroqClient`'s).
        """
        payload = self._build_payload(messages)

        def _call() -> tuple[str, dict]:
            with Timer() as t:
                text, prompt_tokens, completion_tokens = self._run_generation(messages)
            assert t.elapsed_s is not None
            return text, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_s": t.elapsed_s,
                "model": self.model,
            }

        if bypass_cache:
            text, meta = _call()
            cached = False
        else:
            text, meta, cached = self._cache.call_with_meta(payload, _call)

        result = CompletionResult(
            text=text,
            prompt_tokens=(meta.get("prompt_tokens") if meta else None),
            completion_tokens=(meta.get("completion_tokens") if meta else None),
            latency_s=(None if cached else (meta.get("latency_s") if meta else None)),
            cached=cached,
            model=(meta.get("model") if meta else None) or self.model,
        )
        self._log_call(result)
        return result

    def _log_call(self, result: CompletionResult) -> None:
        """Append one JSONL row for `result` to `call_log_path` -- identical row shape to
        `GroqClient._log_call`, so both providers land in one shared log.
        """
        import json
        from datetime import datetime, timezone

        path = resolve_path(self.call_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": result.latency_s,
            "cached": result.cached,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


# --------------------------------------------------------------------------
# Model probe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProbeResult:
    """One row of `ft_model_probe_v1.csv`."""

    model: str
    load_peak_vram_mb: float | None
    resident_vram_mb: float | None
    generation_peak_vram_mb: float | None
    tokens_per_second: float | None
    fits: str  # "yes" | "marginal" | "no"
    error: str | None


def probe_model(
    model_name: str,
    *,
    gpu_total_vram_mb: float,
    generation_tokens: int = 2048,
    load_in_4bit: bool = True,
) -> ModelProbeResult:
    """Load `model_name`, measure VRAM at load and at a forced `generation_tokens`-token
    generation, and tokens/s -- then release it before returning.

    Args:
        model_name: HuggingFace model id.
        gpu_total_vram_mb: this GPU's total VRAM (from `nvidia-smi`), the denominator
            `_FITS_YES_FRACTION` is measured against.
        generation_tokens: exact completion length to force via `min_new_tokens ==
            max_new_tokens` -- a real inference call would often stop early at an EOS
            token, which would understate the KV-cache growth a genuinely long
            generation (or a training sequence of comparable length) actually costs.
        load_in_4bit: forwarded to `LocalHFClient`.

    Returns:
        A `ModelProbeResult`. On any failure (OOM, a gated/inaccessible repo, a network
        error) `error` is populated, VRAM/tokens-per-second fields are `None`, and `fits`
        is `"no"` -- a probe failure is itself a measurement, not a crash, matching
        `sample_contexts`/`generate_all`'s established graceful-degradation pattern
        elsewhere in this milestone.
    """
    import torch

    if not torch.cuda.is_available():
        return ModelProbeResult(
            model=model_name,
            load_peak_vram_mb=None,
            resident_vram_mb=None,
            generation_peak_vram_mb=None,
            tokens_per_second=None,
            fits="no",
            error="CUDA not available on this machine",
        )

    client = LocalHFClient(model=model_name, device="cuda", load_in_4bit=load_in_4bit)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        client.load()
        load_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        resident_vram_mb = torch.cuda.memory_allocated() / (1024**2)

        torch.cuda.reset_peak_memory_stats()
        prompt = [{"role": "user", "content": "Write a detailed, extremely long essay about shoes."}]
        t0 = time.perf_counter()
        _text, _prompt_tokens, completion_tokens = client._run_generation(
            prompt, max_new_tokens=generation_tokens, min_new_tokens=generation_tokens
        )
        elapsed_s = time.perf_counter() - t0
        generation_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        tokens_per_second = completion_tokens / elapsed_s if elapsed_s > 0 else None

        if generation_peak_vram_mb < gpu_total_vram_mb * _FITS_YES_FRACTION:
            fits = "yes"
        elif generation_peak_vram_mb < gpu_total_vram_mb:
            fits = "marginal"
        else:
            fits = "no"

        return ModelProbeResult(
            model=model_name,
            load_peak_vram_mb=load_peak_vram_mb,
            resident_vram_mb=resident_vram_mb,
            generation_peak_vram_mb=generation_peak_vram_mb,
            tokens_per_second=tokens_per_second,
            fits=fits,
            error=None,
        )
    except Exception as e:  # noqa: BLE001 -- a probe failure is a measurement, not a crash
        logger.warning("Probe failed for %s: %s", model_name, e, exc_info=True)
        return ModelProbeResult(
            model=model_name,
            load_peak_vram_mb=None,
            resident_vram_mb=None,
            generation_peak_vram_mb=None,
            tokens_per_second=None,
            fits="no",
            error=str(e),
        )
    finally:
        client.unload()


def probe_models(
    model_names: tuple[str, ...],
    *,
    gpu_total_vram_mb: float,
    generation_tokens: int = 2048,
    load_in_4bit: bool = True,
) -> pd.DataFrame:
    """`probe_model` over each of `model_names`, one row per model, in order."""
    rows = [
        probe_model(
            name,
            gpu_total_vram_mb=gpu_total_vram_mb,
            generation_tokens=generation_tokens,
            load_in_4bit=load_in_4bit,
        )
        for name in model_names
    ]
    return pd.DataFrame(
        [
            {
                "model": r.model,
                "load_peak_vram_mb": r.load_peak_vram_mb,
                "resident_vram_mb": r.resident_vram_mb,
                "generation_peak_vram_mb": r.generation_peak_vram_mb,
                "tokens_per_second": r.tokens_per_second,
                "fits": r.fits,
                "error": r.error,
            }
            for r in rows
        ]
    )


def select_base_model(probe_df: pd.DataFrame) -> str | None:
    """Pick the largest model that measured `fits == "yes"`, preferring `"yes"` over
    `"marginal"` and larger `generation_peak_vram_mb` (a proxy for model capacity) as the
    tiebreak among `"yes"` rows.

    Returns:
        The chosen model id, or `None` if no row fits (a real, reportable outcome -- see
        module docstring's fp16/CPU-offload fallback note).
    """
    fitting = probe_df[probe_df["fits"] == "yes"]
    if fitting.empty:
        return None
    return fitting.sort_values("generation_peak_vram_mb", ascending=False).iloc[0]["model"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _get_gpu_total_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024**2)
    except ImportError:
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe"], help="What to do.")
    parser.add_argument("--config", default="configs/finetune.yaml", help="Path to fine-tuning config YAML.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model ids to probe (default: {DEFAULT_CANDIDATE_MODELS}).",
    )
    parser.add_argument("--generation-tokens", type=int, default=2048, help="Forced completion length for the probe.")
    parser.add_argument(
        "--out",
        default="results/tables/ft_model_probe_v1.csv",
        help="Where to write the probe CSV.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    gpu_total_vram_mb = _get_gpu_total_vram_mb()
    if gpu_total_vram_mb is None:
        logger.error("No CUDA GPU detected; the probe requires one to measure anything meaningful.")
        return 1

    model_names = tuple(args.models) if args.models else DEFAULT_CANDIDATE_MODELS
    logger.info("GPU total VRAM: %.0f MiB. Probing %d model(s)...", gpu_total_vram_mb, len(model_names))

    probe_df = probe_models(model_names, gpu_total_vram_mb=gpu_total_vram_mb, generation_tokens=args.generation_tokens)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(out_path, index=False)
    logger.info("Wrote probe table to %s", out_path)

    chosen = select_base_model(probe_df)
    if chosen is None:
        logger.warning(
            "No candidate model fit within the measured VRAM budget -- see PLAN.md §10's "
            "fp16/CPU-offload fallback."
        )
    else:
        row = probe_df[probe_df["model"] == chosen].iloc[0]
        logger.info(
            "Chosen base model: %s (generation peak VRAM %.0f MiB, %.1f tok/s)",
            chosen,
            row["generation_peak_vram_mb"],
            row["tokens_per_second"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
