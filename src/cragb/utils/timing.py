"""Timing + resource-measurement primitives (T5.1; PLAN.md §5, §3 E6, §8 G4).

PLAN.md §5 lists `utils/timing.py` in the intended source tree but it was
never built through M1-M4b; every experiment that needed a duration so far
either didn't need one (retrieval/generation correctness) or hand-rolled a
`time.time()` delta (`retrieval_index_build_v1.json`'s `build_seconds`).
M5 (cost & latency, E6/G4) needs several *comparable* timing numbers across
retrieval and generation, so this module is the one place that fixes the
clock, the warm-up policy, and the percentile convention, rather than each
harness (`cost_latency_retrieval.py`, `run_cost_latency.py`, ...) picking its
own.

Conventions fixed here, load-bearing for every M5 table:

- **Clock:** `time.perf_counter()`, not `time.time()`. `perf_counter` is
  monotonic and immune to wall-clock adjustments (NTP sync, DST); `time.time()`
  can jump backwards mid-measurement and silently produce a negative or
  wrong duration.
- **Warm-up:** the first call into freshly-loaded code (a cold BM25 index,
  an unwarmed CUDA context, Python's own import/JIT-adjacent caches) is
  routinely slower than steady-state and would bias a small sample. Warm-up
  iterations are always executed and always discarded, never averaged in.
- **Percentiles:** `numpy.percentile`'s default linear-interpolation method,
  applied via `latency_stats`, so every table in this project reports p50/p90/p95
  under the same interpolation rule.
- **QPS:** `n / sum(seconds)` — the throughput of *this* sequential,
  single-client measurement, not a claim about concurrent server capacity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil


@dataclass
class Timer:
    """Context manager measuring wall-clock elapsed time via a monotonic clock.

    Records `elapsed_s` on `__exit__`, including when the wrapped block
    raises — the exception propagates unchanged (the manager never
    suppresses it), but the partial duration up to the raise is still
    captured, since a caller measuring "how long did this attempt take"
    usually wants that even for a failed attempt.

    Example:
        with Timer() as t:
            do_work()
        t.elapsed_s  # seconds, float
    """

    elapsed_s: float | None = field(default=None, init=False)
    _start: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self._start is not None
        self.elapsed_s = time.perf_counter() - self._start
        return False


def time_calls(
    fn: Callable[..., Any],
    args_list: list[tuple[Any, ...]],
    repeats: int = 1,
    warmup: int = 0,
) -> list[float]:
    """Time `fn(*args)` for each `args` in `args_list`, discarding warm-up.

    Runs the full `args_list` `warmup` times first with no timing (letting
    caches, indexes, and CUDA contexts reach steady state), then runs it
    `repeats` more times, recording one duration per individual call.

    Args:
        fn: callable to time; called once per element of `args_list`.
        args_list: positional-argument tuples, one per logical call (e.g.
            one per benchmark question).
        repeats: number of timed passes over `args_list`.
        warmup: number of untimed passes over `args_list` run first.

    Returns:
        A flat list of `repeats * len(args_list)` per-call durations in
        seconds, in the order the calls were made. Warm-up calls are not
        included, individually or averaged.

    Raises:
        ValueError: if `args_list` is empty, `repeats < 1`, or `warmup < 0`.
    """
    if not args_list:
        raise ValueError("args_list must be non-empty")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        for args in args_list:
            fn(*args)

    durations: list[float] = []
    for _ in range(repeats):
        for args in args_list:
            with Timer() as t:
                fn(*args)
            assert t.elapsed_s is not None
            durations.append(t.elapsed_s)
    return durations


def latency_stats(seconds: list[float]) -> dict[str, float | int]:
    """Summarize a list of per-call durations into standard latency stats.

    Percentiles use `numpy.percentile`'s default linear-interpolation
    method. `qps` is `n / sum(seconds)` — sequential, single-client
    throughput implied by this sample, not concurrent server capacity.

    Args:
        seconds: per-call durations in seconds (as returned by `time_calls`).

    Returns:
        Dict with `n` (int) and `mean`, `p50`, `p90`, `p95`, `min`, `max`,
        `qps` (all float, seconds except `qps` which is calls/second).

    Raises:
        ValueError: if `seconds` is empty.
    """
    if not seconds:
        raise ValueError("seconds must be non-empty")

    import numpy as np

    arr = np.asarray(seconds, dtype=float)
    total = float(arr.sum())
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "qps": (arr.size / total) if total > 0 else float("inf"),
    }


def peak_memory() -> dict[str, float | None]:
    """Current process RSS and, if CUDA is available, peak allocated VRAM.

    `rss_mb` is the current resident set size, not a tracked historical
    peak (psutil exposes no cross-platform peak-RSS counter) — callers
    wanting a peak should sample this at the point of expected maximum
    usage, or poll repeatedly and take the max themselves. `vram_mb` *is*
    a true peak (`torch.cuda.max_memory_allocated()`, reset via
    `torch.cuda.reset_peak_memory_stats()` if a caller wants it scoped to
    one operation) because CUDA tracks it natively.

    Returns:
        Dict with `rss_mb` (float) and `vram_mb` (float, or `None` when
        torch is not installed or no CUDA device is available).
    """
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    vram_mb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass

    return {"rss_mb": rss_mb, "vram_mb": vram_mb}
