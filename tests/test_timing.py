"""Unit tests for cragb.utils.timing (T5.1; M5.md T5.1).

Covers the validation checks M5.md specifies for this task: percentiles on
a fixed, hand-checkable list match numpy's linear-interpolation convention;
warm-up calls are genuinely excluded from the returned durations (verified
by counting actual invocations, not by eyeballing a mean); `peak_memory`
degrades to `vram_mb=None` rather than raising when CUDA is unavailable;
and `Timer` still records a duration when the wrapped block raises.
"""

from __future__ import annotations

import time

import pytest

from cragb.utils.timing import Timer, latency_stats, peak_memory, time_calls


class TestTimer:
    def test_records_a_positive_duration(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_s is not None
        assert t.elapsed_s > 0

    def test_records_duration_even_when_block_raises(self):
        timer = Timer()
        with pytest.raises(ValueError):
            with timer:
                time.sleep(0.01)
                raise ValueError("boom")
        assert timer.elapsed_s is not None
        assert timer.elapsed_s > 0


class TestTimeCalls:
    def test_returns_one_duration_per_timed_call(self):
        args_list = [(1,), (2,), (3,)]
        durations = time_calls(lambda x: x, args_list, repeats=2, warmup=0)
        assert len(durations) == len(args_list) * 2
        assert all(d >= 0 for d in durations)

    def test_warmup_calls_execute_but_are_excluded_from_the_result(self):
        calls: list[int] = []

        def fn(x: int) -> None:
            calls.append(x)

        args_list = [(1,), (2,)]
        durations = time_calls(fn, args_list, repeats=3, warmup=5)

        # (warmup + repeats) full passes over args_list actually executed...
        assert len(calls) == (5 + 3) * len(args_list)
        # ...but only the `repeats` passes are timed/returned.
        assert len(durations) == 3 * len(args_list)

    def test_rejects_empty_args_list(self):
        with pytest.raises(ValueError):
            time_calls(lambda: None, [], repeats=1, warmup=0)

    def test_rejects_non_positive_repeats(self):
        with pytest.raises(ValueError):
            time_calls(lambda x: x, [(1,)], repeats=0, warmup=0)

    def test_rejects_negative_warmup(self):
        with pytest.raises(ValueError):
            time_calls(lambda x: x, [(1,)], repeats=1, warmup=-1)


class TestLatencyStats:
    def test_percentiles_on_a_known_list_match_numpy_linear_interpolation(self):
        seconds = [float(x) for x in range(1, 101)]  # 1..100
        stats = latency_stats(seconds)

        assert stats["n"] == 100
        assert stats["mean"] == pytest.approx(50.5)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(100.0)
        assert stats["p50"] == pytest.approx(50.5)
        assert stats["p90"] == pytest.approx(90.1)
        assert stats["p95"] == pytest.approx(95.05)

    def test_qps_matches_n_over_total_duration(self):
        seconds = [0.1, 0.2, 0.3, 0.4]
        stats = latency_stats(seconds)
        assert stats["qps"] == pytest.approx(len(seconds) / sum(seconds))

    def test_single_value_collapses_all_percentiles_to_it(self):
        stats = latency_stats([2.0])
        assert stats["n"] == 1
        for key in ("mean", "p50", "p90", "p95", "min", "max"):
            assert stats[key] == pytest.approx(2.0)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            latency_stats([])


class TestPeakMemory:
    def test_returns_positive_rss_and_a_well_typed_vram(self):
        result = peak_memory()
        assert result["rss_mb"] > 0
        assert result["vram_mb"] is None or isinstance(result["vram_mb"], float)
