"""共享统计工具:把"4 轮取后 3"稳定中位数 + 真 p95 提取为单一来源。

被 ``benchmark_ops.py`` / ``profiling/bench_cutile.py`` / ``python/mnist/benchmark.py`` import。
"""

from __future__ import annotations

import statistics
from typing import Callable, Iterable


def percentile(sorted_values: list[float], pct: float) -> float:
    """返回真正的百分位 (线性插值)。

    替代原 ``benchmark.py`` 里 ``int(len*0.95)`` off-by-one 的写法。
    pct ∈ [0,1]。输入必须已排序。
    """
    if not sorted_values:
        raise ValueError("percentile requires non-empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pct = max(0.0, min(1.0, pct))
    pos = (len(sorted_values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def p95(values: Iterable[float]) -> float:
    """p95 of a list-like, defensive against unsorted input."""
    arr = sorted(float(v) for v in values)
    return percentile(arr, 0.95)


def median(values: Iterable[float]) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        raise ValueError("median requires non-empty list")
    return statistics.median(arr)


def stable_median(
    run_once: Callable[[], float],
    rounds: int = 4,
    take_last: int = 3,
) -> dict:
    """跑 ``rounds`` 次 ``run_once()`` (每次返回 1 个 ms),丢前 ``rounds-take_last`` 轮.

    返回 ``{round_ms: [...], discarded: [...], mean_last_n_ms, std_last_n_ms}``,
    与 ``profiling/bench_cutile.py`` 原本的 schema 兼容。
    """
    assert rounds >= take_last >= 1, "need rounds >= take_last >= 1"
    times = [float(run_once()) for _ in range(rounds)]
    discarded = times[: rounds - take_last]
    kept = times[rounds - take_last :]
    mean = statistics.mean(kept)
    std = statistics.stdev(kept) if len(kept) > 1 else 0.0
    return {
        "round_ms": times,
        "discarded": discarded[0] if len(discarded) == 1 else discarded,
        "mean_last_n_ms": mean,
        "std_last_n_ms": std,
        # legacy keys (kept for backward compat with existing JSONs)
        "mean_last_3_ms": mean,
        "std_last_3_ms": std,
    }
