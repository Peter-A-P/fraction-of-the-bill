"""The arithmetic of a load test: percentiles, throughput, and the intervals on both.

A closed-loop load client keeps N requests in flight, sending the next as each one returns,
and records when each request was sent, when its first token arrived and when it finished.
Everything published about serving is computed here from those records, so the definitions
are written down once and tested against streams whose answers are known.

Percentiles are nearest-rank: the p-th percentile of n values is the value at rank
ceil(p / 100 * n) in sorted order. It is always a latency that some request actually had,
which interpolation does not guarantee, and it is what a reader reproducing p99 from the
published raw timings will get without having to ask which of the nine textbook methods was
used. With fewer than 100 requests a p99 is the maximum, and the interval says how little
that is worth.

Throughput is completed requests over the measured window, from the first request sent to
the last one finished, after a warm-up is discarded. A failed request counts against the
error rate and not towards throughput, because a server that answers quickly by failing
is not fast.

Every figure carries a 95% bootstrap interval over requests, like every other number this
project publishes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict

#: Resamples for the bootstrap intervals on latency percentiles and throughput.
RESAMPLES: Final = 2_000


class RequestTiming(BaseModel):
    """One request as the load client saw it. Times are seconds on one monotonic clock."""

    model_config = ConfigDict(frozen=True)

    sent_at: float
    #: When the first output token arrived. None if the request failed before producing one.
    first_token_at: float | None
    done_at: float
    output_tokens: int
    ok: bool
    #: Prompt tokens, as the server counted them. The overlay divides the prefill share of
    #: a call's cost by their mean, so it is recorded rather than estimated.
    input_tokens: int = 0


class Measured(BaseModel):
    """A measured quantity, its 95% interval, and the unit it is in."""

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float
    n: int
    unit: str

    def __str__(self) -> str:
        return (
            f"{self.point:,.1f} {self.unit} ({self.low:,.1f} to {self.high:,.1f}, n = {self.n:,})"
        )


class LoadSummary(BaseModel):
    """What one concurrency level of one model on one GPU measured."""

    model_config = ConfigDict(frozen=True)

    concurrency: int
    requests: int
    errors: int
    window_seconds: float
    requests_per_second: Measured
    output_tokens_per_second: Measured
    ttft_p50_ms: Measured
    ttft_p99_ms: Measured
    e2e_p50_ms: Measured
    e2e_p99_ms: Measured


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the value at rank ceil(p / 100 * n), 1-based, in sorted order."""
    if not values:
        raise ValueError("a percentile of no values is undefined")
    if not 0 < p <= 100:
        raise ValueError(f"p must be in (0, 100]; got {p}")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def _bootstrap(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    unit: str,
) -> Measured:
    data = np.asarray(values, dtype=float)
    point = statistic(data)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(RESAMPLES, data.size))
    stats = np.array([statistic(data[row]) for row in draws])
    low, high = np.percentile(stats, [2.5, 97.5])
    return Measured(point=point, low=float(low), high=float(high), n=int(data.size), unit=unit)


def _nearest_rank(p: float) -> Callable[[np.ndarray], float]:
    def statistic(data: np.ndarray) -> float:
        return percentile(data.tolist(), p)

    return statistic


def mean_measured(values: Sequence[float], *, unit: str, seed: int = 0) -> Measured:
    """The mean of a sample with its 95% interval. Used for money as well as for time."""
    if not values:
        raise ValueError("a mean of no values is undefined")
    return _bootstrap(values, lambda data: float(data.mean()), seed=seed, unit=unit)


def percentile_measured(
    values: Sequence[float], p: float, *, unit: str = "ms", seed: int = 0
) -> Measured:
    """One nearest-rank percentile with its 95% interval, for a set of calls that was not a
    load test: the frontier baselines are paced by the vendor rather than by a closed loop,
    so they have latencies worth reporting but no throughput to report with them."""
    if not values:
        raise ValueError("a percentile of no values is undefined")
    return _bootstrap(values, _nearest_rank(p), seed=seed, unit=unit)


def summarise(
    timings: Sequence[RequestTiming],
    *,
    concurrency: int,
    warmup_seconds: float = 0.0,
    seed: int = 0,
) -> LoadSummary:
    """Summarise one run. Requests sent during the warm-up are discarded before anything else.

    The throughput interval resamples requests and keeps the window fixed. That is the
    honest version for a closed loop at steady state, where the window is set by the run
    length rather than drawn from anything, and it is narrower than it would be if run to
    run variation were included; the published tables say which GPU and how long.
    """
    if not timings:
        raise ValueError("no requests to summarise")
    start = min(t.sent_at for t in timings) + warmup_seconds
    measured = [t for t in timings if t.sent_at >= start]
    if not measured:
        raise ValueError("the warm-up discarded every request")
    ok = [t for t in measured if t.ok]
    if not ok:
        raise ValueError("every measured request failed")

    window = max(t.done_at for t in measured) - min(t.sent_at for t in measured)
    if window <= 0:
        raise ValueError("the measured window has no duration")

    completed = [1.0 if t.ok else 0.0 for t in measured]
    tokens = [float(t.output_tokens) if t.ok else 0.0 for t in measured]
    ttft = [(t.first_token_at - t.sent_at) * 1000 for t in ok if t.first_token_at is not None]
    e2e = [(t.done_at - t.sent_at) * 1000 for t in ok]
    if not ttft:
        raise ValueError("no successful request recorded a first token")

    def per_second(data: np.ndarray) -> float:
        return float(data.sum() / window)

    return LoadSummary(
        concurrency=concurrency,
        requests=len(measured),
        errors=len(measured) - len(ok),
        window_seconds=window,
        requests_per_second=_bootstrap(completed, per_second, seed=seed, unit="req/s"),
        output_tokens_per_second=_bootstrap(tokens, per_second, seed=seed, unit="tok/s"),
        ttft_p50_ms=_bootstrap(ttft, _nearest_rank(50), seed=seed, unit="ms"),
        ttft_p99_ms=_bootstrap(ttft, _nearest_rank(99), seed=seed, unit="ms"),
        e2e_p50_ms=_bootstrap(e2e, _nearest_rank(50), seed=seed, unit="ms"),
        e2e_p99_ms=_bootstrap(e2e, _nearest_rank(99), seed=seed, unit="ms"),
    )
