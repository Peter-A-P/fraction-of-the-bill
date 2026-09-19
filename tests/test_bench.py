"""Load-test arithmetic, the self-hosted price entry, and the break-even, against known answers.

Nothing here has been measured yet. These tests exist so that when it is, the numbers that
come out are the ones the definitions say, checked against streams and prices whose answers
can be worked out by hand.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from smallprint.bench.breakeven import (
    BreakEvenInputs,
    break_even,
    capacity_per_gpu_month,
    curve,
    self_hosted_monthly,
)
from smallprint.bench.cost import (
    HOURS_PER_MONTH,
    GpuPrice,
    PriceKind,
    price_entry,
    usd_per_call,
    usd_per_thousand,
)
from smallprint.bench.load import RequestTiming, percentile, summarise
from smallprint.cli import app


def gpu(usd_per_hour: float) -> GpuPrice:
    return GpuPrice(
        gpu="L4",
        provider="example",
        kind=PriceKind.SPOT,
        usd_per_hour=usd_per_hour,
        checked=dt.date(2026, 9, 19),
        source="https://example.com/pricing",
    )


# -- percentiles and the load summary ---------------------------------------------------------


def test_nearest_rank_percentiles_on_one_to_a_hundred() -> None:
    values = [float(v) for v in range(100, 0, -1)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0
    assert percentile(values, 1) == 1.0


def test_with_fewer_than_a_hundred_values_p99_is_the_maximum() -> None:
    assert percentile([5.0, 1.0, 9.0, 3.0], 99) == 9.0


def test_a_percentile_is_always_a_value_that_occurred() -> None:
    """Interpolation would give 2.5 here. No request took 2.5."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


@pytest.mark.parametrize("p", [0, -1, 101])
def test_percentiles_outside_the_range_are_refused(p: float) -> None:
    with pytest.raises(ValueError, match="p must be"):
        percentile([1.0], p)


def test_a_percentile_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="undefined"):
        percentile([], 50)


def stream(n: int = 100, *, failures: int = 0) -> list[RequestTiming]:
    """A closed loop at concurrency 1: request i sent at i / 10 s, done 100 ms later, first
    token after i + 1 ms, 50 output tokens. The last `failures` requests fail."""
    out = []
    for i in range(n):
        sent = i / 10
        ok = i < n - failures
        out.append(
            RequestTiming(
                sent_at=sent,
                first_token_at=sent + (i + 1) / 1000 if ok else None,
                done_at=sent + 0.1,
                output_tokens=50 if ok else 0,
                ok=ok,
            )
        )
    return out


def test_a_known_stream_summarises_to_its_known_answers() -> None:
    summary = summarise(stream(), concurrency=1)
    assert summary.window_seconds == pytest.approx(10.0)
    assert summary.requests_per_second.point == pytest.approx(10.0)
    assert summary.output_tokens_per_second.point == pytest.approx(500.0)
    assert summary.ttft_p50_ms.point == pytest.approx(50.0)
    assert summary.ttft_p99_ms.point == pytest.approx(99.0)
    assert summary.e2e_p99_ms.point == pytest.approx(100.0)
    assert summary.errors == 0


def test_a_failed_request_counts_against_errors_and_not_towards_throughput() -> None:
    summary = summarise(stream(failures=10), concurrency=1)
    assert summary.errors == 10
    assert summary.requests_per_second.point == pytest.approx(9.0)
    assert summary.ttft_p99_ms.n == 90


def test_the_warm_up_is_discarded_before_anything_is_computed() -> None:
    summary = summarise(stream(), concurrency=1, warmup_seconds=1.0)
    assert summary.requests == 90
    assert summary.window_seconds == pytest.approx(9.0)
    assert summary.ttft_p50_ms.point == pytest.approx(55.0)


def test_every_figure_carries_an_interval_around_its_point() -> None:
    summary = summarise(stream(failures=7), concurrency=1)
    for measured in (
        summary.requests_per_second,
        summary.output_tokens_per_second,
        summary.ttft_p50_ms,
        summary.ttft_p99_ms,
        summary.e2e_p50_ms,
    ):
        assert measured.low <= measured.point <= measured.high
    assert summary.ttft_p50_ms.low < summary.ttft_p50_ms.high


def test_a_run_where_everything_failed_is_refused_rather_than_summarised() -> None:
    with pytest.raises(ValueError, match="failed"):
        summarise(stream(10, failures=10), concurrency=1)


# -- the price entry -------------------------------------------------------------------------


def test_cost_per_call_is_rent_over_calls_served_in_the_hour() -> None:
    assert usd_per_call(0.80, 2.0, 1.0) == pytest.approx(0.80 / 7200)
    assert usd_per_call(0.80, 2.0, 0.5) == pytest.approx(2 * 0.80 / 7200)


def test_the_price_entry_reproduces_cost_per_call_from_rate_and_throughput() -> None:
    entry = price_entry(
        "smallprint-4b-awq@l4",
        gpu(0.80),
        requests_per_second=2.0,
        output_tokens_per_second=400.0,
        utilisation=0.5,
    )
    # Back out the rate: per-call cost times calls served in an hour at this utilisation.
    assert entry.usd_per_call * 2.0 * 3600 * 0.5 == pytest.approx(0.80)
    assert entry.usd_per_million_output_tokens == pytest.approx(0.80 / (400 * 3600 * 0.5) * 1e6)
    assert usd_per_thousand(entry.usd_per_call) == pytest.approx(entry.usd_per_call * 1000)


@pytest.mark.parametrize("utilisation", [0.0, -0.1, 1.01])
def test_a_utilisation_outside_zero_to_one_is_refused(utilisation: float) -> None:
    with pytest.raises(ValueError, match="utilisation"):
        usd_per_call(0.80, 2.0, utilisation)


def test_a_price_without_its_date_is_not_a_price() -> None:
    with pytest.raises(ValidationError):
        GpuPrice.model_validate(
            {"gpu": "L4", "provider": "x", "kind": "spot", "usd_per_hour": 0.8, "source": "x"}
        )


# -- the break-even --------------------------------------------------------------------------


def inputs(
    *, rate: float = 1.0, rps: float = 1.0, api: float = 0.01, fixed: float = 0.0
) -> BreakEvenInputs:
    return BreakEvenInputs(
        price=gpu(rate), requests_per_second=rps, api_usd_per_call=api, fixed_usd_per_month=fixed
    )


def brute_force(case: BreakEvenInputs, utilisation: float, up_to: int) -> int | None:
    """The first whole volume at which the API costs at least the staircase, by evaluating
    the staircase at every volume up to `up_to`."""
    volume = np.arange(0, up_to + 1, dtype=float)
    capacity = capacity_per_gpu_month(case, utilisation)
    gpus = np.maximum(1, np.ceil(volume / capacity))
    self_hosted = gpus * case.price.usd_per_hour * case.hours_per_month + case.fixed_usd_per_month
    api = volume * case.api_usd_per_call
    hits = np.nonzero(api >= self_hosted - 1e-9)[0]
    return int(hits[0]) if hits.size else None


def test_one_gpu_breaks_even_at_one_month_of_rent_in_api_calls() -> None:
    point = break_even(inputs(), 0.5)
    assert point.gpus_at_break_even == 1
    assert point.volume_per_month == pytest.approx(HOURS_PER_MONTH / 0.01)
    assert point.monthly_cost_at_break_even == pytest.approx(HOURS_PER_MONTH)


def test_without_fixed_costs_the_break_even_does_not_depend_on_utilisation() -> None:
    """One GPU-month of rent, in API calls, whatever the utilisation, as long as one GPU can
    carry that volume. Utilisation decides whether it can, not where the line is."""
    volumes = [p.volume_per_month for p in curve(inputs())]
    assert volumes == [pytest.approx(HOURS_PER_MONTH / 0.01)] * len(volumes)


def test_a_gpu_dearer_per_call_than_the_api_never_breaks_even() -> None:
    point = break_even(inputs(rps=0.001), 0.1)
    assert point.volume_per_month is None
    assert point.self_hosted_usd_per_call > point.api_usd_per_call


def test_a_break_even_many_gpus_out_matches_the_staircase_evaluated_everywhere() -> None:
    """Fixed costs push the break-even to the eighteenth GPU; worked by hand in the module."""
    case = inputs(rps=1.0, api=0.003, fixed=1000.0)
    point = break_even(case, 0.1)
    assert point.gpus_at_break_even == 18
    assert point.volume_per_month is not None
    expected = brute_force(case, 0.1, up_to=5_000_000)
    assert expected is not None
    assert point.volume_per_month == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("fixed", [0.0, 250.0, 5000.0])
@pytest.mark.parametrize("utilisation", [0.1, 0.5, 0.9])
def test_the_break_even_agrees_with_brute_force_across_inputs(
    fixed: float, utilisation: float
) -> None:
    case = inputs(rps=0.5, api=0.002, fixed=fixed)
    point = break_even(case, utilisation)
    expected = brute_force(case, utilisation, up_to=3_000_000)
    if expected is None:
        assert point.volume_per_month is None or point.volume_per_month > 3_000_000
    else:
        assert point.volume_per_month == pytest.approx(expected, abs=1.0)


def test_the_staircase_rents_whole_gpus() -> None:
    case = inputs()
    capacity = capacity_per_gpu_month(case, 0.5)
    one = self_hosted_monthly(case, capacity, 0.5)
    two = self_hosted_monthly(case, capacity + 1, 0.5)
    assert two == pytest.approx(2 * one)


def test_the_curve_is_drawn_from_ten_to_ninety_percent() -> None:
    assert [p.utilisation for p in curve(inputs())] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


def test_the_breakeven_command_prints_its_inputs_and_the_curve() -> None:
    result = CliRunner().invoke(
        app,
        [
            "breakeven",
            "--gpu", "L4",
            "--provider", "example",
            "--kind", "spot",
            "--usd-per-hour", "0.80",
            "--checked", "2026-09-19",
            "--source", "https://example.com/pricing",
            "--requests-per-second", "0.05",
            "--api-usd-per-call", "0.005",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "checked 2026-09-19" in result.output
    assert "never" in result.output  # at 10%, one L4 at 0.05 req/s costs more per call
    assert "90%" in result.output


def test_the_breakeven_command_refuses_a_price_without_a_date() -> None:
    result = CliRunner().invoke(
        app,
        [
            "breakeven",
            "--gpu", "L4",
            "--provider", "example",
            "--kind", "spot",
            "--usd-per-hour", "0.80",
            "--source", "x",
            "--requests-per-second", "1",
            "--api-usd-per-call", "0.005",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
