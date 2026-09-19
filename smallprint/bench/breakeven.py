"""The break-even: the monthly volume at which owning the model stops costing more than renting it.

The API is paid per call, so its monthly cost is volume times the ledger's cost per call.
A GPU is rented whole, by the hour, whether it serves one call or its capacity. So the
self-hosted monthly cost is a staircase: enough whole GPUs for the volume at the planned
utilisation, times the rate, times the hours in a month, plus any fixed cost the reader
wants to count (an engineer's time, storage, a load balancer), which defaults to zero and
is printed either way.

    GPUs needed    k(V) = max(1, ceil(V / capacity)),  capacity = rps * 3600 * H * u
    self-hosted    S(V) = k(V) * rate * H + fixed
    API            A(V) = V * api cost per call

The break-even volume is the smallest V with A(V) >= S(V). Within the band served by k
GPUs, A rises and S is flat, so the answer in that band is (k * rate * H + fixed) / api
cost, provided that is within the band. Bands are tried in order. If one GPU at the planned
utilisation already costs more per call than the API does, no volume breaks even and the
answer is None, which is printed as such rather than as a very large number.

Past the break-even the staircase can briefly put self-hosting back above the API: the
first calls that need another GPU pay its whole month. The published point is where
self-hosting first stops costing more, and `self_hosted_monthly` is there for any reader
who wants the cost at a particular volume instead.

Utilisation is the reader's choice, not a measurement: how hard they are willing to run a
GPU given their latency target and the shape of their traffic. So the break-even is a curve
over utilisation, from 10% to 90%, with every input in the table beside it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.bench.cost import HOURS_PER_MONTH, SECONDS_PER_HOUR, GpuPrice, usd_per_call

#: The utilisations the published curve is drawn at.
UTILISATIONS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class BreakEvenPoint(BaseModel):
    """One point of the curve."""

    model_config = ConfigDict(frozen=True)

    utilisation: float
    self_hosted_usd_per_call: float
    api_usd_per_call: float
    #: Extractions per month at which self-hosting is first no dearer. None if never.
    volume_per_month: float | None
    gpus_at_break_even: int | None
    monthly_cost_at_break_even: float | None


class BreakEvenInputs(BaseModel):
    """Everything the curve depends on, so a reader can substitute their own."""

    model_config = ConfigDict(frozen=True)

    price: GpuPrice
    requests_per_second: float
    api_usd_per_call: float
    fixed_usd_per_month: float = 0.0
    hours_per_month: float = HOURS_PER_MONTH


def capacity_per_gpu_month(inputs: BreakEvenInputs, utilisation: float) -> float:
    """Extractions one GPU serves in a month at the planned utilisation."""
    return inputs.requests_per_second * SECONDS_PER_HOUR * inputs.hours_per_month * utilisation


def self_hosted_monthly(inputs: BreakEvenInputs, volume: float, utilisation: float) -> float:
    """The staircase: whole GPUs for the volume, rented all month, plus the fixed cost."""
    if volume < 0:
        raise ValueError("volume cannot be negative")
    gpus = max(1, math.ceil(volume / capacity_per_gpu_month(inputs, utilisation)))
    return gpus * inputs.price.usd_per_hour * inputs.hours_per_month + inputs.fixed_usd_per_month


def break_even(inputs: BreakEvenInputs, utilisation: float) -> BreakEvenPoint:
    """The smallest monthly volume at which self-hosting costs no more than the API."""
    per_call = usd_per_call(inputs.price.usd_per_hour, inputs.requests_per_second, utilisation)
    api = inputs.api_usd_per_call
    if api <= 0:
        raise ValueError("the API cost per call must be positive")
    point = BreakEvenPoint(
        utilisation=utilisation,
        self_hosted_usd_per_call=per_call,
        api_usd_per_call=api,
        volume_per_month=None,
        gpus_at_break_even=None,
        monthly_cost_at_break_even=None,
    )
    capacity = capacity_per_gpu_month(inputs, utilisation)
    gpu_month = inputs.price.usd_per_hour * inputs.hours_per_month
    # Past the first band each extra GPU adds gpu_month of cost for capacity * api of
    # revenue saved. If that is a loss, a band that does not break even is followed only by
    # worse ones, so the search can stop there.
    gpus = 1
    while True:
        needed = (gpus * gpu_month + inputs.fixed_usd_per_month) / api
        band_low = 0.0 if gpus == 1 else (gpus - 1) * capacity
        if needed <= gpus * capacity:
            volume = max(needed, band_low)
            return point.model_copy(
                update={
                    "volume_per_month": volume,
                    "gpus_at_break_even": gpus,
                    "monthly_cost_at_break_even": self_hosted_monthly(inputs, volume, utilisation),
                }
            )
        if per_call >= api:
            return point
        gpus += 1


def curve(
    inputs: BreakEvenInputs, utilisations: Sequence[float] = UTILISATIONS
) -> tuple[BreakEvenPoint, ...]:
    """The break-even at each utilisation."""
    return tuple(break_even(inputs, u) for u in utilisations)
