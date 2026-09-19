"""What a call costs: the self-hosted price entry and cost per 1,000 extractions.

Frontier cost per call comes from the gateway ledger, which records what each call was
billed. Self-hosted cost per call has no bill to read, so it is derived, and the derivation
is registered with the gateway as that host's price entry so both kinds of cost land in one
ledger computed one way:

    cost per call = GPU-hour rate / (calls per second at measured throughput * 3600 * utilisation)

Utilisation is the share of rented GPU time spent serving at that throughput. At 100% the
GPU does nothing but serve at the measured rate, which no real deployment achieves; at 50%
it is idle or under-loaded half the time and every call carries twice the rent. It is an
input the reader supplies, not a measurement, which is why the break-even is a curve over it.

Every price carries the date it was read and where from. A rate without a date cannot be
checked, and a cost computed from one cannot be either.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

SECONDS_PER_HOUR: Final = 3600.0
#: The average month, 365.25 * 24 / 12. Break-even is stated per month.
HOURS_PER_MONTH: Final = 730.5


class PriceKind(StrEnum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"


class GpuPrice(BaseModel):
    """One GPU-hour rate, as read on one day from one page."""

    model_config = ConfigDict(frozen=True)

    gpu: str
    provider: str
    kind: PriceKind
    usd_per_hour: float = Field(gt=0)
    checked: dt.date
    source: str

    def __str__(self) -> str:
        return (
            f"{self.gpu} {self.kind.value} at {self.provider}, "
            f"US${self.usd_per_hour:.3f}/h, checked {self.checked.isoformat()}"
        )


class PriceEntry(BaseModel):
    """What the gateway is told a self-hosted call costs, and how that was worked out."""

    model_config = ConfigDict(frozen=True)

    host: str
    price: GpuPrice
    requests_per_second: float
    output_tokens_per_second: float
    utilisation: float
    usd_per_call: float
    usd_per_million_output_tokens: float


def _check_utilisation(utilisation: float) -> None:
    if not 0 < utilisation <= 1:
        raise ValueError(f"utilisation must be in (0, 1]; got {utilisation}")


def usd_per_call(usd_per_hour: float, requests_per_second: float, utilisation: float) -> float:
    """Rent divided by calls served in the hour it buys."""
    _check_utilisation(utilisation)
    if usd_per_hour <= 0 or requests_per_second <= 0:
        raise ValueError("the rate and the throughput must both be positive")
    return usd_per_hour / (requests_per_second * SECONDS_PER_HOUR * utilisation)


def price_entry(
    host: str,
    price: GpuPrice,
    *,
    requests_per_second: float,
    output_tokens_per_second: float,
    utilisation: float,
) -> PriceEntry:
    """The price entry the gateway ledger uses for a self-hosted host.

    Per call for the ledger's own arithmetic, and per million output tokens because that is
    the unit frontier prices are quoted in, so the two can be read side by side.
    """
    per_call = usd_per_call(price.usd_per_hour, requests_per_second, utilisation)
    per_million = usd_per_call(price.usd_per_hour, output_tokens_per_second, utilisation) * 1e6
    return PriceEntry(
        host=host,
        price=price,
        requests_per_second=requests_per_second,
        output_tokens_per_second=output_tokens_per_second,
        utilisation=utilisation,
        usd_per_call=per_call,
        usd_per_million_output_tokens=per_million,
    )


def usd_per_thousand(usd_per_call: float) -> float:
    """Cost per 1,000 extractions, the unit the results table reports."""
    return usd_per_call * 1000
