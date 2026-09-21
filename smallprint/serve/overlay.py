"""The self-hosted price overlay: a GPU-hour rate turned into the gateway's own units.

The gateway costs every call per million input and per million output tokens, because that
is how vendors bill. A rented GPU bills by the hour. So a self-hosted host's rates are
derived, from the rate paid and the throughput measured, and written into a dated price
file in the gateway's own format. Then a self-hosted call and a frontier call land in one
ledger, costed by one piece of arithmetic, which is the rule this project runs on.

**Splitting one cost into two rates.** The cost of a call is fixed by the rent and the
throughput. How it divides between input and output tokens is not, and the division
matters: a call with a long prompt and a short answer costs a GPU something different
from the reverse. The split here charges each kind of token for the GPU time it used.
Prefill, which reads the prompt, is the time to first token; decoding, which writes the
answer, is the rest. So the input rate carries the prefill share of the cost and the
output rate the decode share, and both are measured by the load test rather than chosen.

The one invariant, tested: a request with the mean input and output token counts costs
exactly what the rent divided by the throughput says a call costs.

**The date and the derivation travel with the rates.** A price file's `source` field names
the card, the provider, the hourly rate, the day it was read, the throughput, the
utilisation and the prefill share. A rate without its derivation cannot be checked, and a
cost table built on one cannot be either.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final

import yaml
from boundary.config import PriceList
from pydantic import BaseModel, ConfigDict, Field

from smallprint.bench.cost import GpuPrice, usd_per_call
from smallprint.bench.load import LoadSummary

MILLION: Final = 1_000_000


class OverlayRates(BaseModel):
    """What one self-hosted model costs per million tokens, and how that was reached."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    price: GpuPrice
    requests_per_second: float = Field(gt=0)
    utilisation: float = Field(gt=0, le=1)
    mean_input_tokens: float = Field(gt=0)
    mean_output_tokens: float = Field(gt=0)
    #: The share of a request's GPU time spent reading the prompt.
    prefill_share: float = Field(ge=0, le=1)
    usd_per_call: float
    input_per_million: float
    output_per_million: float

    def call_cost(self, input_tokens: float, output_tokens: float) -> float:
        """What the gateway will charge a call with these token counts."""
        return (
            self.input_per_million * input_tokens + self.output_per_million * output_tokens
        ) / MILLION

    def source(self) -> str:
        p = self.price
        return (
            f"Measured, not quoted. {p.gpu} {p.kind.value} at {p.provider}, "
            f"US${p.usd_per_hour:g}/h read {p.checked.isoformat()} from {p.source}; "
            f"{self.requests_per_second:g} req/s measured; utilisation {self.utilisation:.0%}; "
            f"prefill share {self.prefill_share:.3f} of GPU time; mean "
            f"{self.mean_input_tokens:.0f} tokens in, {self.mean_output_tokens:.0f} out; "
            f"US${self.usd_per_call:.8f} a call."
        )


def prefill_share(summary: LoadSummary) -> float:
    """Time to first token over end-to-end time, at the median.

    Medians rather than means, so that a handful of requests stuck behind a long batch do
    not decide how every token is priced.
    """
    e2e = summary.e2e_p50_ms.point
    if e2e <= 0:
        raise ValueError("an end-to-end time of zero cannot be split")
    return min(1.0, summary.ttft_p50_ms.point / e2e)


def rates(
    provider: str,
    model: str,
    price: GpuPrice,
    *,
    requests_per_second: float,
    utilisation: float,
    mean_input_tokens: float,
    mean_output_tokens: float,
    prefill: float,
) -> OverlayRates:
    """Per-million rates such that a mean request costs exactly the rent per call."""
    per_call = usd_per_call(price.usd_per_hour, requests_per_second, utilisation)
    if mean_input_tokens <= 0 or mean_output_tokens <= 0:
        raise ValueError("mean token counts must be positive and measured")
    if not 0 <= prefill <= 1:
        raise ValueError(f"prefill share must be in [0, 1]; got {prefill}")
    return OverlayRates(
        provider=provider,
        model=model,
        price=price,
        requests_per_second=requests_per_second,
        utilisation=utilisation,
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        prefill_share=prefill,
        usd_per_call=per_call,
        input_per_million=per_call * prefill / mean_input_tokens * MILLION,
        output_per_million=per_call * (1 - prefill) / mean_output_tokens * MILLION,
    )


def price_list(entries: list[OverlayRates], *, date: dt.date) -> PriceList:
    """The overlay as boundary's own model, so the gateway's validation runs on it here.

    Built through `PriceList` rather than written as a dictionary and hoped about: a field
    the gateway would refuse is refused now, before the file exists.
    """
    if not entries:
        raise ValueError("an overlay with no rates in it")
    per_provider: dict[str, dict[str, dict[str, float]]] = {}
    for entry in entries:
        per_provider.setdefault(entry.provider, {})[entry.model] = {
            "input": entry.input_per_million,
            "output": entry.output_per_million,
        }
    return PriceList.model_validate(
        {
            "version": 1,
            "date": date,
            "currency": "USD",
            "source": " ".join(e.source() for e in entries),
            "per_million_tokens": per_provider,
        }
    )


def write(directory: Path, entries: list[OverlayRates], *, date: dt.date) -> Path:
    """Write the dated overlay file the gateway's `self_hosted_prices` directory holds.

    The file name is the date, which is how the gateway picks the newest; an existing file
    for the same date is refused rather than overwritten, because a ledger row already
    costed from it would then cite a date whose rates had changed underneath it.
    """
    listing = price_list(entries, date=date)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date.isoformat()}.yaml"
    if path.exists():
        raise FileExistsError(
            f"{path} exists; rows may already be costed from it. Write the new rates under "
            "a new date"
        )
    path.write_text(
        yaml.safe_dump(listing.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return path
