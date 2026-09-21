"""The self-hosted price overlay, checked against the gateway itself rather than against a
description of it: the file is written, a real Gateway is built on it, and the gateway's own
validation decides whether it is acceptable."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from boundary import ConfigError, Gateway

from smallprint.bench.cost import GpuPrice, PriceKind
from smallprint.serve import overlay

PRICE = GpuPrice(
    gpu="RTX 4090",
    provider="runpod",
    kind=PriceKind.ON_DEMAND,
    usd_per_hour=0.74,
    checked=dt.date(2026, 9, 21),
    source="https://www.runpod.io/pricing",
)
DAY = dt.date(2026, 9, 21)


def entry(**changes: float) -> overlay.OverlayRates:
    arguments: dict[str, float] = {
        "requests_per_second": 4.0,
        "utilisation": 0.5,
        "mean_input_tokens": 3300,
        "mean_output_tokens": 300,
        "prefill": 0.2,
    }
    arguments.update(changes)
    return overlay.rates(
        "selfhosted",
        "smallprint-4b",
        PRICE,
        requests_per_second=arguments["requests_per_second"],
        utilisation=arguments["utilisation"],
        mean_input_tokens=arguments["mean_input_tokens"],
        mean_output_tokens=arguments["mean_output_tokens"],
        prefill=arguments["prefill"],
    )


def test_a_mean_request_costs_exactly_the_rent_per_call() -> None:
    """The invariant the whole overlay rests on."""
    rates = entry()
    rent_per_call = 0.74 / (4.0 * 3600 * 0.5)
    assert rates.usd_per_call == pytest.approx(rent_per_call)
    assert rates.call_cost(3300, 300) == pytest.approx(rent_per_call)


def test_each_kind_of_token_is_charged_for_the_gpu_time_it_used() -> None:
    rates = entry(prefill=0.2)
    input_share = rates.input_per_million * 3300 / 1e6 / rates.usd_per_call
    assert input_share == pytest.approx(0.2)
    # Decoding is where the time goes, so an output token costs far more than an input one.
    assert rates.output_per_million > 10 * rates.input_per_million


def test_all_prefill_or_all_decode_are_the_limits_not_errors() -> None:
    assert entry(prefill=0.0).input_per_million == 0.0
    assert entry(prefill=1.0).output_per_million == 0.0


def test_half_the_utilisation_doubles_the_price_of_every_token() -> None:
    full, half = entry(utilisation=1.0), entry(utilisation=0.5)
    assert half.input_per_million == pytest.approx(2 * full.input_per_million)
    assert half.output_per_million == pytest.approx(2 * full.output_per_million)


@pytest.mark.parametrize("bad", [{"prefill": 1.5}, {"mean_output_tokens": 0.0}])
def test_rates_from_impossible_measurements_are_refused(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        entry(**bad)


def test_the_source_carries_the_rate_its_date_and_how_it_was_derived() -> None:
    source = entry().source()
    for needed in ("RTX 4090", "runpod", "0.74", "2026-09-21", "4 req/s", "50%", "0.200"):
        assert needed in source


def configure(tmp_path: Path, *, flagged: bool) -> Path:
    """A gateway configuration with one self-hosted provider and an overlay directory."""
    (tmp_path / "caps.yaml").write_text(
        "version: 1\nportfolio_monthly_usd: 400\n"
        "projects:\n  fraction-of-the-bill:\n    monthly_usd: 100\n",
        encoding="utf-8",
    )
    flag = "    self_hosted: true\n" if flagged else ""
    (tmp_path / "boundary.yaml").write_text(
        "version: 1\nprices: builtin\ncaps: caps.yaml\nself_hosted_prices: prices\n"
        f"ledger:\n  path: {(tmp_path / 'ledger.sqlite').as_posix()}\n"
        "providers:\n  selfhosted:\n    kind: openai_compat\n"
        "    base_url: http://gpu.example.internal:8000/v1\n" + flag + "routes: {}\n",
        encoding="utf-8",
    )
    return tmp_path / "boundary.yaml"


def test_the_gateway_accepts_the_overlay_and_prices_the_model_from_it(tmp_path: Path) -> None:
    path = overlay.write(tmp_path / "prices", [entry()], date=DAY)
    assert path.name == "2026-09-21.yaml"
    gateway = Gateway.from_config(configure(tmp_path, flagged=True), project="fraction-of-the-bill")
    try:
        listing = gateway.self_hosted_prices
        assert listing is not None
        rate = listing.lookup("selfhosted", "smallprint-4b")
        assert rate is not None
        assert rate.output == pytest.approx(entry().output_per_million)
        assert listing.name == "2026-09-21"
    finally:
        gateway.close()


def test_the_gateway_refuses_an_overlay_for_a_host_not_flagged_self_hosted(
    tmp_path: Path,
) -> None:
    """Vendor rates have one copy. An overlay naming an unflagged provider would be a second."""
    overlay.write(tmp_path / "prices", [entry()], date=DAY)
    with pytest.raises(ConfigError, match="self_hosted"):
        Gateway.from_config(configure(tmp_path, flagged=False), project="fraction-of-the-bill")


def test_a_dated_overlay_is_never_rewritten_under_the_same_date(tmp_path: Path) -> None:
    """A row already costed from it would cite a date whose rates changed underneath it."""
    overlay.write(tmp_path, [entry()], date=DAY)
    with pytest.raises(FileExistsError, match="new date"):
        overlay.write(tmp_path, [entry(requests_per_second=8.0)], date=DAY)
