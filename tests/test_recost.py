"""Costs recomputed from a run's own response bytes. The case that made it necessary: GPT-5.6
bills a cache write at 1.25x input, and the gateway this project first pinned priced it as
plain input."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from boundary.config import PriceEntry, PriceList

from smallprint.recost import recost

PRICES = PriceList(
    version=1,
    date=dt.date(2026, 10, 1),
    currency="USD",
    source="test",
    per_million_tokens={
        "openai": {
            "gpt-5.6-sol": PriceEntry(input=4.0, output=20.0, cache_read=0.4, cache_write=5.0)
        },
        "anthropic": {"claude-sonnet-5": PriceEntry(input=2.0, output=10.0, cache_read=0.2)},
    },
)


def write_raw(run_dir: Path, provider: str, bodies: dict[int, dict[str, object]]) -> None:
    raw = run_dir / "raw" / "run"
    raw.mkdir(parents=True)
    with (raw / f"{provider}.jsonl").open("w", encoding="utf-8") as handle:
        for ledger_id, body in bodies.items():
            record = {
                "ledger_id": ledger_id,
                "provider": provider,
                "response": {"status": 200, "body": {"text": json.dumps(body)}},
            }
            handle.write(json.dumps(record) + "\n")


def openai_body(
    prompt: int, written: int, read: int = 0, completion: int = 200
) -> dict[str, object]:
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": written},
        }
    }


def test_a_cache_write_costs_a_quarter_more_than_the_input_it_replaces(tmp_path: Path) -> None:
    """A million-token prompt, all written to the cache: US$5.00 of writes, not US$4.00."""
    write_raw(tmp_path, "openai", {7: openai_body(1_000_000, 1_000_000, completion=0)})
    result = recost(tmp_path, "openai/gpt-5.6-sol", prices=PRICES)
    assert result is not None
    assert result.costs[7] == pytest.approx(5.00)
    assert result.price_list == "2026-10-01"


def test_a_call_with_no_writes_costs_what_it_did(tmp_path: Path) -> None:
    write_raw(tmp_path, "openai", {1: openai_body(10_000, 0, read=4_000)})
    result = recost(tmp_path, "openai/gpt-5.6-sol", prices=PRICES)
    assert result is not None
    expected = 6_000 / 1e6 * 4.0 + 4_000 / 1e6 * 0.4 + 200 / 1e6 * 20.0
    assert result.costs[1] == pytest.approx(expected)


def test_a_rate_the_list_lacks_leaves_the_recorded_cost(tmp_path: Path) -> None:
    """Anthropic's entry here has no write rate, so a call that wrote stays as recorded."""
    body: dict[str, object] = {
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 50}
    }
    write_raw(tmp_path, "anthropic", {3: body})
    assert recost(tmp_path, "anthropic/claude-sonnet-5", prices=PRICES) is None


def test_a_run_without_raw_bytes_or_a_known_vendor_is_not_recosted(tmp_path: Path) -> None:
    assert recost(tmp_path, "openai/gpt-5.6-sol", prices=PRICES) is None
    write_raw(tmp_path, "selfhosted", {1: openai_body(10, 0)})
    assert recost(tmp_path, "selfhosted/2b", prices=PRICES) is None
