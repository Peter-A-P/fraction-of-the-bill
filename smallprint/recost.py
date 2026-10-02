"""A run's cost recomputed from the vendor's own bytes, under the gateway this project pins.

The ledger rows and each prediction's `cost_usd` are what the gateway wrote at the time, and
they stay as written. They can be wrong without being edited: boundary v0.3.0 read only
`cached_tokens` from an OpenAI response and priced GPT-5.6's cache writes, which OpenAI bills
at 1.25x input, as ordinary input. Every OpenAI call this project made wrote its whole prompt
to the cache, so every OpenAI run was under-costed by 16 to 22% (04's invoice check of
September found it against OpenAI's console).

Each run keeps the response bytes the vendor returned (`raw/`, pass-through mode), so the
published cost can be recomputed rather than estimated or patched: the pinned gateway's own
adapter reads the usage from the bytes, and its own cost function prices it at its latest
packaged price list. A run with no raw bytes, or a provider the gateway has no adapter here
for, keeps its recorded cost.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from boundary.config import PACKAGED_PRICES, PriceList, latest_price_list
from boundary.ledger.prices import cost_usd
from boundary.providers.anthropic import AnthropicAdapter
from boundary.providers.openai_compat import OpenAICompatAdapter
from boundary.types import Usage


class _ReadsUsage(Protocol):
    def parse_usage(self, raw: Any) -> Usage: ...


#: The vendors this project called, by the provider name the raw store records.
ADAPTERS: Mapping[str, _ReadsUsage] = {
    "openai": OpenAICompatAdapter(),
    "anthropic": AnthropicAdapter(),
}


@dataclass(frozen=True)
class Recost:
    """Costs by ledger row, and the price list and gateway they were computed under."""

    costs: Mapping[int, float]
    price_list: str

    @property
    def basis(self) -> str:
        return f"recomputed from the response bytes at price list {self.price_list}"


def recost(run_dir: Path, model: str, *, prices: PriceList | None = None) -> Recost | None:
    """Every call of a run with raw bytes, costed from those bytes. None when nothing can be.

    `model` is the run's `provider/model-id`. A call whose usage needs a rate the list does
    not carry is left out, so it keeps its recorded cost rather than gaining a guessed one.
    """
    provider, _, model_id = model.partition("/")
    adapter = ADAPTERS.get(provider)
    raw_dir = run_dir / "raw"
    if adapter is None or not raw_dir.is_dir():
        return None
    price_list = prices if prices is not None else latest_price_list(PACKAGED_PRICES)
    entry = price_list.per_million_tokens.get(provider, {}).get(model_id)
    if entry is None:
        return None
    costs: dict[int, float] = {}
    for path in sorted(raw_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("provider") != provider:
                    continue
                body = record["response"]["body"]
                text = body.get("text") if isinstance(body, dict) else None
                if not text:
                    continue
                try:
                    usage = adapter.parse_usage(json.loads(text))
                except json.JSONDecodeError:
                    continue
                cost = cost_usd(usage, entry)
                if cost is not None:
                    costs[int(record["ledger_id"])] = cost
    return Recost(costs=costs, price_list=price_list.name) if costs else None
