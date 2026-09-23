"""A load test, whole: every concurrency level for one served model on one card, on disk.

`client.run_level` measures one level; this runs the sweep the plan names, 1, 8, 32 and 64
in flight, and writes it as one record with the card and its dated price beside the
timings. The raw timings are kept, not just the summaries, so a reader can recompute any
percentile, and the price travels with them because a throughput without the rent it was
bought at is not a cost.

Everything published about serving is read back from these records: the serving columns
of the results, the self-hosted price overlay the gateway costs calls with, and the
self-hosted points of the Pareto chart.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from boundary import ChatRequest
from pydantic import BaseModel, ConfigDict

from smallprint.bench.client import LevelResult, StreamCaller, run_level
from smallprint.bench.cost import GpuPrice

RECORD: Final = "bench.json"

#: Requests per level: enough for a p99 that is not simply the maximum, and at least ten
#: rounds of the loop at the highest concurrency, so the window is mostly steady state.
MIN_REQUESTS: Final = 200
ROUNDS: Final = 10


class LoadRun(BaseModel):
    """One served model on one card, every level of the sweep."""

    model_config = ConfigDict(frozen=True)

    #: As the gateway was called: `selfhosted/<run>-<format>`.
    model: str
    price: GpuPrice
    warmup_seconds: float
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    levels: tuple[LevelResult, ...] = ()

    def level(self, concurrency: int) -> LevelResult | None:
        return next((lv for lv in self.levels if lv.concurrency == concurrency), None)

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / RECORD
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return path


def read(out_dir: Path) -> LoadRun:
    return LoadRun.model_validate_json((out_dir / RECORD).read_text(encoding="utf-8"))


def requests_per_level(concurrency: int) -> int:
    return max(MIN_REQUESTS, ROUNDS * concurrency)


def sweep(
    caller: StreamCaller,
    requests: Sequence[ChatRequest],
    *,
    model: str,
    price: GpuPrice,
    levels: Sequence[int],
    out_dir: Path,
    warmup_seconds: float = 30.0,
    total: Callable[[int], int] = requests_per_level,
    on_level: Callable[[LevelResult], None] | None = None,
) -> LoadRun:
    """Every level in turn, lowest first, the record rewritten after each.

    Rewritten after each level so a pod lost at concurrency 64 keeps the three levels it
    finished. A level in which the gateway retried anything is kept and flagged by its own
    `retried` count; the report refuses to publish from it.
    """
    if not levels:
        raise ValueError("no concurrency levels")
    if any(r.model != model for r in requests):
        raise ValueError(f"every request must be for {model}")
    record = LoadRun(
        model=model,
        price=price,
        warmup_seconds=warmup_seconds,
        started_at=dt.datetime.now(dt.UTC),
    )
    record.write(out_dir)
    for concurrency in sorted(levels):
        result = run_level(
            caller,
            requests,
            concurrency=concurrency,
            total=total(concurrency),
            purpose=model,
            run_id=f"load-{model.rsplit('/', 1)[-1]}-c{concurrency}",
            warmup_seconds=warmup_seconds,
        )
        record = record.model_copy(update={"levels": (*record.levels, result)})
        record.write(out_dir)
        if on_level is not None:
            on_level(result)
    record = record.model_copy(update={"finished_at": dt.datetime.now(dt.UTC)})
    record.write(out_dir)
    return record


def mean_tokens(result: LevelResult, *, warmup_seconds: float) -> tuple[float, float]:
    """Mean input and output tokens of the successful requests the summary counted."""
    if not result.timings:
        raise ValueError("no requests")
    start = min(t.sent_at for t in result.timings) + warmup_seconds
    counted = [t for t in result.timings if t.ok and t.sent_at >= start]
    if not counted:
        raise ValueError("no successful requests after the warm-up")
    if any(t.input_tokens <= 0 for t in counted):
        raise ValueError("input tokens were not recorded for every request")
    n = len(counted)
    return sum(t.input_tokens for t in counted) / n, sum(t.output_tokens for t in counted) / n
