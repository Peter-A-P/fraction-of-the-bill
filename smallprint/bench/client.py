"""The closed-loop load client: N requests in flight, the next sent as each one returns.

Closed rather than open loop because the question is what one card sustains, and a closed
loop at concurrency N is how a pool of N workers in a real deployment behaves: nobody sends
request N+1 until a worker is free. An open loop at a fixed arrival rate measures a queue
once the rate passes capacity, and the queue's length is a property of the test, not the
server.

**Through the gateway, streamed.** Every call goes through the gateway so its cost lands in
the ledger like every other call, and it is streamed because streaming is where the gateway
records time to first token. The gateway refuses to stream in pass-through mode, so the
load test runs in standard mode, and a load-test configuration sets `retry.max_attempts`
to 1: a retried request's latency includes the retry, and a load test with retries on is
measuring the retry policy. The client counts retries in every result so a run where that
was not so says so rather than hiding it inside a percentile.

**The real prompt distribution.** The requests are built from test filings with the same
prompt every accuracy number uses, cycled in order, so the latency measured is the latency
of this task and not of a synthetic prompt of some convenient length.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from boundary import ChatRequest, ChatResponse, Mode
from boundary.errors import ProviderError
from pydantic import BaseModel, ConfigDict

from smallprint.bench.load import LoadSummary, RequestTiming, summarise
from smallprint.data.build import SplitItem
from smallprint.prompts import Prompt


class StreamCaller(Protocol):
    """What the load client needs from the gateway. `Gateway` satisfies it."""

    async def achat_stream(
        self,
        request: ChatRequest,
        *,
        purpose: str,
        run_id: str | None = ...,
        mode: Mode = ...,
    ) -> ChatResponse: ...


class LevelResult(BaseModel):
    """One concurrency level: the raw timings, what they summarise to, and the caveats."""

    model_config = ConfigDict(frozen=True)

    concurrency: int
    timings: tuple[RequestTiming, ...]
    #: Requests the gateway retried. Should be zero; see the module docstring.
    retried: int
    #: The most requests ever in flight at once. Equal to `concurrency` in a correct run,
    #: and the test that the loop is closed rather than serial.
    max_in_flight: int
    summary: LoadSummary | None


def requests_for(
    items: Sequence[SplitItem],
    prompt: Prompt,
    model: str,
    *,
    max_tokens: int = 1024,
    temperature: float | None = 0.0,
) -> list[ChatRequest]:
    """One request per filing, in the shape every accuracy run sends.

    At temperature 0, as the fine-tunes' accuracy runs are, because the answers timed should
    be the answers graded: sampling changes how long an answer is, and the time to write it.
    """
    if not items:
        raise ValueError("a load test needs at least one filing to send")
    return [
        ChatRequest(
            model=model,
            messages=prompt.messages(s.item.text),
            system=prompt.system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in items
    ]


async def closed_loop(
    caller: StreamCaller,
    requests: Sequence[ChatRequest],
    *,
    concurrency: int,
    total: int,
    purpose: str,
    run_id: str,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[RequestTiming], int, int]:
    """Send `total` requests with `concurrency` in flight. Returns timings, retries, peak."""
    if concurrency < 1 or total < 1:
        raise ValueError("concurrency and total must both be at least one")
    if not requests:
        raise ValueError("no requests to send")
    timings: list[RequestTiming] = []
    counters = {"next": 0, "in_flight": 0, "peak": 0, "retried": 0}

    async def worker() -> None:
        while counters["next"] < total:
            index = counters["next"]
            counters["next"] += 1
            request = requests[index % len(requests)]
            counters["in_flight"] += 1
            counters["peak"] = max(counters["peak"], counters["in_flight"])
            sent = clock()
            response: ChatResponse | None
            try:
                response = await caller.achat_stream(
                    request, purpose="load:" + purpose, run_id=run_id, mode=Mode.STANDARD
                )
            except ProviderError:
                # A connection the server dropped is a failed request, counted against its
                # level, not the end of the load test: the first on the card, 2026-09-26,
                # lost three finished levels to one ReadError in the fourth.
                response = None
            finally:
                counters["in_flight"] -= 1
            done = clock()
            if response is None:
                timings.append(
                    RequestTiming(
                        sent_at=sent,
                        first_token_at=None,
                        done_at=done,
                        output_tokens=0,
                        ok=False,
                        input_tokens=0,
                    )
                )
                continue
            counters["retried"] += 1 if response.retries else 0
            first = sent + response.ttft_ms / 1000 if response.ttft_ms is not None else None
            timings.append(
                RequestTiming(
                    sent_at=sent,
                    first_token_at=first,
                    done_at=done,
                    output_tokens=response.usage.output_tokens,
                    ok=response.ok,
                    input_tokens=response.usage.input_tokens,
                )
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return timings, counters["retried"], counters["peak"]


def run_level(
    caller: StreamCaller,
    requests: Sequence[ChatRequest],
    *,
    concurrency: int,
    total: int,
    purpose: str,
    run_id: str,
    warmup_seconds: float = 0.0,
    runner: asyncio.Runner | None = None,
) -> LevelResult:
    """One concurrency level, run to completion and summarised.

    `runner` is the event loop the level runs in. A sweep passes one for all its levels:
    the gateway's HTTP client keeps its connections in the loop that first used them, and
    a level run in a fresh loop reused them from a closed one, "Event loop is closed", on
    the first load test on the card (2026-09-26).
    """
    run = runner.run if runner is not None else asyncio.run
    timings, retried, peak = run(
        closed_loop(
            caller,
            requests,
            concurrency=concurrency,
            total=total,
            purpose=purpose,
            run_id=run_id,
        )
    )
    try:
        summary: LoadSummary | None = summarise(
            timings, concurrency=concurrency, warmup_seconds=warmup_seconds
        )
    except ValueError:
        # Every request failed, or the warm-up took them all. The timings are kept so the
        # failure can be read; there is simply nothing to summarise.
        summary = None
    return LevelResult(
        concurrency=concurrency,
        timings=tuple(timings),
        retried=retried,
        max_in_flight=peak,
        summary=summary,
    )
