"""The website's data: one JSON file the page reads, written from the same runs as the report.

The website (`site/`) is a static page served on its own host. It holds no numbers of its
own: every figure on it comes out of `site/results.json`, and that file comes out of here,
from the same rows, load tests and break-even arithmetic as the README's tables. So the
page cannot drift from the repository any more than the tables can.

The runs themselves are not in the repository (they are gigabytes of answers and ledger),
so the file is written on the machine that has them and committed, and the deploy only
uploads `site/` as it is.

The page's calculator does the break-even again in the browser, with the reader's own
prices, in `site/breakeven.js`. The curves here are the Python answer at the published
inputs, and `tests/test_site.py` runs the script against the Python, so the two cannot disagree
on the numbers the page prints by default.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from smallprint.baseline import read_split
from smallprint.bench.breakeven import BreakEvenPoint, break_even
from smallprint.bench.cost import HOURS_PER_MONTH, SECONDS_PER_HOUR
from smallprint.bench.run import LoadRun
from smallprint.data.split import Split
from smallprint.grade import Interval
from smallprint.report import (
    QUOTED_UTILISATION,
    SERVING_CONCURRENCY,
    TABLE_UTILISATIONS,
    Row,
    best_frontier,
    breakeven_inputs,
    cost_anchor,
    load,
    measured_level,
    read_loads,
    run_dirs,
    self_hosted_per_1000,
)
from smallprint.serve.launch import parse_served_name

#: Where the page reads its numbers from, relative to the site directory.
RESULTS: Final = "results.json"

#: The host's configuration: the headers, the content security policy among them.
HOST_CONFIG: Final = "staticwebapp.config.json"

#: A suite's row in a gate decision: `| field | items | ... | verdict |`.
_SUITE: Final = re.compile(r"^\| (?P<field>[a-z_]+) \| \d+ \|.*\| (?P<verdict>pass|block|warn) \|$")
_HEADLINE: Final = re.compile(r"^## Gate: (?P<verdict>[A-Z]+)")


class Share(BaseModel):
    """A proportion and its 95% interval."""

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float

    @classmethod
    def of(cls, interval: Interval) -> Share:
        return cls(point=interval.point, low=interval.low, high=interval.high)


class Frontier(BaseModel):
    """One frontier API run: what it got right and what it was billed."""

    model_config = ConfigDict(frozen=True)

    key: str
    model: str
    prompt: str
    accuracy: Share
    every_field: Share
    usd_per_1000: float
    #: True when the cost is recomputed from the run's bytes rather than the ledger as written.
    recosted: bool
    latency_p50_s: float


class Served(BaseModel):
    """One fine-tune in one format on one rented card."""

    model_config = ConfigDict(frozen=True)

    name: str
    size: str
    format: str
    gpu: str
    provider: str
    usd_per_hour: float
    price_checked: dt.date
    requests_per_second: float
    ttft_p99_ms: float
    #: None where the format's answers were not graded on the post-cutoff set.
    accuracy: Share | None
    every_field: Share | None
    usd_per_1000: float


class Curve(BaseModel):
    """The break-even of one served model against one frontier run, across utilisation."""

    model_config = ConfigDict(frozen=True)

    served: int
    against: str
    points: tuple[BreakEvenPoint, ...]


class Gate(BaseModel):
    """The release gate's decision on one fine-tune against one frontier run."""

    model_config = ConfigDict(frozen=True)

    candidate: str
    baseline: str
    verdict: str
    suites: int
    blocked: tuple[str, ...]


class SiteData(BaseModel):
    """Everything the page shows."""

    model_config = ConfigDict(frozen=True)

    written: dt.date
    filings: int
    hours_per_month: float
    seconds_per_hour: float
    concurrency: int
    quoted_utilisation: float
    anchor: str
    ceiling: str
    frontier: tuple[Frontier, ...]
    served: tuple[Served, ...]
    curves: tuple[Curve, ...]
    gates: tuple[Gate, ...]


def _frontier(row: Row) -> Frontier:
    s = row.summary
    return Frontier(
        key=row.key,
        model=s.model,
        prompt=s.style.value.replace("_", "-"),
        accuracy=Share.of(s.accuracy),
        every_field=Share.of(s.exact_match),
        usd_per_1000=row.usd_per_1000,
        recosted=s.cost_basis is not None,
        latency_p50_s=s.latency_p50_ms.point / 1000,
    )


def _served(run: LoadRun, post: Sequence[Row]) -> Served | None:
    summary = measured_level(run, SERVING_CONCURRENCY)
    cost = self_hosted_per_1000(run)
    if summary is None or cost is None:
        return None
    size, name, fmt = parse_served_name(run.model)
    row = next((r for r in post if r.summary.model == run.model), None)
    return Served(
        name=name,
        size=size,
        format=fmt.value,
        gpu=run.price.gpu,
        provider=run.price.provider,
        usd_per_hour=run.price.usd_per_hour,
        price_checked=run.price.checked,
        requests_per_second=summary.requests_per_second.point,
        ttft_p99_ms=summary.ttft_p99_ms.point,
        accuracy=Share.of(row.summary.accuracy) if row else None,
        every_field=Share.of(row.summary.exact_match) if row else None,
        usd_per_1000=cost.point,
    )


def read_gate(decision: Path) -> Gate:
    """One `decision.txt` the gate wrote: its verdict, and the fields it blocked on."""
    lines = decision.read_text(encoding="utf-8").splitlines()
    headline = next((m for m in map(_HEADLINE.match, lines) if m), None)
    if headline is None:
        raise ValueError(f"{decision} has no gate verdict")
    suites = [m for m in map(_SUITE.match, lines) if m]
    if not suites:
        raise ValueError(f"{decision} has no suites")
    candidate, _, baseline = decision.parent.name.rpartition("-vs-")
    return Gate(
        candidate=candidate,
        baseline=baseline,
        verdict=headline["verdict"].lower(),
        suites=len(suites),
        blocked=tuple(m["field"] for m in suites if m["verdict"] == "block"),
    )


def site_data(
    frontier: Sequence[Row],
    post: Sequence[Row],
    runs: Sequence[LoadRun],
    gates: Sequence[Gate],
    *,
    written: dt.date,
) -> SiteData:
    """The page's numbers from the report's rows: frontier costed runs, every served model
    measured at the quoted concurrency, and each one's break-even against the cost anchor
    and the most accurate frontier run."""
    anchor, ceiling = cost_anchor(frontier), best_frontier(frontier)
    ordered = sorted(runs, key=lambda r: (parse_served_name(r.model)[1:], r.price.usd_per_hour))
    served: list[Served] = []
    curves: list[Curve] = []
    for run in ordered:
        entry = _served(run, post)
        if entry is None:
            continue
        for against in [anchor] if ceiling is anchor else [anchor, ceiling]:
            inputs = breakeven_inputs(run, against)
            if inputs is not None:
                points = tuple(break_even(inputs, u) for u in TABLE_UTILISATIONS)
                curves.append(Curve(served=len(served), against=against.key, points=points))
        served.append(entry)
    return SiteData(
        written=written,
        filings=anchor.summary.graded,
        hours_per_month=HOURS_PER_MONTH,
        seconds_per_hour=SECONDS_PER_HOUR,
        concurrency=SERVING_CONCURRENCY,
        quoted_utilisation=QUOTED_UTILISATION,
        anchor=anchor.key,
        ceiling=ceiling.key,
        frontier=tuple(_frontier(r) for r in frontier if r.summary.usd_per_1000 is not None),
        served=tuple(served),
        curves=tuple(curves),
        gates=tuple(gates),
    )


def build(
    finetuned_root: Path,
    frontier_root: Path,
    bench_root: Path,
    build_dir: Path,
    gate_root: Path,
    *,
    written: dt.date,
) -> SiteData:
    """Read the runs, the load tests and the gate decisions, as `smallprint report` does."""
    items = read_split(build_dir, Split.TEST_POST_CUTOFF.value)
    post, _ = load(run_dirs(finetuned_root, Split.TEST_POST_CUTOFF.value), items)
    frontier, _ = load(run_dirs(frontier_root, Split.TEST_POST_CUTOFF.value), items)
    gates = [read_gate(d) for d in sorted(gate_root.glob("*/decision.txt"))]
    return site_data(frontier, post, read_loads(bench_root), gates, written=written)


def write(data: SiteData, site_dir: Path) -> Path:
    """`results.json` in the site directory, the only file in it this writes."""
    out = site_dir / RESULTS
    out.write_text(data.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def host_headers(site_dir: Path) -> dict[str, str]:
    """The headers the host sends with every file, from its configuration."""
    config = json.loads((site_dir / HOST_CONFIG).read_text(encoding="utf-8"))
    headers: dict[str, str] = config["globalHeaders"]
    return headers


class _WithHostHeaders(SimpleHTTPRequestHandler):
    """A static file server that sends the host's headers, so the content security policy
    refuses locally whatever it would refuse on the live site."""

    extra: ClassVar[Mapping[str, str]] = {}

    def end_headers(self) -> None:
        for name, value in self.extra.items():
            self.send_header(name, value)
        super().end_headers()


def preview(site_dir: Path, port: int) -> ThreadingHTTPServer:
    """A local server for the site with the host's headers; the caller serves it. Not
    `python -m http.server`, which sends none of them and so shows a page the policy would
    partly refuse (project 08 lost a chart on its live site for two weeks that way)."""
    handler = type("Handler", (_WithHostHeaders,), {"extra": host_headers(site_dir)})
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(site_dir)))
