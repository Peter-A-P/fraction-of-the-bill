"""The load test on disk, the price overlay made from it, the serving and break-even tables
and the Pareto chart. Every one is arithmetic on records, so every one is checked here
against records whose answers can be worked out by hand."""

from __future__ import annotations

import asyncio
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from boundary import ChatRequest, ChatResponse
from test_baseline import RIGHT, WRONG, items
from test_formats import make_run
from test_serving import SlowServer, some_requests
from typer.testing import CliRunner

from smallprint import chart, report
from smallprint.bench import run as bench
from smallprint.bench.client import LevelResult
from smallprint.bench.cost import GpuPrice, PriceKind
from smallprint.bench.load import LoadSummary, Measured, RequestTiming
from smallprint.cli import app
from smallprint.serve import launch

# Wide, so a refusal is not wrapped mid-phrase by the box the error is printed in.
runner = CliRunner(env={"COLUMNS": "240"})

PRICE = GpuPrice(
    gpu="RTX 4090",
    provider="Runpod",
    kind=PriceKind.ON_DEMAND,
    usd_per_hour=0.36,
    checked=dt.date(2026, 9, 23),
    source="console",
)


def measured(point: float, unit: str = "req/s", spread: float = 0.1) -> Measured:
    return Measured(
        point=point, low=point * (1 - spread), high=point * (1 + spread), n=400, unit=unit
    )


def level(concurrency: int, rps: float, *, retried: int = 0) -> LevelResult:
    timings = tuple(
        RequestTiming(
            sent_at=float(i),
            first_token_at=i + 0.2,
            done_at=i + 1.0,
            output_tokens=200,
            ok=True,
            input_tokens=3000,
        )
        for i in range(10)
    )
    return LevelResult(
        concurrency=concurrency,
        timings=timings,
        retried=retried,
        max_in_flight=concurrency,
        summary=LoadSummary(
            concurrency=concurrency,
            requests=400,
            errors=0,
            window_seconds=400 / rps,
            requests_per_second=measured(rps),
            output_tokens_per_second=measured(rps * 200, "tok/s"),
            ttft_p50_ms=measured(200.0, "ms"),
            ttft_p99_ms=measured(900.0, "ms"),
            e2e_p50_ms=measured(1000.0, "ms"),
            e2e_p99_ms=measured(3000.0, "ms"),
        ),
    )


def load_run(model: str, rps: float = 10.0, *, retried: int = 0) -> bench.LoadRun:
    return bench.LoadRun(
        model=model,
        price=PRICE,
        warmup_seconds=0.0,
        started_at=dt.datetime(2026, 9, 23, tzinfo=dt.UTC),
        levels=(level(1, rps / 8), level(32, rps, retried=retried)),
    )


# -- the load test on disk ------------------------------------------------------------------


def test_the_sweep_runs_every_level_lowest_first_and_keeps_the_timings(tmp_path: Path) -> None:
    requests = some_requests()
    model = requests[0].model
    record = bench.sweep(
        SlowServer(0.01),
        requests,
        model=model,
        price=PRICE,
        levels=[4, 1],
        out_dir=tmp_path,
        warmup_seconds=0.0,
        total=lambda c: 6 * c,
    )
    assert [lv.concurrency for lv in record.levels] == [1, 4]
    assert [len(lv.timings) for lv in record.levels] == [6, 24]
    again = bench.read(tmp_path)
    assert again.finished_at is not None and again.level(4) == record.level(4)
    # The timings carry the prompt length the overlay divides by.
    assert bench.mean_tokens(again.levels[0], warmup_seconds=0.0) == (3000.0, 300.0)


class LoopBoundServer(SlowServer):
    """Like the gateway's HTTP client: keeps its connections in the first loop that used
    them, and fails in any other."""

    def __init__(self, seconds: float) -> None:
        super().__init__(seconds)
        self.loop: asyncio.AbstractEventLoop | None = None

    async def achat_stream(self, request: ChatRequest, **kwargs: object) -> ChatResponse:
        running = asyncio.get_running_loop()
        self.loop = self.loop or running
        if running is not self.loop:
            raise RuntimeError("Event loop is closed")
        return await super().achat_stream(request, **kwargs)  # type: ignore[arg-type]


def test_every_level_of_a_sweep_runs_in_one_event_loop(tmp_path: Path) -> None:
    """The first load test on the card, 2026-09-26, stopped at its second level: each level
    had its own loop, and the gateway's connections belonged to the first."""
    requests = some_requests()
    record = bench.sweep(
        LoopBoundServer(0.001),
        requests,
        model=requests[0].model,
        price=PRICE,
        levels=[1, 2, 4],
        out_dir=tmp_path,
        warmup_seconds=0.0,
        total=lambda c: 3 * c,
    )
    assert [len(lv.timings) for lv in record.levels] == [3, 6, 12]


def test_a_sweep_is_for_one_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every request"):
        bench.sweep(
            SlowServer(0.01),
            some_requests(),
            model="selfhosted/other-bf16",
            price=PRICE,
            levels=[1],
            out_dir=tmp_path,
        )


def test_a_level_has_enough_requests_for_a_p99_and_ten_rounds() -> None:
    assert bench.requests_per_level(1) == 200
    assert bench.requests_per_level(64) == 640


def test_the_load_test_asks_for_the_answers_the_accuracy_runs_graded() -> None:
    assert {r.temperature for r in some_requests()} == {0.0}


def test_the_served_config_turns_retries_off() -> None:
    base = yaml.safe_load(Path("boundary.yaml").read_text(encoding="utf-8"))
    config = launch.served_config(base, "http://127.0.0.1:8000/v1")
    assert config["retry"]["max_attempts"] == 1


def test_bench_refuses_a_config_that_retries(tmp_path: Path) -> None:
    config = tmp_path / "boundary.yaml"
    config.write_text(Path("boundary.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "bench", "run", "--model", "selfhosted/2b-r16-bf16", "--out", str(tmp_path / "b"),
            "--gpu", "RTX 4090", "--provider", "Runpod", "--kind", "on_demand",
            "--usd-per-hour", "0.36", "--checked", "2026-09-23", "--source", "console",
            "--config", str(config),
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "retries must be off" in result.output


# -- the overlay -----------------------------------------------------------------------------


def test_the_overlay_prices_a_mean_call_at_the_rent_per_call(tmp_path: Path) -> None:
    load_run("selfhosted/2b-r16-bf16").write(tmp_path / "b")
    result = runner.invoke(
        app,
        [
            "bench", "overlay", "--bench", str(tmp_path / "b"), "--out", str(tmp_path / "prices"),
            "--utilisation", "0.5", "--date", "2026-09-23",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    written = yaml.safe_load((tmp_path / "prices" / "2026-09-23.yaml").read_text())
    rates = written["per_million_tokens"]["selfhosted"]["2b-r16-bf16"]
    per_call = 0.36 / (10.0 * 3600 * 0.5)
    mean_call = (rates["input"] * 3000 + rates["output"] * 200) / 1e6
    assert mean_call == pytest.approx(per_call)
    # Prefill is a fifth of the median call, so the prompt carries a fifth of its cost.
    assert rates["input"] * 3000 / 1e6 == pytest.approx(per_call * 0.2)


def test_the_overlay_refuses_a_level_the_gateway_retried(tmp_path: Path) -> None:
    load_run("selfhosted/2b-r16-bf16", retried=3).write(tmp_path / "b")
    result = runner.invoke(
        app,
        [
            "bench", "overlay", "--bench", str(tmp_path / "b"), "--out", str(tmp_path / "p"),
            "--utilisation", "0.5", "--date", "2026-09-23",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "no clean level" in result.output


# -- the serving and break-even tables ------------------------------------------------------


def rows(tmp_path: Path) -> tuple[list[report.Row], list[report.Row]]:
    tuned, frontier = tmp_path / "tuned", tmp_path / "frontier"
    make_run(tuned / "bf16", {}, model="selfhosted/2b-r16-bf16", run_items=items(), cost=0.0)
    make_run(
        tuned / "q4", {0: WRONG}, model="selfhosted/2b-r16-gguf-q4_k_m", run_items=items(), cost=0.0
    )
    make_run(frontier / "luna", {}, model="openai/luna", run_items=items(), cost=0.001)
    make_run(frontier / "sol", {1: RIGHT}, model="openai/sol", run_items=items(), cost=0.03)
    post, _ = report.load(report.run_dirs(tuned), items())
    front, _ = report.load(report.run_dirs(frontier), items())
    return post, front


def test_self_hosted_cost_is_the_rent_over_the_throughput_at_the_quoted_utilisation() -> None:
    cost = report.self_hosted_per_1000(load_run("selfhosted/2b-r16-bf16", rps=10.0))
    assert cost is not None
    # US$0.36 an hour, 10 a second, half the time: 18,000 calls an hour.
    assert cost.point == pytest.approx(0.36 / 18_000 * 1000)
    assert cost.low < cost.point < cost.high  # the fast end of the rate is the cheap end
    assert report.self_hosted_per_1000(load_run("selfhosted/x-bf16", retried=1)) is None


def test_the_serving_table_pairs_each_format_with_its_bf16_and_breaks_even_on_the_cheapest(
    tmp_path: Path,
) -> None:
    post, front = rows(tmp_path)
    anchor = report.cost_anchor(front)
    assert anchor.summary.model == "openai/luna"
    runs = [load_run("selfhosted/2b-r16-bf16"), load_run("selfhosted/2b-r16-gguf-q4_k_m", 20.0)]
    table = report.serving_table(post, runs, anchor).splitlines()
    assert table[2].startswith("| `2b-r16` | bf16 | reference | 10.00 (9.00 to 11.00) |")
    assert table[3].startswith("| `2b-r16` | gguf-q4_k_m | -3.3% (-10.0% to +0.0%) | 20.00")
    # One card-month is 0.36 * 730.5 = US$262.98; at US$0.001 a call that is 263k calls.
    assert table[2].endswith("| 0.3M |")


def test_the_breakeven_table_says_never_where_one_card_costs_more_than_the_api(
    tmp_path: Path,
) -> None:
    _, front = rows(tmp_path)
    anchor = report.cost_anchor(front)
    # 0.001 a second on one card: US$0.1 a call at 100%, dearer than the API at any load.
    slow = report.breakeven_table([load_run("selfhosted/7b-r16-bf16", rps=0.001)], anchor)
    assert slow.splitlines()[2].endswith("| never | never | never | never | never |")


def test_the_chart_points_are_every_costed_run(tmp_path: Path) -> None:
    post, front = rows(tmp_path)
    points = report.pareto_points(front, post, [load_run("selfhosted/2b-r16-bf16")])
    labels = {p.label: p for p in points}
    assert set(labels) == {"luna zero-shot", "sol zero-shot", "2b-r16 bf16"}
    assert labels["2b-r16 bf16"].self_hosted
    assert labels["luna zero-shot"].usd_per_1000 == pytest.approx(1.0)


# -- the Pareto chart -------------------------------------------------------------------------


def point(label: str, cost: float, acc: float, *, self_hosted: bool = False) -> chart.Point:
    return chart.Point(
        label=label,
        self_hosted=self_hosted,
        usd_per_1000=cost,
        accuracy=acc,
        low=acc - 0.005,
        high=acc + 0.005,
    )


def test_the_front_is_the_runs_nothing_beats_on_both_axes() -> None:
    points = [
        point("cheap", 0.01, 0.95, self_hosted=True),
        point("mid", 1.0, 0.96),
        point("dear-worse", 15.0, 0.959),
        point("dear-best", 30.0, 0.969),
        point("tie", 1.0, 0.95),
    ]
    assert [p.label for p in chart.pareto_front(points)] == ["cheap", "mid", "dear-best"]


def test_the_chart_is_valid_svg_with_every_point_and_labels_on_the_front_only() -> None:
    points = [
        point("tuned", 0.02, 0.965, self_hosted=True),
        point("luna", 0.87, 0.962),
        point("sol", 30.43, 0.969),
        point("sonnet", 11.96, 0.961),
    ]
    svg = chart.render(points, caption="a caption & more")
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    tips = [g.find(f"{ns}title") for g in root.iter(f"{ns}g")]
    assert len([t for t in tips if t is not None and "per 1,000" in (t.text or "")]) == 4
    texts = [t.text for t in root.iter(f"{ns}text")]
    assert "tuned" in texts and "sol" in texts
    assert "luna" not in texts and "sonnet" not in texts  # dominated by the tuned model
    assert "prefers-color-scheme: dark" in svg
    assert "US$0.01" in texts and "US$100" in texts  # decades either side of the data


def test_a_point_without_a_cost_cannot_be_placed_on_a_log_axis() -> None:
    with pytest.raises(ValueError, match="positive cost"):
        chart.render([point("free", 0.0, 0.9)], caption="")


def test_the_report_refuses_a_chart_without_the_load_tests(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", "--finetuned", str(tmp_path), "--chart", str(tmp_path / "c.svg")]
    )
    assert result.exit_code != 0
    assert "load tests" in result.output
