"""The website's numbers: the results file is the report's arithmetic, the gate decisions are
read as the gate wrote them, and the calculator's JavaScript gives the Python's break-even.

The last is the one that matters most. The page lets a reader put in their own prices, so
the break-even runs in the browser, in a second language; these tests run that file under
Node across a grid that reaches the "never" case and volumes needing more than one card,
and fail on any disagreement with `smallprint.bench.breakeven`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest
from items import training_pool
from test_baseline import WRONG, items
from test_formats import make_run
from test_money import PRICE, load_run, rows

from smallprint import report, site
from smallprint.bench.breakeven import BreakEvenInputs, break_even
from smallprint.bench.cost import GpuPrice
from smallprint.train import dataset
from smallprint.train.qlora import RunRecord
from smallprint.train.recipe import BASES, TrainConfig

SITE = Path(__file__).resolve().parents[1] / "site"

DECISION = """## Gate: BLOCK

Spec `fraction-of-the-bill` (a63a5a76c892744c).

| Suite | Paired items | Baseline | Candidate | Difference | Margin | Holm p | Verdict |
|---|---:|---|---|---|---:|---:|---|
| revenue | 705 | 100.0% (99.6 to 100.0) | 99.6% (99.0 to 100.0) | -0.4% (-1.0 to +0.3) | -3% | 0.000 | pass |
| net_income | 705 | 97.3% (96.0 to 98.4) | 95.3% (93.6 to 96.9) | -2.0% (-3.8 to -0.3) | -3% | 0.226 | block |
| auditor_name | 12 | 99.7% (99.3 to 100.0) | 99.7% (99.3 to 100.0) | +0.0% (-0.4 to +0.4) | -3% | 1.000 | warn |

**Reasons**

- net_income: cannot rule out a 3% drop
"""


def test_a_gate_decision_is_read_for_its_verdict_and_the_fields_it_blocked(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "2b-r64-lr1e-4-nall-s0-e1-vs-luna" / "decision.txt"
    decision.parent.mkdir()
    decision.write_text(DECISION, encoding="utf-8")
    gate = site.read_gate(decision)
    assert gate.candidate == "2b-r64-lr1e-4-nall-s0-e1"
    assert gate.baseline == "luna"
    assert gate.verdict == "block"
    assert gate.suites == 3
    assert gate.blocked == ("net_income",)  # a warning is not a block


def test_a_decision_without_a_verdict_is_refused_rather_than_read_as_a_pass(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "x-vs-luna" / "decision.txt"
    decision.parent.mkdir()
    decision.write_text(DECISION.replace("## Gate: BLOCK", "## Gate"), encoding="utf-8")
    with pytest.raises(ValueError, match="no gate verdict"):
        site.read_gate(decision)


def test_the_site_data_is_the_reports_arithmetic(tmp_path: Path) -> None:
    post, _ = rows(tmp_path)
    # A cheap API that misses one filing and a dear one that misses none, so the cost anchor
    # and the most accurate run are two runs and each served model gets a curve against both.
    apis = tmp_path / "apis"
    make_run(apis / "luna", {0: WRONG}, model="openai/luna", run_items=items(), cost=0.001)
    make_run(apis / "sol", {}, model="openai/sol", run_items=items(), cost=0.03)
    front, _ = report.load(report.run_dirs(apis), items())
    runs = [load_run("selfhosted/2b-r16-bf16"), load_run("selfhosted/2b-r16-gguf-q4_k_m", rps=2.0)]
    data = site.site_data(front, post, runs, [], written=dt.date(2026, 10, 3))
    assert data.anchor == "openai/luna zero_shot"
    assert data.ceiling == "openai/sol zero_shot"
    assert [s.format for s in data.served] == ["bf16", "gguf-q4_k_m"]
    for served, run in zip(data.served, runs, strict=True):
        cost = report.self_hosted_per_1000(run)
        assert cost is not None and served.usd_per_1000 == pytest.approx(cost.point)
    # One curve against the cost anchor and one against the ceiling, for each served model.
    assert [(c.served, c.against) for c in data.curves] == [
        (0, data.anchor),
        (0, data.ceiling),
        (1, data.anchor),
        (1, data.ceiling),
    ]
    anchor = report.cost_anchor(front)
    inputs = report.breakeven_inputs(runs[0], anchor)
    assert inputs is not None
    assert data.curves[0].points[2] == break_even(inputs, report.TABLE_UTILISATIONS[2])


def test_a_served_model_with_no_measurement_at_the_quoted_load_is_left_off(
    tmp_path: Path,
) -> None:
    post, front = rows(tmp_path)
    retried = load_run("selfhosted/2b-r16-bf16", retried=1)
    data = site.site_data(front, post, [retried], [], written=dt.date(2026, 10, 3))
    assert data.served == () and data.curves == ()


def test_the_results_file_round_trips(tmp_path: Path) -> None:
    post, front = rows(tmp_path)
    data = site.site_data(
        front, post, [load_run("selfhosted/2b-r16-bf16")], [], written=dt.date(2026, 10, 3)
    )
    written = site.write(data, tmp_path)
    assert site.SiteData.model_validate_json(written.read_text(encoding="utf-8")) == data


# -- the models and how they were trained ---------------------------------------------------


def sweep_rows(tmp_path: Path) -> list[report.Row]:
    runs: dict[str, dict[int, str | None]] = {
        "2b-r16-lr1e-4-nall-s0-e1": {0: WRONG, 1: WRONG},  # the base recipe
        "2b-r64-lr1e-4-nall-s0-e1": {},  # rank changed, the best: chosen
        "2b-r16-lr2e-4-nall-s0-e1": {0: WRONG},  # learning rate changed
        "2b-r16-lr1e-4-n1000-s0-e1": {0: WRONG, 1: WRONG, 2: WRONG},  # volume changed
        "2b-r64-lr2e-4-nall-s0-e1": {},  # two things changed: not a sweep point
        "2b-r64-lr1e-4-nall-s1-e1": {},  # another seed: not a sweep point
        "2b-r64-lr1e-4-nall-s0-e1-distill-luna": {},  # an ablation: not a sweep point
    }
    for run, answers in runs.items():
        make_run(
            tmp_path / run, answers, model=f"selfhosted/{run}-bf16", run_items=items(), cost=0.0
        )
    make_run(
        tmp_path / "q8",
        {},
        model="selfhosted/2b-r16-lr1e-4-nall-s0-e1-gguf-q8_0",
        run_items=items(),
        cost=0.0,
    )
    return report.load(report.run_dirs(tmp_path), items())[0]


def test_the_sweep_is_the_one_factor_runs_in_bf16_with_the_chosen_one_marked(
    tmp_path: Path,
) -> None:
    points = site.sweep_points(sweep_rows(tmp_path), ["2b-r64-lr1e-4-nall-s0-e1"])
    assert [(p.varied, p.value, p.chosen) for p in points] == [
        ("Base recipe", "rank 16, 1e-4, every filing", False),
        ("Adapter rank", "64", True),
        ("Learning rate", "2e-4", False),
        ("Training filings", "1,000", False),
    ]
    assert points[1].accuracy.point > points[0].accuracy.point


def test_a_recipe_is_its_training_record_with_its_seeds_and_untuned_base(
    tmp_path: Path,
) -> None:
    _, manifest = dataset.build(training_pool())
    config = TrainConfig(base=BASES["2b"].repo, size="2b", rank=64, alpha=128, epochs=1)
    record = RunRecord(
        run_id=config.run_id,
        config=config,
        dataset=manifest,
        base_revision=BASES["2b"].revision,
        started_at=dt.datetime(2026, 9, 23, tzinfo=dt.UTC),
        steps=317,
        seconds_per_step=27.0,
        versions={"python": "3.13.15", "peft": "0.21.0"},
    )
    tuned, untuned = tmp_path / "tuned", tmp_path / "untuned"
    seeds: list[tuple[int, dict[int, str | None]]] = [(0, {}), (1, {0: WRONG}), (2, {})]
    for seed, answers in seeds:
        run = f"2b-r64-lr1e-4-nall-s{seed}-e1"
        make_run(tuned / run, answers, model=f"selfhosted/{run}-bf16", run_items=items(), cost=0.0)
    # A second epoch and a quantised format of the same recipe are not seeds.
    make_run(
        tuned / "e2",
        {},
        model="selfhosted/2b-r64-lr1e-4-nall-s0-e2-bf16",
        run_items=items(),
        cost=0.0,
    )
    make_run(
        tuned / "q8",
        {},
        model="selfhosted/2b-r64-lr1e-4-nall-s0-e1-gguf-q8_0",
        run_items=items(),
        cost=0.0,
    )
    make_run(
        untuned / "2b",
        {0: WRONG, 1: WRONG},
        model="selfhosted/untuned-2b-it",
        run_items=items(),
        cost=0.0,
    )
    make_run(untuned / "4b", {}, model="selfhosted/untuned-4b-it", run_items=items(), cost=0.0)
    post = report.load(report.run_dirs(tuned), items())[0]
    prompted = report.load(report.run_dirs(untuned), items())[0]

    r = site.recipe("2b-r64-lr1e-4-nall-s0-e1", record, post, prompted)
    assert r.facts.name == "Gemma 4 E2B"
    assert (r.rank, r.alpha, r.epochs, r.steps) == (64, 128, 1, 317)
    assert r.effective_batch == config.batch_size * config.grad_accum
    assert r.train_filings == manifest.train
    assert len(r.seeds) == 3 and r.seeds[1].point < r.seeds[0].point
    assert [u.prompt for u in r.untuned] == ["zero-shot"]  # only its own base's
    assert r.published == "Peter-A-P/smallprint-2b"


def test_every_size_the_page_can_show_has_its_facts() -> None:
    assert set(site.BASE_FACTS) == set(BASES)


# -- the committed page -----------------------------------------------------------------------


def test_the_committed_results_file_agrees_with_the_break_even_it_prints() -> None:
    """Every curve in site/results.json is what the Python gives for the inputs beside it,
    so a results file edited by hand, or written by an older break-even, fails here."""
    data = site.SiteData.model_validate_json((SITE / site.RESULTS).read_text(encoding="utf-8"))
    prices = {f.key: f.usd_per_1000 for f in data.frontier}
    assert data.anchor in prices and data.ceiling in prices
    for curve in data.curves:
        served = data.served[curve.served]
        inputs = BreakEvenInputs(
            price=GpuPrice(
                gpu=served.gpu,
                provider=served.provider,
                kind=PRICE.kind,
                usd_per_hour=served.usd_per_hour,
                checked=served.price_checked,
                source="site/results.json",
            ),
            requests_per_second=served.requests_per_second,
            api_usd_per_call=prices[curve.against] / 1000,
        )
        for point in curve.points:
            assert point == break_even(inputs, point.utilisation)


def test_the_page_loads_nothing_from_another_host() -> None:
    """The content security policy allows only this origin, so a script, stylesheet or font
    from anywhere else would be blocked in production and work in a local preview."""
    page = (SITE / "index.html").read_text(encoding="utf-8")
    loaded = re.findall(r'<script[^>]* src="([^"]+)"', page)
    loaded += re.findall(r'<link[^>]* href="([^"]+)"', page)
    assert loaded == ["breakeven.js", "app.js", "style.css"]
    config = json.loads((SITE / "staticwebapp.config.json").read_text(encoding="utf-8"))
    assert "default-src 'self'" in config["globalHeaders"]["Content-Security-Policy"]


def test_the_local_preview_sends_the_hosts_headers() -> None:
    server = site.preview(SITE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            policy = response.headers["Content-Security-Policy"]
            assert "Frontier Quality" in response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
    assert policy == site.host_headers(SITE)["Content-Security-Policy"]


# -- the calculator's JavaScript --------------------------------------------------------------

NODE = shutil.which("node")

RUNNER = """
const B = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map((c) => {
  const p = B.breakEven(c.inputs, c.utilisation);
  return { volume: p.volumePerMonth, gpus: p.gpus, cost: p.monthlyCost, perCall: p.selfHostedUsdPerCall };
});
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="Node is not installed; CI has it")
def test_the_calculator_gives_the_pythons_break_even() -> None:
    grid = []
    for rate in (0.27, 0.49, 2.5):
        for rps in (0.05, 0.7, 3.29):
            for api in (0.0005, 0.00101, 0.03707):
                for fixed in (0.0, 900.0):
                    for utilisation in (0.1, 0.5, 0.9):
                        grid.append((rate, rps, api, fixed, utilisation))
    cases = [
        {
            "inputs": {
                "usdPerHour": rate,
                "requestsPerSecond": rps,
                "apiUsdPerCall": api,
                "fixedUsdPerMonth": fixed,
                "hoursPerMonth": 730.5,
                "secondsPerHour": 3600.0,
            },
            "utilisation": utilisation,
        }
        for rate, rps, api, fixed, utilisation in grid
    ]
    assert NODE is not None
    result = subprocess.run(
        [NODE, "-e", RUNNER, str(SITE / "breakeven.js")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    answers = json.loads(result.stdout)
    kinds = set()
    for (rate, rps, api, fixed, utilisation), js in zip(grid, answers, strict=True):
        price = PRICE.model_copy(update={"usd_per_hour": rate})
        py = break_even(
            BreakEvenInputs(
                price=price,
                requests_per_second=rps,
                api_usd_per_call=api,
                fixed_usd_per_month=fixed,
            ),
            utilisation,
        )
        assert js["perCall"] == pytest.approx(py.self_hosted_usd_per_call, rel=1e-12)
        assert js["gpus"] == py.gpus_at_break_even
        if py.volume_per_month is None:
            assert js["volume"] is None
            kinds.add("never")
        else:
            assert js["volume"] == pytest.approx(py.volume_per_month, rel=1e-12)
            assert js["cost"] == pytest.approx(py.monthly_cost_at_break_even, rel=1e-12)
            kinds.add("one card" if py.gpus_at_break_even == 1 else "more cards")
    # The grid has to reach every branch, or agreeing on it proves less than it says.
    assert kinds == {"never", "one card", "more cards"}
