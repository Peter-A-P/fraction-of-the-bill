"""The tables, written from the runs rather than by hand.

Every published table comes out of here, because a table typed by a person is a table that
can quietly stop matching the runs behind it. The README says so: the report command fills
the tables, and nobody edits them in place.

A row is one run directory. The comparison column is a paired delta against a named
baseline over the identical filings, not a difference of two independently rounded
averages, because the runs share their items and the paired interval is the honest one.

The fine-tuned runs are a second directory of the same shape, one run per served model and
split, the model named `selfhosted/<training run>-<format>` (`serve.launch.served_name`).
Their table pairs each with the most accurate frontier run on the same filings, and puts
the pre-cutoff run of the same model beside it as the contamination gap, which is the one
difference here that cannot be paired: the two sets are different filings by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.baseline import (
    BaselineSummary,
    grade_run,
    read_manifest,
    read_predictions,
    read_split,
    summarise,
    within,
)
from smallprint.bench.breakeven import BreakEvenInputs, BreakEvenPoint, break_even
from smallprint.bench.cost import usd_per_call, usd_per_thousand
from smallprint.bench.load import LoadSummary
from smallprint.bench.run import RECORD as LOAD_RECORD
from smallprint.bench.run import LoadRun
from smallprint.bench.run import read as read_load
from smallprint.chart import Point
from smallprint.chart import render as render_chart
from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.grade import Interval, ItemGrade, paired_delta_ci, unpaired_delta_ci
from smallprint.quant.quality import Format, judge
from smallprint.recost import recost
from smallprint.serve.launch import PROVIDER, parse_served_name, served_name

#: A blank line between the parts of the report. Named because this module is all
#: string building and a bare escape in the middle of a join reads as a typo.
SEPARATOR = chr(10) + chr(10)


class Row(BaseModel):
    """One run, summarised, with the grades kept so deltas can be paired."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    directory: str
    summary: BaselineSummary
    grades: tuple[ItemGrade, ...]

    @property
    def key(self) -> str:
        return f"{self.summary.model} {self.summary.style.value}"

    @property
    def usd_per_1000(self) -> float:
        return self.summary.usd_per_1000.point if self.summary.usd_per_1000 else float("nan")


def run_dirs(root: Path, split: str | None = None) -> list[Path]:
    """Every directory under `root` that holds a run, of one split if one is named.

    Filtered by split before anything is graded, because a run over another split covers
    none of this one's filings and would otherwise be reported as a partial run of it.
    """
    found = sorted(d for d in root.iterdir() if d.is_dir() and (d / "run.json").is_file())
    if split is None:
        return found
    return [d for d in found if read_manifest(d).split == split]


def load(
    directories: Iterable[Path], items: Sequence[SplitItem], *, complete_only: bool = True
) -> tuple[list[Row], list[Path]]:
    """The runs, cheapest first, and the directories left out.

    A run that covers only part of the split is left out of the table by default. Smoke
    runs and rehearsals live beside the real ones, they are the same shape on disk, and a
    twenty-item row sitting in a published table next to seven-hundred-item rows is a
    mistake waiting to be quoted. They also cannot be paired, which is how this was found.
    A run that also answered items a later build dropped still covers the split, and its
    answers to those are set aside.
    """
    wanted = {s.item.item_id for s in items}
    rows, skipped = [], []
    for directory in directories:
        manifest = read_manifest(directory)
        predictions = read_predictions(directory)
        if not predictions or (complete_only and wanted - {p.item_id for p in predictions}):
            skipped.append(directory)
            continue
        grades, _ = grade_run(within(predictions, items)[0], items)
        rows.append(
            Row(
                directory=str(directory),
                summary=summarise(
                    manifest, predictions, items, recosted=recost(directory, manifest.model)
                ),
                grades=tuple(grades),
            )
        )
    # An uncosted run, an untuned base served before any price for it existed, goes last:
    # NaN would sort it anywhere, and the first row is the anchor the deltas are paired with.
    return sorted(rows, key=lambda r: (r.summary.usd_per_1000 is None, r.usd_per_1000)), skipped


def _prompt(summary: BaselineSummary) -> str:
    """The prompt column: the style, and reasoning when it was on, so two runs of one model
    with the same style are never two identical-looking rows."""
    return summary.style.value.replace("_", "-") + (", reasoning" if summary.thinking else "")


def _pct(interval: Interval) -> str:
    return f"{interval.point:.1%} ({interval.low:.1%} to {interval.high:.1%})"


def _signed(interval: Interval) -> str:
    return f"{interval.point:+.1%} ({interval.low:+.1%} to {interval.high:+.1%})"


def frontier_table(rows: Sequence[Row], *, against: str | None = None) -> str:
    """The frontier baseline table: quality, cost and latency, cheapest first.

    `against` names the row every delta is paired with. The default is the cheapest run,
    because the question this project asks is not whether an expensive model is good, it
    is whether anything is worth paying more than the cheapest thing that works.
    """
    if not rows:
        return "_No runs._"
    anchor = next((r for r in rows if r.key == against), rows[0])
    header = (
        "| Model | Prompt | Fields correct (95% CI) | Every field right | "
        f"Paired delta vs {anchor.key} | Cost per 1,000 | Latency p50 | p99 |"
    )
    lines = [header, "|---|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        s = row.summary
        delta = (
            "baseline"
            if row is anchor
            else _signed(paired_delta_ci(list(row.grades), list(anchor.grades)))
        )
        cost = f"US${row.usd_per_1000:,.2f}" if s.usd_per_1000 else "uncosted"
        lines.append(
            f"| `{s.model}` | {_prompt(s)} | {_pct(s.accuracy)} | "
            f"{_pct(s.exact_match)} | {delta} | {cost} | "
            f"{s.latency_p50_ms.point / 1000:,.2f} s | {s.latency_p99_ms.point / 1000:,.2f} s |"
        )
    return "\n".join(lines)


def field_table(row: Row) -> str:
    """Per-field accuracy and the named failure modes, for one run."""
    lines = ["| Field | Accuracy (95% CI) | How it missed |", "|---|---|---|"]
    for report in sorted(row.summary.fields, key=lambda r: r.accuracy.point):
        reasons = ", ".join(f"{k} {v}" for k, v in report.reasons.items()) or "nothing"
        lines.append(f"| `{report.field}` | {_pct(report.accuracy)} | {reasons} |")
    return "\n".join(lines)


def spend(rows: Sequence[Row]) -> float:
    """What these runs cost, from the ledger figures the runs recorded."""
    return sum((r.usd_per_1000 / 1000) * r.summary.graded for r in rows if r.summary.usd_per_1000)


def recost_note(rows: Sequence[Row]) -> str:
    """Says which costs are recomputed from the runs' bytes, and why, or nothing."""
    recosted = sorted({r.summary.price_list or "" for r in rows if r.summary.cost_basis})
    if not recosted:
        return ""
    return (
        " Costs are recomputed from each run's response bytes by the pinned gateway at price "
        f"list {', '.join(recosted)}, which bills a GPT-5.6 cache write at 1.25x input; the "
        "ledger as written priced writes as plain input and under-states OpenAI's runs by 16 "
        "to 22% (smallprint/recost.py)."
    )


def baselines(
    root: Path,
    build_dir: Path,
    split: str,
    *,
    against: str | None = None,
    also: Sequence[Path] = (),
) -> str:
    """The whole baseline section: the table, and the fields of the cheapest run.

    `also` adds the runs of other directories to the same table, which is how the untuned
    bases sit beside the frontier models they are compared with, paired over the same
    filings, without living in the frontier runs' directory.
    """
    items = read_split(build_dir, split)
    directories = [d for r in (root, *also) for d in run_dirs(r, split)]
    rows, skipped = load(directories, items)
    if not rows:
        return "_No runs._"
    parts = [
        frontier_table(rows, against=against),
        f"Measured on {rows[0].summary.graded:,} {split} filings, "
        f"US${spend(rows):,.2f} of calls through the gateway." + recost_note(rows),
        f"**Where `{rows[0].summary.model}` {rows[0].summary.style.value} misses:**",
        field_table(rows[0]),
    ]
    if skipped:
        parts.append(
            "Left out, as runs over part of the split rather than all of it: "
            + ", ".join(f"`{d.name}`" for d in skipped)
            + "."
        )
    return SEPARATOR.join(parts)


def _served(row: Row) -> tuple[str, str, Format]:
    return parse_served_name(row.summary.model)


def best_frontier(rows: Sequence[Row]) -> Row:
    """The frontier run with the highest field accuracy: the bar a fine-tune has to reach."""
    if not rows:
        raise ValueError("no frontier runs to compare with")
    return max(rows, key=lambda r: r.summary.accuracy.point)


def finetuned_table(post: Sequence[Row], pre: Sequence[Row], frontier: Sequence[Row]) -> str:
    """The fine-tunes on the post-cutoff filings, against the best frontier run.

    The delta is paired over the same filings. The last column is pre-cutoff accuracy minus
    post-cutoff, unpaired: positive would be a contamination premium, what the base model
    remembered of filings it was trained on.
    """
    if not post:
        return "_No fine-tuned runs._"
    best = best_frontier(frontier)
    pre_by_model = {r.summary.model: r for r in pre}
    lines = [
        "| Model | Size | Format | Fields correct (95% CI) | Every field right | "
        f"Paired delta vs {best.key} | Pre-cutoff minus post-cutoff |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(post, key=lambda r: (_served(r)[0], _served(r)[1], _served(r)[2].value)):
        size, run, fmt = _served(row)
        s = row.summary
        delta = paired_delta_ci(list(row.grades), list(best.grades))
        before = pre_by_model.get(s.model)
        gap = (
            _signed(unpaired_delta_ci(list(before.grades), list(row.grades)))
            if before is not None
            else "not measured"
        )
        lines.append(
            f"| `{run}` | {size} | {fmt.value} | {_pct(s.accuracy)} | {_pct(s.exact_match)} | "
            f"{_signed(delta)} | {gap} |"
        )
    return "\n".join(lines)


def quantisation_table(post: Sequence[Row]) -> str:
    """Every quantised format against the bf16 build of the same fine-tune, paired.

    A format whose bf16 sibling has not been measured is left out and named, because a
    quantised row without its delta is exactly what the quantisation gate exists to stop.
    """
    by_run: dict[str, dict[Format, Row]] = {}
    for row in post:
        _, run, fmt = _served(row)
        by_run.setdefault(run, {})[fmt] = row
    lines = [
        "| Model | Format | Accuracy delta vs bf16 (95% CI) | Worst field | Ships |",
        "|---|---|---|---|---|",
    ]
    orphans = []
    for run in sorted(by_run):
        formats = by_run[run]
        reference = formats.get(Format.BF16)
        for fmt in sorted(formats, key=lambda f: list(Format).index(f)):
            if fmt is Format.BF16:
                continue
            if reference is None:
                orphans.append(f"`{run}` {fmt.value}")
                continue
            verdict = judge(fmt, list(formats[fmt].grades), list(reference.grades))
            worst = verdict.worst_field
            lines.append(
                f"| `{run}` | {fmt.value} | {_signed(verdict.delta)} | `{worst.field}` "
                f"{worst.delta.point:+.1%} | {'yes' if verdict.ships else 'no'} |"
            )
    if len(lines) == 2:
        lines = ["_No quantised format measured beside its bf16 build._"]
    if orphans:
        lines.append("")
        lines.append(
            "Measured without a bf16 run to judge them against: " + ", ".join(orphans) + "."
        )
    return "\n".join(lines)


def finetuned(root: Path, frontier_root: Path, build_dir: Path) -> str:
    """The fine-tuned section: the quality table and the quantisation table."""
    post_items = read_split(build_dir, Split.TEST_POST_CUTOFF.value)
    post, skipped = load(run_dirs(root, Split.TEST_POST_CUTOFF.value), post_items)
    if not post:
        return "_No fine-tuned runs._"
    pre_dirs = run_dirs(root, Split.TEST_PRE_CUTOFF.value)
    pre = load(pre_dirs, read_split(build_dir, Split.TEST_PRE_CUTOFF.value))[0] if pre_dirs else []
    frontier, _ = load(run_dirs(frontier_root, Split.TEST_POST_CUTOFF.value), post_items)
    parts = [
        finetuned_table(post, pre, frontier),
        f"Measured on {post[0].summary.graded:,} {Split.TEST_POST_CUTOFF.value} filings, "
        "every call through the gateway, temperature as each run's manifest records it.",
        "**Quantisation cost, paired over the same filings:**",
        quantisation_table(post),
    ]
    if skipped:
        parts.append(
            "Left out, as runs over part of the split rather than all of it: "
            + ", ".join(f"`{d.name}`" for d in skipped)
            + "."
        )
    return SEPARATOR.join(parts)


# -- serving and money ---------------------------------------------------------------------

#: The concurrency the serving columns and the self-hosted costs are read at, as the README
#: table names it. 32 in flight is a busy production service on one card without being the
#: saturation point, where latency stops being worth quoting.
SERVING_CONCURRENCY: Final = 32

#: The utilisation self-hosted costs are quoted at in the tables and the chart. The
#: break-even table gives the rest of the curve.
QUOTED_UTILISATION: Final = 0.5

#: The utilisations the break-even table shows. The full curve is `smallprint breakeven`.
TABLE_UTILISATIONS: Final[tuple[float, ...]] = (0.1, 0.3, 0.5, 0.7, 0.9)


def cost_anchor(frontier: Sequence[Row]) -> Row:
    """The frontier run the break-even is against: the cheapest one, because a fine-tune
    that beats the cheapest API on cost has beaten them all, and it is the price a buyer
    who does not need the dearest model would actually pay."""
    costed = [r for r in frontier if r.summary.usd_per_1000 is not None]
    if not costed:
        raise ValueError("no costed frontier run to break even against")
    return min(costed, key=lambda r: r.usd_per_1000)


def _level(run: LoadRun, concurrency: int) -> LoadSummary | None:
    """A level's summary, or None if it was not run, measured nothing, or was retried."""
    level = run.level(concurrency)
    if level is None or level.summary is None or level.retried:
        return None
    return level.summary


def self_hosted_per_1000(run: LoadRun, utilisation: float = QUOTED_UTILISATION) -> Interval | None:
    """Cost per 1,000 at the quoted concurrency and utilisation, with the interval the
    throughput's interval gives it: the low end of the cost is the high end of the rate."""
    summary = _level(run, SERVING_CONCURRENCY)
    if summary is None:
        return None
    rate = run.price.usd_per_hour
    rps = summary.requests_per_second

    def per_1000(requests_per_second: float) -> float:
        return usd_per_thousand(usd_per_call(rate, requests_per_second, utilisation))

    return Interval(
        point=per_1000(rps.point), low=per_1000(rps.high), high=per_1000(rps.low), n=rps.n
    )


def _inputs(run: LoadRun, anchor: Row) -> BreakEvenInputs | None:
    summary = _level(run, SERVING_CONCURRENCY)
    if summary is None:
        return None
    return BreakEvenInputs(
        price=run.price,
        requests_per_second=summary.requests_per_second.point,
        api_usd_per_call=anchor.usd_per_1000 / 1000,
    )


def _volume(point: BreakEvenPoint) -> str:
    if point.volume_per_month is None:
        return "never"
    return f"{point.volume_per_month / 1e6:,.1f}M"


def _by_run(runs: Sequence[LoadRun]) -> list[LoadRun]:
    return sorted(runs, key=lambda r: parse_served_name(r.model)[1:])


def serving_table(post: Sequence[Row], runs: Sequence[LoadRun], anchor: Row) -> str:
    """Quantisation cost, serving and money: one row per served model that was load-tested."""
    by_model = {r.summary.model: r for r in post}
    lines = [
        "| Model | Format | Accuracy delta vs bf16 (95% CI) | "
        f"Req/s at c={SERVING_CONCURRENCY} | TTFT p99 ms | Cost per 1,000 | "
        f"Break-even volume at {QUOTED_UTILISATION:.0%} utilisation |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for run in _by_run(runs):
        _, name, fmt = parse_served_name(run.model)
        summary = _level(run, SERVING_CONCURRENCY)
        if summary is None:
            lines.append(
                f"| `{name}` | {fmt.value} | | not measured at c={SERVING_CONCURRENCY}, "
                "or retried | | | |"
            )
            continue
        delta = "reference"
        if fmt is not Format.BF16:
            row = by_model.get(run.model)
            reference = by_model.get(f"{PROVIDER}/{served_name(name, Format.BF16)}")
            delta = (
                _signed(judge(fmt, list(row.grades), list(reference.grades)).delta)
                if row is not None and reference is not None
                else "not measured"
            )
        cost = self_hosted_per_1000(run)
        inputs = _inputs(run, anchor)
        volume = _volume(break_even(inputs, QUOTED_UTILISATION)) if inputs else ""
        money_cell = (
            f"US${cost.point:,.3f} (US${cost.low:,.3f} to US${cost.high:,.3f})" if cost else ""
        )
        rps, ttft = summary.requests_per_second, summary.ttft_p99_ms
        lines.append(
            f"| `{name}` | {fmt.value} | {delta} | "
            f"{rps.point:,.2f} ({rps.low:,.2f} to {rps.high:,.2f}) | "
            f"{ttft.point:,.0f} ({ttft.low:,.0f} to {ttft.high:,.0f}) | "
            f"{money_cell} | {volume} |"
        )
    return "\n".join(lines)


def breakeven_table(runs: Sequence[LoadRun], anchor: Row) -> str:
    """Extractions a month at which each served model first costs no more than the anchor,
    across utilisation. "never" where one card at that utilisation already costs more per
    call than the API does."""
    lines = [
        "| Model | Format | GPU | " + " | ".join(f"{u:.0%}" for u in TABLE_UTILISATIONS) + " |",
        "|---|---|---|" + "---:|" * len(TABLE_UTILISATIONS),
    ]
    for run in _by_run(runs):
        _, name, fmt = parse_served_name(run.model)
        inputs = _inputs(run, anchor)
        if inputs is None:
            continue
        cells = [_volume(break_even(inputs, u)) for u in TABLE_UTILISATIONS]
        lines.append(f"| `{name}` | {fmt.value} | {run.price} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pareto_points(
    frontier: Sequence[Row], post: Sequence[Row], runs: Sequence[LoadRun]
) -> list[Point]:
    """Every costed run as a point: frontier from the ledger, self-hosted from its load test."""
    points = [
        Point(
            label=f"{r.summary.model.split('/', 1)[-1]} {r.summary.style.value.replace('_', '-')}",
            self_hosted=False,
            usd_per_1000=r.usd_per_1000,
            accuracy=r.summary.accuracy.point,
            low=r.summary.accuracy.low,
            high=r.summary.accuracy.high,
        )
        for r in frontier
        if r.summary.usd_per_1000 is not None
    ]
    by_model = {r.summary.model: r for r in post}
    for run in runs:
        row = by_model.get(run.model)
        cost = self_hosted_per_1000(run)
        if row is None or cost is None:
            continue
        _, name, fmt = parse_served_name(run.model)
        points.append(
            Point(
                label=f"{name} {fmt.value}",
                self_hosted=True,
                usd_per_1000=cost.point,
                accuracy=row.summary.accuracy.point,
                low=row.summary.accuracy.low,
                high=row.summary.accuracy.high,
            )
        )
    return points


def read_loads(root: Path) -> list[LoadRun]:
    """Every load test under `root`, one directory each."""
    return [read_load(d) for d in sorted(root.iterdir()) if (d / LOAD_RECORD).is_file()]


def money(
    finetuned_root: Path,
    frontier_root: Path,
    bench_root: Path,
    build_dir: Path,
    *,
    chart: Path | None = None,
) -> str:
    """The serving and money section, and the Pareto chart if a path is given for it."""
    items = read_split(build_dir, Split.TEST_POST_CUTOFF.value)
    post, _ = load(run_dirs(finetuned_root, Split.TEST_POST_CUTOFF.value), items)
    frontier, _ = load(run_dirs(frontier_root, Split.TEST_POST_CUTOFF.value), items)
    runs = read_loads(bench_root)
    if not runs:
        return "_No load tests._"
    anchor = cost_anchor(frontier)
    caption = (
        "Self-hosted cost is the GPU-hour rate over the throughput measured at "
        f"{SERVING_CONCURRENCY} requests in flight, at {QUOTED_UTILISATION:.0%} utilisation; "
        f"break-even is against `{anchor.key}` at US${anchor.usd_per_1000:,.2f} per 1,000, "
        "the cheapest frontier run, "
        + (
            "recomputed from its response bytes at OpenAI's cache-write rate."
            if anchor.summary.cost_basis
            else "from the gateway ledger."
        )
    )
    parts = [
        serving_table(post, runs, anchor),
        caption,
        "**Break-even, extractions a month, across utilisation:**",
        breakeven_table(runs, anchor),
    ]
    if chart is not None:
        points = pareto_points(frontier, post, runs)
        chart.write_text(render_chart(points, caption=caption), encoding="utf-8", newline="\n")
        parts.append(f"![Field accuracy against cost per 1,000 extractions]({chart.as_posix()})")
    return SEPARATOR.join(parts)
