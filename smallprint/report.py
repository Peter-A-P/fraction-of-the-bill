"""The tables, written from the runs rather than by hand.

Every published table comes out of here, because a table typed by a person is a table that
can quietly stop matching the runs behind it. The README says so: the report command fills
the tables, and nobody edits them in place.

A row is one run directory. The comparison column is a paired delta against a named
baseline over the identical filings, not a difference of two independently rounded
averages, because the runs share their items and the paired interval is the honest one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

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
from smallprint.data.build import SplitItem
from smallprint.grade import Interval, ItemGrade, paired_delta_ci

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


def run_dirs(root: Path) -> list[Path]:
    """Every directory under `root` that holds a finished run."""
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "run.json").is_file())


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
                summary=summarise(manifest, predictions, items),
                grades=tuple(grades),
            )
        )
    return sorted(rows, key=lambda r: r.usd_per_1000), skipped


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
        lines.append(
            f"| `{s.model}` | {s.style.value.replace('_', '-')} | {_pct(s.accuracy)} | "
            f"{_pct(s.exact_match)} | {delta} | US${row.usd_per_1000:,.2f} | "
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


def baselines(root: Path, build_dir: Path, split: str, *, against: str | None = None) -> str:
    """The whole frontier baseline section: the table, and the fields of the cheapest run."""
    items = read_split(build_dir, split)
    rows, skipped = load(run_dirs(root), items)
    if not rows:
        return "_No runs._"
    parts = [
        frontier_table(rows, against=against),
        f"Measured on {rows[0].summary.graded:,} {split} filings, "
        f"US${spend(rows):,.2f} of calls through the gateway.",
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
