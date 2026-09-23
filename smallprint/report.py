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
from smallprint.data.split import Split
from smallprint.grade import Interval, ItemGrade, paired_delta_ci, unpaired_delta_ci
from smallprint.quant.quality import Format, judge
from smallprint.serve.launch import parse_served_name

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
    rows, skipped = load(run_dirs(root, split), items)
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
