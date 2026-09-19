"""The hand audit: a person checks a sample of labels against the pages they came from.

Everything upstream is a program checking itself, and a program cannot find the mistake it
was written to make. So before any training run, a sample of items is read by a person, who
marks each one as right or names the fields that are wrong. The result is a published label
error rate with its interval, which is the ceiling on how much any score measured on this
corpus can be trusted: a model cannot be more accurate than its labels.

The sample is drawn by a keyed hash, so it is the same sample on every machine, and spread
across splits and forms, so that a fault confined to annual reports or to the test pool is
not missed by a sample that happens to hold none. The pages show the input exactly as the
model sees it, with the labels beside it, and the verdicts go in a CSV that the report reads
back.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.data.build import SplitItem
from smallprint.schema import FIELDS, SCHEMA

AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "n",
    "item_id",
    "split",
    "form",
    "fiscal_period",
    "verdict",
    "wrong_fields",
    "note",
)
#: What a verdict may be. Blank means not yet audited.
VERDICTS: Final[frozenset[str]] = frozenset({"ok", "wrong"})

#: z for a 95% interval.
_Z: Final = 1.959964


def _rank(seed: int, item_id: str) -> bytes:
    return hashlib.blake2b(f"audit:{seed}:{item_id}".encode(), digest_size=8).digest()


def sample(items: Sequence[SplitItem], n: int = 200, *, seed: int = 20260919) -> list[SplitItem]:
    """A deterministic sample spread across every (split, form) that has items.

    Strata take turns, each giving its next item by hash rank, until n are drawn, so a small
    stratum is represented at all rather than in proportion to its size.
    """
    strata: dict[tuple[str, str], list[SplitItem]] = defaultdict(list)
    for s in items:
        strata[(s.split.value, s.item.form)].append(s)
    queues = [
        sorted(group, key=lambda s: _rank(seed, s.item.item_id))
        for _, group in sorted(strata.items())
    ]
    chosen: list[SplitItem] = []
    depth = 0
    while len(chosen) < n and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < n:
                chosen.append(queue[depth])
        depth += 1
    return chosen


def _label(value: object) -> str:
    return "null" if value is None else str(value)


def page(number: int, s: SplitItem) -> str:
    """One item as the auditor sees it: the labels to check, then the input as the model sees it."""
    item = s.item
    rows = "\n".join(
        f"| {spec.name} | {_label(getattr(item.truth, spec.name))} | {spec.section.value} |"
        for spec in SCHEMA
    )
    nulled = (
        f"\nLabelled null because the statement prints no line for it: {', '.join(item.not_on_page)}.\n"
        if item.not_on_page
        else ""
    )
    share_scale = (
        item.context.share_scale if item.context.share_scale is not None else item.context.scale
    )
    return f"""# {number:03d} {item.item_id}

{item.form} {item.fiscal_period}, period ended {item.period_end.isoformat()}, filed
{item.filed.isoformat()}, split `{s.split.value}`. Scale read from the page: money
{item.context.scale:,.0f}, shares {share_scale:,.0f}. Labels are in whole units.

Check each label against the input below. If any is wrong, set the verdict to `wrong` in
`audit.csv` and name the fields in `wrong_fields`, separated by spaces.
{nulled}
| Field | Label | Read from |
|---|---|---|
{rows}

## The input

```text
{item.text}
```
"""


def write_audit(out_dir: Path, chosen: Sequence[SplitItem]) -> Path:
    """Write the pages and a blank verdict sheet. Refuses to overwrite verdicts already given."""
    sheet = out_dir / "audit.csv"
    if sheet.exists() and any(row["verdict"] for row in read_sheet(sheet)):
        raise FileExistsError(f"{sheet} already holds verdicts; audit elsewhere or move it")
    pages = out_dir / "items"
    pages.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for number, s in enumerate(chosen, start=1):
            (pages / f"{number:03d}-{s.item.item_id}.md").write_text(
                page(number, s), encoding="utf-8", newline="\n"
            )
            writer.writerow(
                {
                    "n": number,
                    "item_id": s.item.item_id,
                    "split": s.split.value,
                    "form": s.item.form,
                    "fiscal_period": s.item.fiscal_period,
                    "verdict": "",
                    "wrong_fields": "",
                    "note": "",
                }
            )
    return sheet


def read_sheet(sheet: Path) -> list[dict[str, str]]:
    with sheet.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Proportion(BaseModel):
    """A share with its 95% Wilson interval, which stays inside 0 to 1 at small counts."""

    model_config = ConfigDict(frozen=True)

    count: int
    n: int
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.1%} ({self.low:.1%} to {self.high:.1%}, {self.count} of {self.n})"


def wilson(count: int, n: int) -> Proportion:
    """The Wilson score interval. Used rather than a bootstrap because with zero errors in
    200 the bootstrap interval collapses to nothing, and "no errors seen" is not "no errors"."""
    if n <= 0:
        raise ValueError("an interval needs at least one observation")
    if not 0 <= count <= n:
        raise ValueError(f"count {count} is outside 0 to {n}")
    p = count / n
    denominator = 1 + _Z**2 / n
    centre = (p + _Z**2 / (2 * n)) / denominator
    half = _Z * math.sqrt(p * (1 - p) / n + _Z**2 / (4 * n * n)) / denominator
    return Proportion(
        count=count, n=n, point=p, low=max(0.0, centre - half), high=min(1.0, centre + half)
    )


class AuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    audited: int
    pending: int
    items_wrong: Proportion
    wrong_by_field: dict[str, int]


def report(sheet: Path) -> AuditReport:
    """The label error rate from the verdicts given so far, and which fields were wrong."""
    rows = read_sheet(sheet)
    done = [r for r in rows if r["verdict"].strip()]
    for row in done:
        verdict = row["verdict"].strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"row {row['n']}: verdict must be ok or wrong, got {row['verdict']!r}")
        fields = row["wrong_fields"].split()
        unknown = [f for f in fields if f not in FIELDS]
        if unknown:
            raise ValueError(f"row {row['n']}: not schema fields: {unknown}")
        if verdict == "wrong" and not fields:
            raise ValueError(f"row {row['n']}: a wrong verdict has to name the wrong fields")
    if not done:
        raise ValueError("no verdicts given yet")
    wrong = [r for r in done if r["verdict"].strip().lower() == "wrong"]
    by_field = Counter(f for r in wrong for f in r["wrong_fields"].split())
    return AuditReport(
        audited=len(done),
        pending=len(rows) - len(done),
        items_wrong=wilson(len(wrong), len(done)),
        wrong_by_field=dict(by_field.most_common()),
    )
