"""XBRL facts: the truth, and choosing which one of them answers the question.

A filing does not report one revenue. A third-quarter 10-Q reports revenue for the three
months just ended and revenue for the nine months to date, and prints both again for the
same periods of the prior year: four numbers on one line of one statement, all correctly
tagged, all true. Truth for this task is exactly one of them, and picking the wrong one
silently would make the labels wrong in a way that no amount of training or grading could
recover, because the model would be scored against the mistake.

So selection is explicit and it is tested: the fact whose period ends when the report ends
and whose duration is the length the report covers. The other three become distractors,
handed to the grader so that a model which reads the year-to-date column is reported as
having read the wrong period rather than as having produced an unexplained wrong number.

Fact records here are source-agnostic on purpose. They come from the `companyfacts` API
today and from the quarterly Financial Statement Data Sets in bulk, and the selection rule
below has to be the same one either way or the two sources would not be a cross-check of
each other at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.schema import FIELDS, Extraction, FieldKind, FieldSpec, FiscalPeriod

#: How many days the period a report covers runs to, by fiscal period. A fiscal year is a
#: year; every quarter, including the fourth, is a quarter.
TARGET_DAYS: Final[Mapping[str, int]] = {"FY": 365, "Q1": 91, "Q2": 91, "Q3": 91, "Q4": 91}

#: How far a duration may be from its target and still be that period. Fiscal years are 52
#: or 53 weeks and quarters move with the retail calendar, so the bands are wide; they are
#: narrow enough that a nine-month year-to-date figure can never be mistaken for a quarter.
DURATION_BANDS: Final[Mapping[str, tuple[int, int]]] = {
    "FY": (300, 400),
    "Q1": (60, 120),
    "Q2": (60, 120),
    "Q3": (60, 120),
    "Q4": (60, 120),
}

#: A fact's period end may miss the report's by a few days: filers tag a 52/53-week year end
#: as the Saturday it fell on while the cover page carries the month end.
PERIOD_END_SLACK_DAYS: Final = 7

#: How many other readings of the same line to keep as distractors. Four covers the
#: quarter, the year to date and both prior-year comparatives, which is what a statement
#: actually prints.
MAX_DISTRACTORS: Final = 4


class Fact(BaseModel):
    """One XBRL fact as the company filed it."""

    model_config = ConfigDict(frozen=True)

    concept: str
    unit: str
    value: float | str
    #: Absent on an instant: a balance sheet line is a position on a date, not over a period.
    start: dt.date | None
    end: dt.date
    accession: str
    form: str
    filed: dt.date

    @property
    def is_instant(self) -> bool:
        return self.start is None

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days


class UnusableFacts(ValueError):
    """Raised when a payload is not the shape the fact parser expects."""


def parse_company_facts(payload: object) -> list[Fact]:
    """Flatten a `companyfacts` document into facts.

    The document nests taxonomy, then concept, then unit, then a list of reported values,
    and the same value reappears in every later filing that restates it. Nothing is
    deduplicated here: which filing reported a value is exactly what the selection step
    needs, so the accession stays on every fact.
    """
    if not isinstance(payload, Mapping):
        raise UnusableFacts("companyfacts payload is not an object")
    facts_section = payload.get("facts")
    if not isinstance(facts_section, Mapping):
        raise UnusableFacts("companyfacts payload has no facts section")

    out: list[Fact] = []
    for taxonomy, concepts in facts_section.items():
        if not isinstance(concepts, Mapping):
            continue
        for concept, body in concepts.items():
            units = body.get("units") if isinstance(body, Mapping) else None
            if not isinstance(units, Mapping):
                continue
            for unit, entries in units.items():
                if not isinstance(entries, Sequence):
                    continue
                for entry in entries:
                    fact = _fact_from_entry(f"{taxonomy}:{concept}", str(unit), entry)
                    if fact is not None:
                        out.append(fact)
    return out


def _fact_from_entry(concept: str, unit: str, entry: object) -> Fact | None:
    """One entry, or None if it is missing what a fact needs to be placed in time.

    Entries without an accession are skipped rather than repaired. They come from the SEC's
    own aggregation across filings, and a fact that cannot be attributed to a filing cannot
    be used as that filing's label.
    """
    if not isinstance(entry, Mapping):
        return None
    end, accn, value = entry.get("end"), entry.get("accn"), entry.get("val")
    if not isinstance(end, str) or not isinstance(accn, str) or value is None:
        return None
    start = entry.get("start")
    filed = entry.get("filed")
    return Fact(
        concept=concept,
        unit=unit,
        value=value if isinstance(value, str) else float(value),
        start=dt.date.fromisoformat(start) if isinstance(start, str) else None,
        end=dt.date.fromisoformat(end),
        accession=accn,
        form=str(entry.get("form", "")),
        filed=dt.date.fromisoformat(filed)
        if isinstance(filed, str)
        else dt.date.fromisoformat(end),
    )


def _ends_at(fact: Fact, period_end: dt.date) -> bool:
    return abs((fact.end - period_end).days) <= PERIOD_END_SLACK_DAYS


def _in_band(fact: Fact, fiscal_period: str) -> bool:
    days = fact.days
    if days is None:
        return False
    low, high = DURATION_BANDS[fiscal_period]
    return low <= days <= high


def select_fact(
    facts: Sequence[Fact],
    spec: FieldSpec,
    *,
    period_end: dt.date,
    fiscal_period: str,
) -> tuple[Fact | None, tuple[Fact, ...]]:
    """The fact that answers this field for this report, and the other readings of its line.

    Concepts are tried in the order the schema lists them and the first one with a fact for
    this period wins. Consulting a later synonym once one has matched would be worse than
    useless: a filer that reports both `Revenues` and `SalesRevenueNet` for the period means
    different things by them, and the schema's order is the decision about which one the
    task wants.

    A concept the filing reports only for other periods is not a match, and the search
    moves on. Weis Markets tags `Revenues` for its prior years and only contract revenue for
    the current one; Amcor tags `Revenues` by quarter and the year only as contract revenue.
    Stopping at the first concept reported at all, which this first did, dropped both. The
    near misses are kept as distractors either way: they are the same line's other columns.
    """
    near_misses: list[Fact] = []
    for concept in spec.concepts:
        candidates = [f for f in facts if f.concept == concept]
        if not candidates:
            continue

        at_period_end = [f for f in candidates if _ends_at(f, period_end)]
        wants_instant = spec.kind is FieldKind.MONETARY and any(f.is_instant for f in candidates)

        if wants_instant:
            matches = [f for f in at_period_end if f.is_instant]
        elif concept.startswith("dei:"):
            # Cover-page facts describe the document, not a flow over a period, and filers
            # tag them against whichever context they call the document's: a 10-Q for the
            # third quarter usually uses the nine months to date. The first live build lost
            # three filings in ten to requiring a quarter here. The period end is the test.
            matches = at_period_end
        else:
            matches = [f for f in at_period_end if _in_band(f, fiscal_period)]

        if not matches:
            # Reported, but not for the period asked about. Never the nearest thing to hand;
            # a later synonym may have this period, and if none does the answer is nothing.
            near_misses.extend(candidates)
            continue

        target = TARGET_DAYS[fiscal_period]
        chosen = min(
            matches,
            key=lambda f: (
                abs((f.days or target) - target),
                abs((f.end - period_end).days),
                f.filed,
            ),
        )
        others = [f for f in candidates if f is not chosen] + near_misses
        return chosen, tuple(others[:MAX_DISTRACTORS])

    return None, tuple(near_misses[:MAX_DISTRACTORS])


def build_truth(
    facts: Iterable[Fact],
    *,
    accession: str,
    period_end: dt.date,
    fiscal_period: FiscalPeriod,
) -> tuple[Extraction, dict[str, tuple[float | str, ...]]]:
    """The labels for one filing, and the distractors that explain a model's misreadings.

    Only facts this filing itself reported are considered. A later filing restating the same
    period is a different document saying a different thing, and training a model to produce
    a number that was not on the page it was shown is training it to guess.
    """
    from_this_filing = [f for f in facts if f.accession == accession]

    values: dict[str, object] = {}
    distractors: dict[str, tuple[float | str, ...]] = {}

    for name, spec in FIELDS.items():
        chosen, others = select_fact(
            from_this_filing, spec, period_end=period_end, fiscal_period=fiscal_period
        )
        if chosen is not None:
            values[name] = _as_field_value(spec, chosen.value)
        if others:
            distractors[name] = tuple(
                v
                for v in (_as_distractor(spec, o.value) for o in others)
                if v is not None and v != values.get(name)
            )

    return Extraction.model_validate(values), {k: v for k, v in distractors.items() if v}


def _as_field_value(spec: FieldSpec, value: float | str) -> object:
    """Turn a fact's value into what the schema declares for that field."""
    if spec.kind is FieldKind.DATE:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    if spec.kind is FieldKind.CATEGORICAL:
        return str(value)
    return float(value)


def _as_distractor(spec: FieldSpec, value: float | str) -> float | str | None:
    """Distractors are compared, never parsed, so dates stay as the strings they were filed as."""
    if spec.kind in (FieldKind.DATE, FieldKind.CATEGORICAL):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def period_matches(
    facts: Sequence[Fact],
    spec: FieldSpec,
    *,
    period_end: dt.date,
    fiscal_period: str,
) -> list[Fact]:
    """The fact each of a field's concepts reports for this period, in the schema's order.

    `select_fact` takes the first. The rest are there for the one case where the schema's
    order is not enough: two synonyms both reported for the period with different values,
    of which the statement prints only one (see `pair.py`).
    """
    out = []
    for concept in spec.concepts:
        single = spec.model_copy(update={"concepts": (concept,)})
        chosen, _ = select_fact(facts, single, period_end=period_end, fiscal_period=fiscal_period)
        if chosen is not None:
            out.append(chosen)
    return out
