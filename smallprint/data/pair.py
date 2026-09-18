"""Pairing a filing's statements with its XBRL truth, and the filter that keeps the task honest.

An item is the text a model is shown and the facts it is graded against. The pairing is
where the two meet, and the filter here is the rule `PLAN.md` section 3 states: a filing is
kept only when every fact it is labelled with can be found in the text it is shown. A label
that is not on the page turns extraction into inference, and a model trained on such
labels is being taught to produce numbers it cannot see.

"Found" means something exact. A numeric fact is found when a number printed in its own
section, multiplied by that section's printed scale, is within the grader's tolerance of the
fact, so the filter and the grade agree about what a correct reading is. Its own section,
because a revenue figure found in the balance sheet or a total assets figure found on the
cover page is a coincidence and not a location. The sign is not required to match: a loss
printed without parentheses under a line that says "loss" is on the page, and reading its
sign is the model's job.

Every filing that is dropped is dropped for a named reason, and the tally of reasons goes in
the datasheet. A filter whose losses are not published is a filter that could be removing
exactly the hard cases and nobody would know.

`fiscal_period` is the one field that is not searched for. It is never printed as "Q3";
it is read from which columns the income statement carries (three months only, six, or nine
alongside three) or from the report being annual. That is still reading the page, which is
why it stays in the task, but no string search can confirm it.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final, get_args

from pydantic import BaseModel, ConfigDict

from smallprint.data.statements import LocatedFiling, locate
from smallprint.data.xbrl import Fact, build_truth
from smallprint.grade import GradeContext, normalise_categorical, tolerance
from smallprint.schema import (
    FIELDS,
    REQUIRED_FIELDS,
    Extraction,
    FieldKind,
    FieldSpec,
    FiscalPeriod,
)

#: Fields that are read from the shape of the report rather than printed as a value.
NOT_SEARCHED: Final[frozenset[str]] = frozenset({"fiscal_period"})

_FISCAL_PERIODS: Final[Mapping[str, FiscalPeriod]] = {p: p for p in get_args(FiscalPeriod)}

_PERIOD_END_CONCEPT: Final = FIELDS["period_end"].concepts[0]
_FISCAL_PERIOD_CONCEPT: Final = FIELDS["fiscal_period"].concepts[0]

#: A printed number: grouped by commas, or a bare run of digits, either with decimals.
_PRINTED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: A dash standing alone in a statement cell means zero.
_ZERO_DASH = re.compile("(?:^|[|\\s])[-" + chr(0x2013) + chr(0x2014) + "](?=$|[|\\s])")

#: State and territory codes as the cover page spells them out. The cover page says
#: "Delaware", the fact says "DE". Codes outside this table, the EDGAR codes for foreign
#: jurisdictions, are searched for as printed, and a filing whose cover page names its
#: country instead is dropped with its reason counted; the first fetch says how many.
STATE_NAMES: Final[Mapping[str, str]] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}  # fmt: skip


class DropReason(StrEnum):
    """Why a filing did not become an item."""

    #: The locator found no table that is the income statement.
    NO_INCOME_STATEMENT = "no_income_statement"
    NO_BALANCE_SHEET = "no_balance_sheet"
    #: The two statements declare different scales. The grader takes one scale per item,
    #: and guessing which statement a field's tolerance should follow is not worth the rare
    #: filing that does this.
    MIXED_SCALE = "mixed_scale"
    #: The cover-page facts that place the report in time are absent or unreadable.
    NO_PERIOD = "no_period"
    #: A required field has no fact in this filing, so the item could not be graded.
    TRUTH_MISSING = "truth_missing"
    #: A field's fact is not printed where that field is read from.
    UNLOCATABLE = "unlocatable"


class Item(BaseModel):
    """One example: what the model is shown, and what it is graded against."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    cik: int
    form: str
    filed: dt.date
    period_end: dt.date
    fiscal_period: FiscalPeriod
    text: str
    truth: Extraction
    context: GradeContext


class Dropped(BaseModel):
    """A filing that is not an item, and exactly why."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    reason: DropReason
    field: str | None = None

    @property
    def key(self) -> str:
        """The row this drop counts under in the datasheet."""
        return self.reason.value if self.field is None else f"{self.reason.value}:{self.field}"


def printed_numbers(text: str) -> tuple[float, ...]:
    """Every number printed in `text`, as a magnitude. Dashes standing for zero count as 0."""
    values = [float(m.group(0).replace(",", "")) for m in _PRINTED_NUMBER.finditer(text)]
    if _ZERO_DASH.search(text):
        values.append(0.0)
    return tuple(values)


#: Spelled out rather than taken from `strftime`, whose month names follow the locale of the
#: machine the build happens to run on.
_MONTHS: Final[tuple[str, ...]] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip


def _date_forms(day: dt.date) -> tuple[str, ...]:
    month = _MONTHS[day.month - 1]
    short = [month[:3], month[:3] + "."] + (["Sept.", "Sept"] if day.month == 9 else [])
    forms = [f"{m} {d}, {day.year}" for m in (month, *short) for d in (day.day, f"{day.day:02d}")]
    return (*forms, day.isoformat())


def _found_number(spec: FieldSpec, truth: float, located: LocatedFiling, ctx: GradeContext) -> bool:
    match spec.kind:
        case FieldKind.PER_SHARE:
            scale = 1.0
        case FieldKind.SHARE_COUNT:
            scale = ctx.share_scale if ctx.share_scale is not None else ctx.scale
        case _:
            scale = ctx.scale
    allowed = tolerance(spec, truth, ctx)
    target = abs(truth)
    return any(
        abs(n * scale - target) <= allowed
        for n in printed_numbers(located.section_text(spec.section))
    )


def locatable(spec: FieldSpec, truth: object, located: LocatedFiling, ctx: GradeContext) -> bool:
    """Whether a correct reading of this field is printed where the field is read from."""
    if spec.name in NOT_SEARCHED:
        return True
    text = located.section_text(spec.section)
    match spec.kind:
        case FieldKind.DATE:
            assert isinstance(truth, dt.date)
            folded = " ".join(text.split()).casefold()
            return any(form.casefold() in folded for form in _date_forms(truth))
        case FieldKind.CATEGORICAL:
            assert isinstance(truth, str)
            folded = f" {normalise_categorical(text)} "
            names = [truth]
            if spec.name == "state_of_incorporation" and truth.upper() in STATE_NAMES:
                names.append(STATE_NAMES[truth.upper()])
            return any(f" {normalise_categorical(n)} " in folded for n in names)
        case _:
            assert isinstance(truth, float | int)
            return _found_number(spec, float(truth), located, ctx)


def _cover_value(facts: Sequence[Fact], concept: str) -> str | None:
    for fact in facts:
        if fact.concept == concept and isinstance(fact.value, str):
            return fact.value.strip()
    return None


def pair_filing(
    document: bytes,
    facts: Iterable[Fact],
    *,
    cik: int,
    accession: str,
    form: str,
    filed: dt.date,
) -> Item | Dropped:
    """Turn one filing into an item, or say why it cannot be one.

    `facts` may be every fact the company has filed; only this accession's are used, for
    the reason `build_truth` gives. The report's period end and fiscal period come from the
    same facts, the cover-page `dei` ones, rather than from the caller, so that the period
    the truth is selected for is the one the filing itself declared.
    """
    own = [f for f in facts if f.accession == accession]

    period_text = _cover_value(own, _PERIOD_END_CONCEPT)
    fiscal_text = _cover_value(own, _FISCAL_PERIOD_CONCEPT)
    try:
        period_end = dt.date.fromisoformat(period_text) if period_text else None
    except ValueError:
        period_end = None
    fiscal_period = _FISCAL_PERIODS.get(fiscal_text or "")
    if period_end is None or fiscal_period is None:
        return Dropped(item_id=accession, reason=DropReason.NO_PERIOD)

    located = locate(document)
    if located.income is None:
        return Dropped(item_id=accession, reason=DropReason.NO_INCOME_STATEMENT)
    if located.balance is None:
        return Dropped(item_id=accession, reason=DropReason.NO_BALANCE_SHEET)

    declared = {s.scale for s in (located.income, located.balance) if s.scale_declared}
    if len(declared) > 1:
        return Dropped(item_id=accession, reason=DropReason.MIXED_SCALE)
    scale = declared.pop() if declared else 1.0

    truth, distractors = build_truth(
        own, accession=accession, period_end=period_end, fiscal_period=fiscal_period
    )
    for name in REQUIRED_FIELDS:
        if getattr(truth, name) is None:
            return Dropped(item_id=accession, reason=DropReason.TRUTH_MISSING, field=name)

    ctx = GradeContext(scale=scale, share_scale=located.income.share_scale, distractors=distractors)
    for name, spec in FIELDS.items():
        value = getattr(truth, name)
        if value is not None and not locatable(spec, value, located, ctx):
            return Dropped(item_id=accession, reason=DropReason.UNLOCATABLE, field=name)

    return Item(
        item_id=accession,
        cik=cik,
        form=form,
        filed=filed,
        period_end=period_end,
        fiscal_period=fiscal_period,
        text=located.render(),
        truth=truth,
        context=ctx,
    )


def tally(results: Iterable[Item | Dropped]) -> dict[str, int]:
    """Kept and dropped counts, by reason and field, for the datasheet."""
    counts: Counter[str] = Counter()
    for result in results:
        counts["kept" if isinstance(result, Item) else result.key] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
