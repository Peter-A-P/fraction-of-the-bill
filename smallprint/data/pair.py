"""Pairing a filing's statements with its XBRL truth, and the filter that keeps the task honest.

An item is the text a model is shown and the facts it is graded against. The pairing is
where the two meet, and the filter here is the rule `PLAN.md` section 3 states: a filing is
kept only when every fact it is labelled with can be found in the text it is shown. A label
that is not on the page turns extraction into inference, and a model trained on such
labels is being taught to produce numbers it cannot see.

"Found" means something exact. A numeric fact is found when a number printed in its own
section, multiplied by that section's printed scale, is the fact at the precision it is
printed to: within half the printed unit, so 27,596 in thousands finds 27,595,698. Its own
section, because a revenue figure found in the balance sheet or a total assets figure found
on the cover page is a coincidence and not a location. A share count must also sit on a row
that names shares, or a Basic or Diluted row, because in a statement printed in millions a
three-digit share count lands on some expense line by chance. The sign is not required to
match: a loss printed without parentheses under a line that says "loss" is on the page, and
reading its sign is the model's job.

This is stricter than the grader, which allows half a percent, and at first it was not. The
hand audit of 2026-09-22 found that the filter had used the grader's tolerance and so taken
coincidences for locations: a share count "found" on depreciation, another on interest
expense, cash labelled 236,000 beside a page printing $236,340, cash without restricted
cash beside a line that included it. A reading the filter accepts is still one the grader
accepts, which is what keeping the two in step was for.

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

import lxml.etree
from pydantic import BaseModel, ConfigDict, ValidationError

from smallprint.data.statements import LocatedFiling, locate
from smallprint.data.xbrl import Fact, build_truth, period_matches
from smallprint.grade import GradeContext, normalise_categorical, tolerance
from smallprint.schema import (
    FIELDS,
    REQUIRED_FIELDS,
    Extraction,
    FieldKind,
    FieldSpec,
    FiscalPeriod,
)

#: Optional fields whose label becomes null, rather than dropping the filing, when the
#: statement prints no line for them at all. Decided 2026-09-19: about 3% of filings print
#: weighted diluted shares only in a note, and for the page the model is shown, "not
#: reported" is the correct reading. Only when the line is absent: if a share-count line
#: is printed but does not match, the filing is still dropped, because nulling it would
#: mark a correct reading of that line as a hallucination.
NULL_WHEN_LINE_ABSENT: Final[frozenset[str]] = frozenset({"shares_diluted"})

#: What a share-count line looks like, and what a per-share line looks like, which is not one.
_SHARE_COUNT_LINE = re.compile(r"weighted|\bshares\b", re.IGNORECASE)
#: The rows a share count is read from: a share-count line, or the Basic and Diluted rows
#: that statements set under a "Weighted average shares" heading.
_SHARE_COUNT_ROW = re.compile(
    r"weighted|\bshares\b|^\W*(?:basic|diluted)\b|assuming dilution", re.I
)
#: A row that names a count of shares outright, whatever else it says.
_SHARE_COUNT_NAMED = re.compile(
    r"weighted|number of (?:common |ordinary )?shares|shares outstanding", re.I
)
_PER_SHARE_LINE = re.compile(
    r"per\s+(?:common\s+|ordinary\s+)?share\b|per[\s-]share", re.IGNORECASE
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

    #: EDGAR did not serve the document, the company's facts or its filing history.
    NOT_FETCHED = "not_fetched"
    #: A bank or savings institution, out of scope by decision (docs/data.md, 2026-09-19):
    #: it reports interest income and gains on loans, with no single revenue line to grade.
    BANK = "bank"
    #: The filing history does not name a primary document for this accession.
    NO_PRIMARY_DOCUMENT = "no_primary_document"
    #: The document is not HTML the parser can read at all.
    UNREADABLE = "unreadable"
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
    #: A filed fact the schema cannot hold, such as negative cash.
    TRUTH_INVALID = "truth_invalid"
    #: An error no rule anticipated, named by its type in `field`. Counted, never silent.
    PAIRING_ERROR = "pairing_error"
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
    #: Fields the filing tagged but the statement does not print a line for, labelled null
    #: because null is what a reader of this page would answer. See NULL_WHEN_LINE_ABSENT.
    not_on_page: tuple[str, ...] = ()


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


def date_forms(day: dt.date) -> tuple[str, ...]:
    month = _MONTHS[day.month - 1]
    short = [month[:3], month[:3] + "."] + (["Sept.", "Sept"] if day.month == 9 else [])
    forms = [f"{m} {d}, {day.year}" for m in (month, *short) for d in (day.day, f"{day.day:02d}")]
    # Some cover pages print the date as digits: "period ended 03/31/2025".
    numeric = (f"{day.month:02d}/{day.day:02d}/{day.year}", f"{day.month}/{day.day}/{day.year}")
    return (*forms, *numeric, day.isoformat())


def printed_scale(spec: FieldSpec, ctx: GradeContext) -> float:
    """What one printed unit of this field is worth: the scale, or one for per-share."""
    match spec.kind:
        case FieldKind.PER_SHARE:
            return 1.0
        case FieldKind.SHARE_COUNT:
            return ctx.share_scale if ctx.share_scale is not None else ctx.scale
        case _:
            return ctx.scale


def printed_as(spec: FieldSpec, printed: float, truth: float, ctx: GradeContext) -> bool:
    """Whether a printed number is this fact, at the precision it is printed to.

    Per-share amounts are printed to the cent, which the grader's half cent already is.
    Everything else is within half of one printed unit, so rounding to the thousand is
    allowed and nothing else is.
    """
    if spec.kind is FieldKind.PER_SHARE:
        return abs(printed - abs(truth)) <= tolerance(spec, truth, ctx)
    unit = printed_scale(spec, ctx)
    return abs(printed * unit - abs(truth)) <= unit / 2 * (1 + 1e-9)


def share_count_row(label: str) -> bool:
    """Whether a statement row with this label is one a share count is printed on.

    A row naming a count is one, even when it goes on to say what the count is for:
    "Weighted-average shares used in computing net loss per share, basic and diluted" is a
    share count. Only a row that says "per share" without naming a count is a per-share
    amount. The first strict rebuild got this backwards and nulled 417 real share counts.
    """
    if _SHARE_COUNT_NAMED.search(label):
        return True
    return bool(_SHARE_COUNT_ROW.search(label)) and not _PER_SHARE_LINE.search(label)


def _candidates(spec: FieldSpec, located: LocatedFiling) -> tuple[float, ...]:
    """The numbers a field's value may be found among."""
    if spec.kind is FieldKind.SHARE_COUNT and located.income is not None:
        rows = located.income.rows
        cells = [cell for row in rows if row and share_count_row(row[0]) for cell in row[1:]]
        return printed_numbers(" | ".join(cells))
    return printed_numbers(located.section_text(spec.section))


def _found_number(spec: FieldSpec, truth: float, located: LocatedFiling, ctx: GradeContext) -> bool:
    return any(printed_as(spec, n, truth, ctx) for n in _candidates(spec, located))


def locatable(spec: FieldSpec, truth: object, located: LocatedFiling, ctx: GradeContext) -> bool:
    """Whether a correct reading of this field is printed where the field is read from."""
    if spec.name in NOT_SEARCHED:
        return True
    text = located.section_text(spec.section)
    match spec.kind:
        case FieldKind.DATE:
            assert isinstance(truth, dt.date)
            # "December 31 , 2024", with the comma set apart, is printed on real cover pages.
            folded = " ".join(text.split()).replace(" ,", ",").casefold()
            return any(form.casefold() in folded for form in date_forms(truth))
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


def _read_share_scale(truth: Extraction, located: LocatedFiling, ctx: GradeContext) -> GradeContext:
    """Shares printed whole under a heading that scales only the money.

    "(Dollars in thousands except per share amounts)" says nothing about share counts, and
    many filers under such a heading print them whole. When the heading gives shares no
    scale of their own, and the count is on the page whole but not at the money's scale,
    the page has said how it prints shares, and the grader is told so. Found on the first
    live build, where this dropped one filing in thirty.
    """
    shares = truth.shares_diluted
    if shares is None or ctx.share_scale is not None or ctx.scale == 1.0:
        return ctx
    spec = FIELDS["shares_diluted"]
    if locatable(spec, shares, located, ctx):
        return ctx
    whole = ctx.model_copy(update={"share_scale": 1.0})
    return whole if locatable(spec, shares, located, whole) else ctx


def _printed_synonym(
    truth: Extraction,
    facts: Sequence[Fact],
    located: LocatedFiling,
    ctx: GradeContext,
    period_end: dt.date,
    fiscal_period: FiscalPeriod,
) -> Extraction:
    """Among synonyms reported for the period, the one the statement prints.

    The schema's concept order is a preference, and it is usually enough. It is not when a
    filer reports two synonyms for the same period with different values: The Andersons
    tag contract revenue as a fraction of the top line, so `Revenues` has to come first;
    Hasbro tag `Revenues` above the "Net revenues" their statement prints, which is their
    contract revenue. The label is still a fact the company filed. The page decides which of
    the filed synonyms the task means, which is what the field's own description asks for:
    the line of the statement. If none is printed the first stays, and the filter drops it.
    """
    updates: dict[str, object] = {}
    for name, spec in FIELDS.items():
        value = getattr(truth, name)
        if value is None or spec.kind in (FieldKind.DATE, FieldKind.CATEGORICAL):
            continue
        if locatable(spec, value, located, ctx):
            continue
        alternatives = period_matches(
            facts, spec, period_end=period_end, fiscal_period=fiscal_period
        )
        # A first choice within the grader's tolerance of a printed figure is a reading of
        # that line, rounded or restated in its tag. A synonym may replace it only with the
        # same line's figure, not with another line: one filer's cash tag was 236,000 beside
        # a printed $236,340, and a second tag equal to its restricted cash line, 100,000.
        near = float(value) if _near_printed(spec, float(value), located, ctx) else None
        for fact in alternatives[1:]:
            if not isinstance(fact.value, float):
                continue
            if near is not None and abs(fact.value - near) > tolerance(spec, near, ctx):
                continue
            if locatable(spec, fact.value, located, ctx):
                updates[name] = fact.value
                break
    return truth.model_copy(update=updates) if updates else truth


def _near_printed(spec: FieldSpec, truth: float, located: LocatedFiling, ctx: GradeContext) -> bool:
    """Whether a printed number is within the grader's tolerance of this value."""
    allowed = tolerance(spec, truth, ctx)
    unit = printed_scale(spec, ctx)
    return any(abs(n * unit - abs(truth)) <= allowed for n in _candidates(spec, located))


def _line_printed(name: str, located: LocatedFiling) -> bool:
    """Whether the statement prints a line for this field at all, whatever its value."""
    if name != "shares_diluted" or located.income is None:
        raise ValueError(f"no line test for {name}")
    return any(
        row and _SHARE_COUNT_LINE.search(row[0]) and share_count_row(row[0])
        for row in located.income.rows
    )


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

    try:
        located = locate(document)
    except (lxml.etree.ParserError, ValueError):
        return Dropped(item_id=accession, reason=DropReason.UNREADABLE)
    if located.income is None:
        return Dropped(item_id=accession, reason=DropReason.NO_INCOME_STATEMENT)
    if located.balance is None:
        return Dropped(item_id=accession, reason=DropReason.NO_BALANCE_SHEET)

    declared = {s.scale for s in (located.income, located.balance) if s.scale_declared}
    if len(declared) > 1:
        return Dropped(item_id=accession, reason=DropReason.MIXED_SCALE)
    scale = declared.pop() if declared else 1.0

    try:
        truth, distractors = build_truth(
            own, accession=accession, period_end=period_end, fiscal_period=fiscal_period
        )
    except ValidationError as invalid:
        # A filed fact the schema cannot hold: negative cash, found on the full build, or a
        # malformed date. The fact is wrong or the schema is, and either way it is not a
        # label; the field is named so the tally says which.
        location = invalid.errors()[0].get("loc", ())
        field = str(location[0]) if location else None
        return Dropped(item_id=accession, reason=DropReason.TRUTH_INVALID, field=field)
    for name in REQUIRED_FIELDS:
        if getattr(truth, name) is None:
            return Dropped(item_id=accession, reason=DropReason.TRUTH_MISSING, field=name)

    ctx = GradeContext(scale=scale, share_scale=located.income.share_scale, distractors=distractors)
    ctx = _read_share_scale(truth, located, ctx)
    truth = _printed_synonym(truth, own, located, ctx, period_end, fiscal_period)
    not_on_page: list[str] = []
    for name, spec in FIELDS.items():
        value = getattr(truth, name)
        if value is None or locatable(spec, value, located, ctx):
            continue
        if name in NULL_WHEN_LINE_ABSENT and not _line_printed(name, located):
            not_on_page.append(name)
            continue
        return Dropped(item_id=accession, reason=DropReason.UNLOCATABLE, field=name)
    if not_on_page:
        truth = truth.model_copy(update=dict.fromkeys(not_on_page))

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
        not_on_page=tuple(not_on_page),
    )


def tally(results: Iterable[Item | Dropped]) -> dict[str, int]:
    """Kept and dropped counts, by reason and field, for the datasheet."""
    counts: Counter[str] = Counter()
    for result in results:
        if isinstance(result, Item):
            counts["kept"] += 1
            for name in result.not_on_page:
                counts[f"kept_with_null:{name}"] += 1
        else:
            counts[result.key] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
