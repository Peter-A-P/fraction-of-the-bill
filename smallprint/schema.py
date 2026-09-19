"""The extraction schema: the fifteen fields, their XBRL truth concepts and their tolerances.

One module holds the task definition because three different things have to agree about it:
the prompt shown to a model, the training target written by the pairing step, and the
grader. When they disagree the measurement is wrong in a way no test downstream can catch,
so they all read from `SCHEMA` here.

A field's truth is the XBRL fact the company itself filed, named by `concepts` in order of
preference. The first concept a filing reports wins; the rest are the synonyms that
different filers and different taxonomy versions use for the same line. No model is
consulted at any point, which is the rule this project shares with the release gate: truth
is the fact.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FieldKind(StrEnum):
    """How a field is compared to its truth.

    The kind decides the tolerance rule, not the Python type: `MONETARY` and `SHARE_COUNT`
    are both floats, but a share count printed in thousands and a dollar amount printed in
    millions round differently, and `PER_SHARE` is printed to the cent and must be compared
    to the cent rather than to a percentage of itself.
    """

    MONETARY = "monetary"
    PER_SHARE = "per_share"
    SHARE_COUNT = "share_count"
    DATE = "date"
    CATEGORICAL = "categorical"


class Section(StrEnum):
    """Where on the filing a field's value is printed.

    The locatability filter searches for a value only in the section its field belongs to.
    Searching the whole document would find almost any number somewhere, usually in a note
    or in the prior-year column of a different statement, and would keep filings whose
    label is not actually on the page the field is read from.
    """

    COVER = "cover"
    INCOME = "income"
    BALANCE = "balance"
    #: The signature block of the audit report, in an annual report only.
    AUDITOR = "auditor"


#: Relative tolerance for scaled numeric fields. A statement printed in thousands has
#: already thrown away everything below the thousand, so the largest honest disagreement
#: between a correct reading and the filed fact is half of the reporting unit; the relative
#: term only matters for filings that print full units and still disagree in the last
#: digits. Half a percent is generous enough never to fail a correct reading, and tight
#: enough that a wrong line item never passes: the nearest distinct line on a statement is
#: not within 0.5% of its neighbour except by accident, and the distractor check in
#: `grade.py` catches that case by name rather than by tolerance.
RELATIVE_TOLERANCE: Final = 0.005

#: Absolute tolerance for per-share amounts, in the reporting currency. Earnings per share
#: is printed to the cent, so half a cent is the rounding boundary.
PER_SHARE_TOLERANCE: Final = 0.005


class FieldSpec(BaseModel):
    """One field of the extraction task."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: FieldKind
    section: Section
    #: Shown to the model in the prompt. It is the whole of the task specification for this
    #: field, so it says what to do when the statement offers more than one candidate.
    description: str
    #: XBRL concepts that carry this field's truth, in preference order. `dei:` concepts are
    #: entity facts from the cover page; the rest are us-gaap.
    concepts: tuple[str, ...]
    #: Whether a filing may legitimately omit this field. A field that is not `optional` and
    #: has no fact makes a filing the pairing step drops, because an item whose truth is
    #: absent cannot be graded either way.
    optional: bool = False
    #: Allowed values, for categoricals whose range is closed.
    choices: tuple[str, ...] | None = None


SCHEMA: Final[tuple[FieldSpec, ...]] = (
    FieldSpec(
        name="period_end",
        kind=FieldKind.DATE,
        section=Section.COVER,
        description=(
            "The last day of the period this report covers, as YYYY-MM-DD. Not the filing "
            "date and not the end of the prior-year comparative period."
        ),
        concepts=("dei:DocumentPeriodEndDate",),
    ),
    FieldSpec(
        name="fiscal_period",
        kind=FieldKind.CATEGORICAL,
        section=Section.COVER,
        description=(
            "The fiscal period this report covers: FY for an annual report, or Q1, Q2 or Q3 "
            "for a quarterly one."
        ),
        concepts=("dei:DocumentFiscalPeriodFocus",),
        choices=("FY", "Q1", "Q2", "Q3", "Q4"),
    ),
    FieldSpec(
        name="revenue",
        kind=FieldKind.MONETARY,
        section=Section.INCOME,
        description=(
            "Total revenue for the period, in whole units of the reporting currency. The top "
            "line of the income statement, for the current period only."
        ),
        # Revenues first. It is the total by definition; the contract-revenue concept is
        # what ASC 606 revenue is tagged with, and a filer with other revenue (commodity
        # derivatives, leases, interest) tags only part of its top line with it. The first
        # live build found a filer whose contract revenue was $371m of $2,659m of sales.
        concepts=(
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
    ),
    FieldSpec(
        name="cost_of_revenue",
        kind=FieldKind.MONETARY,
        section=Section.INCOME,
        description=(
            "Cost of revenue or cost of goods sold for the period, in whole units of the "
            "reporting currency, as a positive number."
        ),
        concepts=(
            "us-gaap:CostOfRevenue",
            "us-gaap:CostOfGoodsAndServicesSold",
            "us-gaap:CostOfGoodsSold",
        ),
        optional=True,
    ),
    FieldSpec(
        name="operating_income",
        kind=FieldKind.MONETARY,
        section=Section.INCOME,
        description=(
            "Operating income or loss for the period, in whole units of the reporting "
            "currency. Negative if a loss."
        ),
        concepts=("us-gaap:OperatingIncomeLoss",),
        optional=True,
    ),
    FieldSpec(
        name="net_income",
        kind=FieldKind.MONETARY,
        section=Section.INCOME,
        description=(
            "Net income or loss attributable to the company for the period, in whole units "
            "of the reporting currency. Negative if a loss."
        ),
        concepts=("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"),
    ),
    FieldSpec(
        name="eps_basic",
        kind=FieldKind.PER_SHARE,
        section=Section.INCOME,
        description="Basic earnings per share for the period, in currency units per share.",
        concepts=("us-gaap:EarningsPerShareBasic",),
        optional=True,
    ),
    FieldSpec(
        name="eps_diluted",
        kind=FieldKind.PER_SHARE,
        section=Section.INCOME,
        description="Diluted earnings per share for the period, in currency units per share.",
        concepts=("us-gaap:EarningsPerShareDiluted",),
        optional=True,
    ),
    FieldSpec(
        name="shares_diluted",
        kind=FieldKind.SHARE_COUNT,
        section=Section.INCOME,
        description=(
            "Weighted average diluted shares outstanding for the period, as a whole number "
            "of shares rather than in thousands or millions."
        ),
        concepts=(
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstandingAdjusted",
        ),
        optional=True,
    ),
    FieldSpec(
        name="total_assets",
        kind=FieldKind.MONETARY,
        section=Section.BALANCE,
        description=(
            "Total assets at the period end, in whole units of the reporting currency. The "
            "current balance sheet column, not the comparative one."
        ),
        concepts=("us-gaap:Assets",),
    ),
    FieldSpec(
        name="total_liabilities",
        kind=FieldKind.MONETARY,
        section=Section.BALANCE,
        description=(
            "Total liabilities at the period end, in whole units of the reporting currency. "
            "Not total liabilities and equity."
        ),
        concepts=("us-gaap:Liabilities",),
        optional=True,
    ),
    FieldSpec(
        name="cash_and_equivalents",
        kind=FieldKind.MONETARY,
        section=Section.BALANCE,
        description=(
            "Cash and cash equivalents at the period end, in whole units of the reporting "
            "currency, excluding short-term investments held on a separate line."
        ),
        concepts=(
            "us-gaap:CashAndCashEquivalentsAtCarryingValue",
            "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    ),
    FieldSpec(
        name="stockholders_equity",
        kind=FieldKind.MONETARY,
        section=Section.BALANCE,
        description=(
            "Total stockholders equity attributable to the company at the period end, in "
            "whole units of the reporting currency. Negative if a deficit."
        ),
        concepts=(
            "us-gaap:StockholdersEquity",
            "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    FieldSpec(
        name="auditor_name",
        kind=FieldKind.CATEGORICAL,
        section=Section.AUDITOR,
        description=(
            "The name of the independent registered public accounting firm that signed the "
            "audit report, as printed. Null in a quarterly report, which is reviewed rather "
            "than audited."
        ),
        concepts=("dei:AuditorName",),
        optional=True,
    ),
    FieldSpec(
        name="state_of_incorporation",
        kind=FieldKind.CATEGORICAL,
        section=Section.COVER,
        description=(
            "The two-letter code of the state or country of incorporation from the cover "
            "page, for example DE or NY."
        ),
        concepts=("dei:EntityIncorporationStateCountryCode",),
        optional=True,
    ),
)

FIELDS: Final[dict[str, FieldSpec]] = {spec.name: spec for spec in SCHEMA}

#: The fields a filing must have truth for before it can become an item.
REQUIRED_FIELDS: Final[tuple[str, ...]] = tuple(s.name for s in SCHEMA if not s.optional)

FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class Extraction(BaseModel):
    """What a model returns, and what the pairing step writes as the target.

    Every field is nullable because a statement may genuinely not report it, and a model
    that guesses rather than returning null is making a different and worse error than one
    that abstains. The grader tells those two apart by name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    period_end: dt.date | None = None
    fiscal_period: FiscalPeriod | None = None
    revenue: float | None = None
    cost_of_revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    shares_diluted: Annotated[float | None, Field(ge=0)] = None
    total_assets: Annotated[float | None, Field(ge=0)] = None
    total_liabilities: Annotated[float | None, Field(ge=0)] = None
    cash_and_equivalents: Annotated[float | None, Field(ge=0)] = None
    stockholders_equity: float | None = None
    auditor_name: str | None = None
    state_of_incorporation: str | None = None

    @model_validator(mode="after")
    def _fields_match_schema(self) -> Self:
        """Fail loudly if this model and `SCHEMA` have drifted apart.

        The three consumers of the task definition only stay in step because this runs on
        every instance ever built, including the first one a test constructs.
        """
        declared = set(type(self).model_fields)
        specified = set(FIELDS)
        if declared != specified:
            missing = sorted(specified - declared)
            extra = sorted(declared - specified)
            raise ValueError(
                f"Extraction and SCHEMA disagree: missing {missing}, unexpected {extra}"
            )
        return self


def json_schema_for_prompt() -> str:
    """The field list as it appears in the prompt, one line per field.

    Deliberately not `Extraction.model_json_schema()`: pydantic's output carries `anyOf`
    wrappers and `$defs` that spend prompt tokens on JSON Schema mechanics rather than on
    the part a model gets wrong, which is which number on the page belongs in which field.
    """
    kinds = {
        FieldKind.DATE: "string, YYYY-MM-DD",
        FieldKind.CATEGORICAL: "string",
        FieldKind.MONETARY: "number",
        FieldKind.PER_SHARE: "number",
        FieldKind.SHARE_COUNT: "number",
    }
    lines = []
    for spec in SCHEMA:
        kind = "one of " + ", ".join(spec.choices) if spec.choices is not None else kinds[spec.kind]
        lines.append(f"  {spec.name} ({kind}, or null): {spec.description}")
    return "\n".join(lines)
