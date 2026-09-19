"""Choosing the right fact out of the four a statement prints on one line.

The fixture is a third-quarter 10-Q, because that is the hardest honest case: revenue
appears for the three months just ended, for the nine months to date, and for both of the
prior year's matching periods. All four are correctly tagged. Exactly one is the label.
"""

from __future__ import annotations

import datetime as dt

import pytest

from smallprint.data.xbrl import (
    Fact,
    UnusableFacts,
    build_truth,
    parse_company_facts,
    select_fact,
)
from smallprint.schema import FIELDS

ACCESSION = "0000320193-26-000075"
PERIOD_END = dt.date(2026, 9, 26)


def duration(
    concept: str,
    value: float,
    start: dt.date,
    end: dt.date,
    accession: str = ACCESSION,
    unit: str = "USD",
) -> Fact:
    return Fact(
        concept=concept,
        unit=unit,
        value=value,
        start=start,
        end=end,
        accession=accession,
        form="10-Q",
        filed=dt.date(2026, 10, 30),
    )


def instant(
    concept: str, value: float, end: dt.date = PERIOD_END, accession: str = ACCESSION
) -> Fact:
    return Fact(
        concept=concept,
        unit="USD",
        value=value,
        start=None,
        end=end,
        accession=accession,
        form="10-Q",
        filed=dt.date(2026, 10, 30),
    )


REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"

#: What the income statement of a Q3 10-Q actually carries for one line.
FOUR_COLUMNS = [
    duration(REVENUE, 94_900_000_000, dt.date(2026, 6, 28), PERIOD_END),
    duration(REVENUE, 294_900_000_000, dt.date(2025, 9, 29), PERIOD_END),
    duration(REVENUE, 89_500_000_000, dt.date(2025, 6, 30), dt.date(2025, 9, 27)),
    duration(REVENUE, 274_500_000_000, dt.date(2024, 9, 30), dt.date(2025, 9, 27)),
]


def test_the_quarter_is_chosen_over_the_year_to_date() -> None:
    """Both end on the day the report ends. Only one of them is the quarter."""
    chosen, _ = select_fact(
        FOUR_COLUMNS, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert chosen is not None
    assert chosen.value == 94_900_000_000


def test_the_other_three_columns_become_distractors() -> None:
    _, others = select_fact(
        FOUR_COLUMNS, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert {f.value for f in others} == {
        294_900_000_000,
        89_500_000_000,
        274_500_000_000,
    }


def test_an_annual_report_takes_the_year_not_the_quarter() -> None:
    facts = [
        duration(REVENUE, 100_000_000_000, dt.date(2026, 6, 28), PERIOD_END),
        duration(REVENUE, 390_000_000_000, dt.date(2025, 9, 29), PERIOD_END),
    ]
    chosen, _ = select_fact(facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="FY")
    assert chosen is not None
    assert chosen.value == 390_000_000_000


def test_a_fifty_three_week_year_is_still_a_year() -> None:
    """Retail filers run 52 or 53 weeks, and the extra week must not disqualify the fact."""
    facts = [duration(REVENUE, 1.0, dt.date(2025, 9, 22), PERIOD_END)]
    chosen, _ = select_fact(facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="FY")
    assert chosen is not None


def test_a_period_end_a_few_days_off_the_cover_page_still_matches() -> None:
    """Filers tag the Saturday their year ended; the cover page says the month end."""
    facts = [duration(REVENUE, 1.0, dt.date(2025, 10, 1), dt.date(2026, 9, 30))]
    chosen, _ = select_fact(facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="FY")
    assert chosen is not None


def test_a_period_end_a_quarter_away_does_not_match() -> None:
    facts = [duration(REVENUE, 1.0, dt.date(2025, 7, 1), dt.date(2026, 6, 30))]
    chosen, others = select_fact(
        facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="FY"
    )
    assert chosen is None
    assert others  # the near miss is still offered as a distractor


def test_a_balance_sheet_line_is_an_instant_and_the_prior_column_is_a_distractor() -> None:
    facts = [
        instant("us-gaap:Assets", 365_000_000_000),
        instant("us-gaap:Assets", 352_000_000_000, end=dt.date(2025, 9, 27)),
    ]
    chosen, others = select_fact(
        facts, FIELDS["total_assets"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert chosen is not None
    assert chosen.value == 365_000_000_000
    assert [f.value for f in others] == [352_000_000_000]


def test_the_first_concept_the_filing_reports_wins_and_later_synonyms_are_not_consulted() -> None:
    """Total revenue over contract revenue, which can be a fraction of the top line.

    The Andersons, on the first live build: $371m of contract revenue against $2,659m of
    sales and merchandising revenues. Revenues is the total by definition.
    """
    facts = [
        duration(REVENUE, 371.0, dt.date(2026, 6, 28), PERIOD_END),
        duration("us-gaap:Revenues", 2659.0, dt.date(2026, 6, 28), PERIOD_END),
    ]
    chosen, others = select_fact(
        facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert chosen is not None
    assert chosen.value == 2659.0
    assert all(f.concept == "us-gaap:Revenues" for f in others)


def test_a_later_synonym_is_used_when_the_first_is_absent() -> None:
    facts = [duration(REVENUE, 99.0, dt.date(2026, 6, 28), PERIOD_END)]
    chosen, _ = select_fact(facts, FIELDS["revenue"], period_end=PERIOD_END, fiscal_period="Q3")
    assert chosen is not None
    assert chosen.value == 99.0


def test_a_field_the_filing_never_reports_is_absent_rather_than_approximated() -> None:
    chosen, others = select_fact(
        [], FIELDS["operating_income"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert chosen is None
    assert others == ()


def test_truth_is_built_only_from_the_filing_that_is_being_labelled() -> None:
    """A later filing restating the period is a different document saying a different thing."""
    facts = [
        *FOUR_COLUMNS,
        duration(REVENUE, 1.0, dt.date(2026, 6, 28), PERIOD_END, accession="0000-99-000001"),
        instant("us-gaap:Assets", 365_000_000_000),
        instant("us-gaap:CashAndCashEquivalentsAtCarryingValue", 28_000_000_000),
        instant("us-gaap:StockholdersEquity", 60_000_000_000),
        duration("us-gaap:NetIncomeLoss", 23_600_000_000, dt.date(2026, 6, 28), PERIOD_END),
        duration("dei:DocumentPeriodEndDate", 0.0, dt.date(2026, 6, 28), PERIOD_END).model_copy(
            update={"value": "2026-09-26"}
        ),
    ]
    truth, distractors = build_truth(
        facts, accession=ACCESSION, period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert truth.revenue == 94_900_000_000
    assert truth.net_income == 23_600_000_000
    assert truth.total_assets == 365_000_000_000
    assert truth.period_end == dt.date(2026, 9, 26)
    assert 1.0 not in distractors.get("revenue", ())


def test_the_distractors_a_grader_receives_never_include_the_answer() -> None:
    truth, distractors = build_truth(
        FOUR_COLUMNS, accession=ACCESSION, period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert truth.revenue not in distractors["revenue"]
    assert 294_900_000_000 in distractors["revenue"]


def test_company_facts_flatten_with_their_accession_kept() -> None:
    payload = {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2026-09-26",
                                "val": 365_000_000_000,
                                "accn": ACCESSION,
                                "form": "10-Q",
                                "filed": "2026-10-30",
                            }
                        ]
                    }
                }
            }
        },
    }
    facts = parse_company_facts(payload)
    assert len(facts) == 1
    assert facts[0].concept == "us-gaap:Assets"
    assert facts[0].accession == ACCESSION
    assert facts[0].is_instant


def test_an_entry_with_no_accession_is_skipped_not_repaired() -> None:
    """A fact that cannot be attributed to a filing cannot be that filing's label."""
    payload = {
        "facts": {
            "us-gaap": {
                "Assets": {"units": {"USD": [{"end": "2026-09-26", "val": 1}]}},
            }
        }
    }
    assert parse_company_facts(payload) == []


def test_a_payload_of_the_wrong_shape_fails_loudly() -> None:
    with pytest.raises(UnusableFacts, match="no facts section"):
        parse_company_facts({"cik": 1})
    with pytest.raises(UnusableFacts, match="not an object"):
        parse_company_facts([1, 2, 3])


def test_a_cover_fact_tagged_against_the_year_to_date_is_still_the_cover_fact() -> None:
    """A Q3 10-Q tags its cover page against the nine months to date. Found on the first
    live build, where requiring a quarter-length context dropped three filings in ten."""
    facts = [
        Fact(
            concept="dei:DocumentPeriodEndDate",
            unit="",
            value=PERIOD_END.isoformat(),
            start=dt.date(2025, 12, 28),
            end=PERIOD_END,
            accession=ACCESSION,
            form="10-Q",
            filed=dt.date(2026, 10, 30),
        )
    ]
    chosen, _ = select_fact(facts, FIELDS["period_end"], period_end=PERIOD_END, fiscal_period="Q3")
    assert chosen is not None


def test_a_filer_with_only_cash_has_its_cash_taken_as_cash_and_equivalents() -> None:
    facts = [instant("us-gaap:Cash", 10_603_681.0)]
    chosen, _ = select_fact(
        facts, FIELDS["cash_and_equivalents"], period_end=PERIOD_END, fiscal_period="Q3"
    )
    assert chosen is not None
    assert chosen.value == 10_603_681.0
