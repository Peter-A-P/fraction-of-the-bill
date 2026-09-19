"""Pairing and the locatability filter.

The filter's job is to refuse. Most of these tests put a label somewhere the model cannot
read it, on the page but in the wrong section, hidden in the markup, or at a scale the page
does not declare, and assert that the filing is dropped with that reason named.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import filings
import pytest

from smallprint.data.pair import (
    Dropped,
    DropReason,
    Item,
    locatable,
    pair_filing,
    printed_numbers,
    tally,
)
from smallprint.data.statements import LocatedFiling
from smallprint.data.xbrl import Fact
from smallprint.grade import GradeContext, grade_item
from smallprint.schema import FIELDS


def pair(
    *,
    annual: bool = False,
    facts: list[Fact] | None = None,
    **document: Any,
) -> Item | Dropped:
    return pair_filing(
        filings.document(annual=annual, **document),
        facts if facts is not None else filings.facts(annual=annual),
        cik=filings.CIK,
        accession=filings.ANNUAL_ACCESSION if annual else filings.QUARTERLY_ACCESSION,
        form="10-K" if annual else "10-Q",
        filed=dt.date(2026, 11, 4),
    )


def kept(result: Item | Dropped) -> Item:
    assert isinstance(result, Item), result
    return result


def dropped(result: Item | Dropped) -> Dropped:
    assert isinstance(result, Dropped), result
    return result


@pytest.mark.parametrize("annual", [False, True])
def test_a_clean_filing_becomes_an_item_with_its_truth_and_scale(annual: bool) -> None:
    item = kept(pair(annual=annual))
    assert item.fiscal_period == ("FY" if annual else "Q3")
    assert item.truth.revenue == (1_234_567_000 if annual else 312_450_000)
    assert item.truth.net_income == (-45_200_000 if annual else -4_520_000)
    assert item.truth.state_of_incorporation == "DE"
    assert item.context.scale == 1e3
    assert filings.HIDDEN_SENTINEL not in item.text


def test_the_prior_year_column_travels_with_the_item_as_a_distractor() -> None:
    item = kept(pair())
    assert 298_100_000 in item.context.distractors["revenue"]


def test_an_item_graded_against_itself_is_exact_and_survives_serialisation() -> None:
    item = kept(pair(annual=True))
    assert grade_item(item.item_id, item.truth, item.truth, item.context).exact
    assert Item.model_validate_json(item.model_dump_json()) == item


def test_a_label_the_page_does_not_print_is_dropped() -> None:
    result = dropped(pair(facts=filings.facts(overrides={"us-gaap:Revenues": 555_555_000.0})))
    assert result.reason is DropReason.UNLOCATABLE
    assert result.field == "revenue"


def test_a_label_printed_only_in_the_hidden_header_is_dropped() -> None:
    hidden = float(filings.HIDDEN_SENTINEL.replace(",", "")) * 1000
    result = dropped(pair(facts=filings.facts(overrides={"us-gaap:Revenues": hidden})))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "revenue")


def test_a_label_printed_only_in_the_discussion_section_is_dropped() -> None:
    """The highlights table prints this cash figure. The balance sheet does not."""
    cash = float(filings.HIGHLIGHTS_CASH.replace(",", "")) * 1000
    overrides: dict[str, float | str] = {"us-gaap:CashAndCashEquivalentsAtCarryingValue": cash}
    result = dropped(pair(facts=filings.facts(overrides=overrides)))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "cash_and_equivalents")


def test_a_number_printed_in_the_other_statement_does_not_count() -> None:
    """Total assets is on the page. It is not on the income statement, where this field is read."""
    overrides: dict[str, float | str] = {"us-gaap:OperatingIncomeLoss": 250_000_000.0}
    result = dropped(pair(facts=filings.facts(overrides=overrides)))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "operating_income")


def test_a_statement_in_thousands_that_does_not_say_so_is_dropped_not_mislabelled() -> None:
    result = dropped(pair(scale_note="", balance_scale_note=""))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "revenue")


def test_a_label_within_the_grader_tolerance_of_the_page_is_kept() -> None:
    """312,999 thousand is 0.18% from the printed 312,450. The grader would accept that reading,
    so the filter does too: the two share one rule about what a correct reading is."""
    kept(pair(facts=filings.facts(overrides={"us-gaap:Revenues": 312_999_000.0})))


def test_statements_that_declare_different_scales_are_dropped() -> None:
    result = dropped(pair(scale_note="(In millions, except per share data)"))
    assert result.reason is DropReason.MIXED_SCALE


def test_shares_printed_whole_under_an_except_share_heading_are_found() -> None:
    item = kept(
        pair(
            scale_note="(In thousands, except share and per share data)",
            shares=("14,580,000", "14,760,000", "14,560,000", "14,790,000"),
        )
    )
    assert item.context.share_scale == 1.0


def test_shares_printed_whole_under_a_heading_that_scales_only_money_are_found() -> None:
    """St. Joe's statements, on the first live build: dollars in thousands, shares whole."""
    item = kept(
        pair(
            scale_note="(Dollars in thousands except per share amounts)",
            shares=("14,580,000", "14,760,000", "14,560,000", "14,790,000"),
        )
    )
    assert item.context.share_scale == 1.0


def test_shares_printed_in_thousands_under_an_except_share_heading_are_dropped() -> None:
    """The heading says shares are whole; the page prints them in thousands. The label is not there."""
    result = dropped(pair(scale_note="(In thousands, except share and per share data)"))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "shares_diluted")


def test_an_auditor_printed_differently_but_equivalently_is_found() -> None:
    kept(pair(annual=True, auditor="DELOITTE AND TOUCHE, L.L.P."))


def test_an_auditor_the_signature_does_not_name_is_dropped() -> None:
    result = dropped(pair(annual=True, auditor="Ernst &amp; Young LLP"))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "auditor_name")


def test_a_state_code_the_cover_page_does_not_spell_out_is_dropped() -> None:
    overrides: dict[str, float | str] = {"dei:EntityIncorporationStateCountryCode": "NV"}
    result = dropped(pair(facts=filings.facts(overrides=overrides)))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "state_of_incorporation")


def test_a_filing_without_its_cover_period_facts_is_dropped() -> None:
    facts = [f for f in filings.facts() if f.concept != "dei:DocumentPeriodEndDate"]
    assert dropped(pair(facts=facts)).reason is DropReason.NO_PERIOD


def test_a_filing_missing_a_required_fact_is_dropped_by_name() -> None:
    facts = [f for f in filings.facts() if f.concept != "us-gaap:NetIncomeLoss"]
    result = dropped(pair(facts=facts))
    assert (result.reason, result.field) == (DropReason.TRUTH_MISSING, "net_income")


def test_facts_from_another_filing_are_not_used() -> None:
    other = [f.model_copy(update={"accession": "0001234567-26-000001"}) for f in filings.facts()]
    assert dropped(pair(facts=other)).reason is DropReason.NO_PERIOD


@pytest.mark.parametrize(
    "printed", ["September 30 , 2026", "09/30/2026", "9/30/2026", "Sept. 30, 2026"]
)
def test_the_period_end_is_found_as_cover_pages_actually_print_it(printed: str) -> None:
    """The first two shapes cost 40 filings in 236 on the first live build."""
    located = LocatedFiling(
        cover=f"For the quarterly period ended {printed}", income=None, balance=None, auditor=""
    )
    assert locatable(FIELDS["period_end"], filings.Q3_END, located, GradeContext())


def test_printed_numbers_reads_magnitudes_and_dashes_for_zero() -> None:
    dash = filings.EM_DASH
    assert printed_numbers(f"Other | {dash} | $(1,234.5)") == (1234.5, 0.0)
    assert 0.0 not in printed_numbers("I.R.S. No. 94-1234567")


def test_the_tally_names_every_loss() -> None:
    results = [
        pair(),
        pair(scale_note="(In millions, except per share data)"),
        pair(facts=filings.facts(overrides={"us-gaap:Revenues": 555_555_000.0})),
        pair(facts=filings.facts(overrides={"us-gaap:Revenues": 444_444_000.0})),
    ]
    assert tally(results) == {"unlocatable:revenue": 2, "kept": 1, "mixed_scale": 1}


def test_of_two_revenue_synonyms_for_the_period_the_printed_one_is_the_label() -> None:
    """Hasbro, on the 300-company build: Revenues tagged at 4,745.9m, the statement's "Net
    revenues" line 4,135.5m, which is their contract revenue. The page decides."""
    facts = filings.facts(overrides={"us-gaap:Revenues": 555_555_000.0})
    tagged = next(f for f in facts if f.concept == "us-gaap:Revenues" and f.value == 555_555_000.0)
    contract = tagged.model_copy(
        update={
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "value": 312_450_000.0,
        }
    )
    item = kept(pair(facts=[*facts, contract]))
    assert item.truth.revenue == 312_450_000.0


def test_when_no_synonym_is_printed_the_filing_is_still_dropped() -> None:
    facts = filings.facts(overrides={"us-gaap:Revenues": 555_555_000.0})
    tagged = next(f for f in facts if f.concept == "us-gaap:Revenues" and f.value == 555_555_000.0)
    contract = tagged.model_copy(
        update={
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "value": 444_444_000.0,
        }
    )
    result = dropped(pair(facts=[*facts, contract]))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "revenue")


def test_diluted_shares_with_no_line_on_the_statement_are_labelled_null_not_dropped() -> None:
    """Bristol-Myers Squibb and 3% of filings print weighted shares only in a note. For the
    page the model is shown, "not reported" is the right answer (decided 2026-09-19)."""
    overrides: dict[str, float | str] = {
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 99_999_000.0
    }
    item = kept(pair(facts=filings.facts(overrides=overrides), share_label="Other items, net"))
    assert item.truth.shares_diluted is None
    assert item.not_on_page == ("shares_diluted",)
    assert tally([item]) == {"kept": 1, "kept_with_null:shares_diluted": 1}


def test_a_printed_share_line_that_does_not_match_still_drops_the_filing() -> None:
    """Nulling here would mark a correct reading of the printed line as a hallucination."""
    overrides: dict[str, float | str] = {
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 99_999_000.0
    }
    result = dropped(pair(facts=filings.facts(overrides=overrides)))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "shares_diluted")


def test_only_the_named_optional_field_is_ever_nulled() -> None:
    """Cost of revenue is optional too, but a missing line for it still drops the filing."""
    overrides: dict[str, float | str] = {"us-gaap:CostOfRevenue": 99_999_000.0}
    result = dropped(pair(facts=filings.facts(overrides=overrides)))
    assert (result.reason, result.field) == (DropReason.UNLOCATABLE, "cost_of_revenue")
