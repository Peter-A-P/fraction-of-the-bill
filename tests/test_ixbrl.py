"""Cover-page facts read from the filing's own inline XBRL tags."""

from __future__ import annotations

import datetime as dt

import filings
import pytest

from smallprint.data.ixbrl import cover_facts, parse_date
from smallprint.data.pair import Item, pair_filing
from smallprint.data.xbrl import Fact


def read(document: bytes, annual: bool = False) -> dict[str, float | str]:
    facts = cover_facts(
        document,
        accession=filings.ANNUAL_ACCESSION if annual else filings.QUARTERLY_ACCESSION,
        form="10-K" if annual else "10-Q",
        filed=dt.date(2026, 11, 4),
    )
    return {f.concept: f.value for f in facts}


def test_the_quarterly_cover_facts_are_read_and_normalised() -> None:
    assert read(filings.document()) == {
        "dei:DocumentPeriodEndDate": "2026-09-30",
        "dei:DocumentFiscalPeriodFocus": "Q3",
        "dei:EntityIncorporationStateCountryCode": "DE",
    }


def test_the_auditor_is_read_from_the_signature_of_an_annual_report() -> None:
    assert read(filings.document(annual=True), annual=True)["dei:AuditorName"] == (
        "Deloitte & Touche LLP"
    )


def test_a_fact_tagged_against_a_dimensional_context_is_not_taken() -> None:
    """The fixture tags NV against a subsidiary's context. The registrant is DE."""
    facts = cover_facts(filings.document(), accession="a", form="10-Q", filed=dt.date(2026, 11, 4))
    states = [f.value for f in facts if f.concept == "dei:EntityIncorporationStateCountryCode"]
    assert states == ["DE"]


def test_a_state_tagged_by_name_becomes_its_code() -> None:
    facts = read(filings.document(state_tag="Delaware"))
    assert facts["dei:EntityIncorporationStateCountryCode"] == "DE"


def test_a_fact_whose_context_is_not_declared_is_left_out() -> None:
    doc = (
        b'<html><body><ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="nowhere">'
        b"Q3</ix:nonNumeric></body></html>"
    )
    assert cover_facts(doc, accession="a", form="10-Q", filed=dt.date(2026, 11, 4)) == []


def test_an_empty_document_has_no_facts_rather_than_an_error() -> None:
    assert cover_facts(b"", accession="a", form="10-Q", filed=dt.date(2026, 11, 4)) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("September 27, 2025", dt.date(2025, 9, 27)),
        ("Sept. 27, 2025", dt.date(2025, 9, 27)),
        ("December" + chr(0xA0) + "31 , 2024", dt.date(2024, 12, 31)),
        ("SEPTEMBER 27 2025", dt.date(2025, 9, 27)),
        ("27 September 2025", dt.date(2025, 9, 27)),
        ("09/27/2025", dt.date(2025, 9, 27)),
        ("2025-09-27", dt.date(2025, 9, 27)),
        ("Septembre 27, 2025", None),
        ("February 30, 2025", None),
        ("the third quarter", None),
    ],
)
def test_cover_dates_are_parsed_in_the_shapes_filers_tag_them(
    text: str, expected: dt.date | None
) -> None:
    assert parse_date(text) == expected


def test_a_filing_pairs_with_its_cover_facts_taken_from_the_document() -> None:
    """The facts API carries the numbers; the document carries its own cover page."""
    document = filings.document(annual=True)
    numeric = [f for f in filings.facts(annual=True) if not f.concept.startswith("dei:")]
    cover: list[Fact] = cover_facts(
        document, accession=filings.ANNUAL_ACCESSION, form="10-K", filed=dt.date(2027, 2, 25)
    )
    result = pair_filing(
        document,
        [*cover, *numeric],
        cik=filings.CIK,
        accession=filings.ANNUAL_ACCESSION,
        form="10-K",
        filed=dt.date(2027, 2, 25),
    )
    assert isinstance(result, Item), result
    assert result.truth.auditor_name == "Deloitte & Touche LLP"
    assert result.truth.period_end == filings.FY_END
