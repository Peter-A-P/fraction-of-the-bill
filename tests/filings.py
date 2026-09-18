"""Synthetic filing documents with the traps real ones set, and the facts that label them.

Nothing here is a real filing; nothing raw is committed. The documents are built to the shape
EDGAR inline XBRL actually takes, and each carries the things the locator has to get past:

* a hidden `ix:header` holding the filing's own tagged facts, with a sentinel number that
  must never reach the model's input;
* a table of contents that names every statement and carries only page numbers;
* a highlights table in the discussion section, placed before the statements, carrying the
  same line items with a sentinel cash figure the balance sheet does not print;
* dollar signs and the parentheses of negatives in cells of their own;
* a balance sheet broken across a page, with a page number and repeated company name between;
* a cash flow statement, which has net income and cash and cash equivalents and is neither
  statement the task reads.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from smallprint.data.xbrl import Fact

NBSP = chr(0xA0)
EM_DASH = chr(0x2014)
RIGHT_QUOTE = chr(0x2019)

#: In the hidden header only. If it reaches the rendered text, hidden content leaked.
HIDDEN_SENTINEL = "987,654"
#: In the highlights table only. If it is found in the balance sheet, the wrong table won.
HIGHLIGHTS_CASH = "77,777"

CIK = 1234567
QUARTERLY_ACCESSION = "0001234567-26-000042"
ANNUAL_ACCESSION = "0001234567-27-000007"

Q3_END = dt.date(2026, 9, 30)
FY_END = dt.date(2026, 12, 31)


def _cell(text: str) -> str:
    return f"<td>{text}</td>"


def _money_row(label: str, values: Sequence[str], dollar: bool = False) -> str:
    """A statement row laid out as filers do: `$` and `)` in cells of their own."""
    cells = [_cell(label)]
    for value in values:
        if dollar:
            cells.append(_cell("$"))
        if value.startswith("("):
            cells += [_cell("("), _cell(value.strip("()")), _cell(")")]
        else:
            cells += [_cell(value), _cell(NBSP)]
    return "<tr>" + "".join(cells) + "</tr>"


def _table(rows: Sequence[str]) -> str:
    return "<table>" + "".join(rows) + "</table>"


def _p(text: str, style: str = "") -> str:
    attr = f' style="{style}"' if style else ""
    return f"<p{attr}>{text}</p>"


def document(
    *,
    annual: bool = False,
    scale_note: str = "(In thousands, except per share data)",
    balance_scale_note: str = "(In thousands)",
    balance_split: bool = True,
    continuation_titled: bool = True,
    income_title: str = "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
    with_highlights: bool = True,
    with_hidden_header: bool = True,
    share_label: str = "Weighted-average shares outstanding, diluted",
    shares: Sequence[str] = ("14,580", "14,760", "14,560", "14,790"),
    auditor: str = "Deloitte &amp; Touche LLP",
) -> bytes:
    period = (
        "fiscal year ended December 31, 2026"
        if annual
        else "quarterly period ended September 30, 2026"
    )
    form = "10-K" if annual else "10-Q"

    hidden = (
        '<div style="display:none"><ix:header><ix:hidden>'
        '<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">'
        + ("FY" if annual else "Q3")
        + "</ix:nonNumeric>"
        f'<ix:nonFraction name="us-gaap:Revenues">{HIDDEN_SENTINEL}</ix:nonFraction>'
        "</ix:hidden></ix:header></div>"
        if with_hidden_header
        else ""
    )

    cover = (
        _p("UNITED STATES SECURITIES AND EXCHANGE COMMISSION", "text-align:center")
        + _p(f"FORM {form}")
        + _p(f"For the {period}")
        + _p("EXAMPLE WIDGETS, INC.")
        + _table(
            [
                "<tr>" + _cell("Delaware") + _cell("94-1234567") + "</tr>",
                "<tr>"
                + _cell("(State or other jurisdiction of incorporation)")
                + _cell("(I.R.S. Employer Identification No.)")
                + "</tr>",
            ]
        )
        + _p(
            "Indicate by check mark whether the registrant (1) has filed all reports required "
            "to be filed by Section 13 or 15(d) of the Securities Exchange Act of 1934."
        )
    )

    contents = _p("TABLE OF CONTENTS") + _table(
        [
            "<tr>" + _cell(f"Condensed Consolidated {name}") + _cell(page) + "</tr>"
            for name, page in (
                ("Statements of Operations", "3"),
                ("Balance Sheets", "4"),
                ("Statements of Cash Flows", "5"),
            )
        ]
    )

    highlights = (
        _p(
            "The following table summarises our results for the periods presented and should "
            "be read together with the condensed consolidated financial statements and the "
            "notes to them included elsewhere in this report, which are the authoritative "
            "source for every figure shown here and for the accounting policies behind them."
        )
        + _table(
            [
                _money_row("Revenue", ["312,450", "298,100"], dollar=True),
                _money_row("Operating income", ["25,300", "30,100"]),
                _money_row("Net income (loss)", ["(4,520)", "12,700"]),
                _money_row("Diluted net income (loss) per share", ["(0.31)", "0.86"]),
                _money_row("Cash position", [HIGHLIGHTS_CASH, "28,500"]),
            ]
        )
        if with_highlights
        else ""
    )

    if annual:
        header = [
            "<tr><td></td><td colspan='4'>Year Ended December 31,</td></tr>",
            "<tr><td></td><td colspan='2'>2026</td><td colspan='2'>2025</td></tr>",
        ]
        body = [
            ("Revenue", ["1,234,567", "1,100,400"], True),
            ("Cost of revenue", ["800,000", "760,000"], False),
            ("Operating income", ["180,000", "150,000"], False),
            ("Provision for income taxes", ["21,000", "19,000"], False),
            ("Net income (loss)", ["(45,200)", "22,700"], True),
            ("Net income (loss) per share, basic", ["(0.31)", "0.16"], True),
            ("Net income (loss) per share, diluted", ["(0.31)", "0.15"], True),
            (share_label, list(shares[:2]), False),
        ]
    else:
        header = [
            "<tr><td></td><td colspan='4'>Three Months Ended September 30,</td>"
            "<td colspan='4'>Nine Months Ended September 30,</td></tr>",
            "<tr><td></td>"
            + "".join(f"<td colspan='2'>{y}</td>" for y in ("2026", "2025", "2026", "2025"))
            + "</tr>",
        ]
        body = [
            ("Revenue", ["312,450", "298,100", "901,300", "850,200"], True),
            ("Cost of revenue", ["180,200", "171,000", "520,000", "495,000"], False),
            ("Operating income", ["25,300", "30,100", "70,000", "82,000"], False),
            ("Provision for income taxes", ["3,100", "4,000", "9,000", "11,000"], False),
            ("Net income (loss)", ["(4,520)", "12,700", "(10,300)", "40,100"], True),
            ("Net income (loss) per share, basic", ["(0.31)", "0.88", "(0.71)", "2.78"], True),
            ("Net income (loss) per share, diluted", ["(0.31)", "0.86", "(0.71)", "2.71"], True),
            (share_label, list(shares), False),
        ]
    income = (
        _p("EXAMPLE WIDGETS, INC.")
        + _p(income_title)
        + _p(scale_note)
        + _p("(Unaudited)" if not annual else "")
        + _table(header + [_money_row(label, v, dollar) for label, v, dollar in body])
    )

    at = "December 31, 2026" if annual else "September 30, 2026"
    assets = [
        f"<tr><td></td><td colspan='2'>{at}</td><td colspan='2'>December 31, 2025</td></tr>",
        _money_row("Cash and cash equivalents", ["31,000", "28,500"], dollar=True),
        _money_row("Accounts receivable, net", ["40,000", "38,000"]),
        _money_row("Total current assets", ["150,000", "140,000"]),
        _money_row("Property and equipment, net", ["100,000", "91,000"]),
        _money_row("Total assets", ["250,000", "231,000"], dollar=True),
    ]
    claims = [
        _money_row("Total current liabilities", ["60,000", "55,000"]),
        _money_row("Long-term debt", ["80,000", "65,000"]),
        _money_row("Total liabilities", ["140,000", "120,000"]),
        _money_row("Common stock", ["15", "15"]),
        _money_row("Retained earnings", ["109,985", "110,985"]),
        _money_row(f"Total stockholders{RIGHT_QUOTE} equity", ["110,000", "111,000"]),
        _money_row(
            f"Total liabilities and stockholders{RIGHT_QUOTE} equity",
            ["250,000", "231,000"],
            dollar=True,
        ),
    ]
    balance_head = (
        _p("EXAMPLE WIDGETS, INC.")
        + _p("CONDENSED CONSOLIDATED BALANCE SHEETS")
        + _p(balance_scale_note)
    )
    if balance_split:
        balance = (
            balance_head
            + _table(assets)
            + _p("4")
            + _p("EXAMPLE WIDGETS, INC.")
            + (
                _p("CONDENSED CONSOLIDATED BALANCE SHEETS (continued)")
                if continuation_titled
                else ""
            )
            + _table(claims)
        )
    else:
        balance = balance_head + _table(assets + claims)

    cash_flows = (
        _p("CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS")
        + _p("(In thousands)")
        + _table(
            [
                _money_row("Net income (loss)", ["(10,300)", "40,100"], dollar=True),
                _money_row("Depreciation and amortization", ["12,000", "11,000"]),
                _money_row("Net cash provided by operating activities", ["20,000", "55,000"]),
                _money_row("Cash and cash equivalents at end of period", ["31,000", "26,000"]),
                _money_row("Income taxes paid", ["8,000", "9,000"]),
            ]
        )
    )

    audit = (
        _p("REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM")
        + _p(
            "In our opinion, the financial statements present fairly, in all material "
            "respects, the financial position of the Company as of December 31, 2026."
        )
        + _p(f"/s/ {auditor}")
        + _p(f"We have served as the Company{RIGHT_QUOTE}s auditor since 2011.")
        + _p("San Jose, California")
        + _p("February 20, 2027")
        if annual
        else ""
    )

    parts = [
        hidden,
        cover,
        contents,
        _p("PART I. FINANCIAL INFORMATION"),
        highlights,
        audit,
        income,
        balance,
        cash_flows,
    ]
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">'
        "<head><title>exw-20260930.htm</title></head><body>" + "".join(parts) + "</body></html>"
    ).encode("utf-8")


def _duration(
    concept: str, value: float | str, start: dt.date, end: dt.date, accession: str, form: str
) -> Fact:
    return Fact(
        concept=concept,
        unit="USD",
        value=value,
        start=start,
        end=end,
        accession=accession,
        form=form,
        filed=end + dt.timedelta(days=35),
    )


def _instant(concept: str, value: float, end: dt.date, accession: str, form: str) -> Fact:
    return Fact(
        concept=concept,
        unit="USD",
        value=value,
        start=None,
        end=end,
        accession=accession,
        form=form,
        filed=end + dt.timedelta(days=35),
    )


def facts(*, annual: bool = False, overrides: dict[str, float | str] | None = None) -> list[Fact]:
    """The facts this filing tagged, current period first, prior-period columns after.

    `overrides` replaces the current-period value of a concept, which is how the tests put a
    label somewhere the page does not print it.
    """
    override = overrides or {}
    if annual:
        accession, form, end = ANNUAL_ACCESSION, "10-K", FY_END
        start, prior_start, prior_end = (
            dt.date(2026, 1, 1),
            dt.date(2025, 1, 1),
            dt.date(2025, 12, 31),
        )
        fiscal = "FY"
        current = {
            "us-gaap:Revenues": 1_234_567_000.0,
            "us-gaap:CostOfRevenue": 800_000_000.0,
            "us-gaap:OperatingIncomeLoss": 180_000_000.0,
            "us-gaap:NetIncomeLoss": -45_200_000.0,
            "us-gaap:EarningsPerShareBasic": -0.31,
            "us-gaap:EarningsPerShareDiluted": -0.31,
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 14_580_000.0,
        }
        prior = {"us-gaap:Revenues": 1_100_400_000.0, "us-gaap:NetIncomeLoss": 22_700_000.0}
    else:
        accession, form, end = QUARTERLY_ACCESSION, "10-Q", Q3_END
        start, prior_start, prior_end = (
            dt.date(2026, 7, 1),
            dt.date(2025, 7, 1),
            dt.date(2025, 9, 30),
        )
        fiscal = "Q3"
        current = {
            "us-gaap:Revenues": 312_450_000.0,
            "us-gaap:CostOfRevenue": 180_200_000.0,
            "us-gaap:OperatingIncomeLoss": 25_300_000.0,
            "us-gaap:NetIncomeLoss": -4_520_000.0,
            "us-gaap:EarningsPerShareBasic": -0.31,
            "us-gaap:EarningsPerShareDiluted": -0.31,
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 14_580_000.0,
        }
        prior = {"us-gaap:Revenues": 298_100_000.0, "us-gaap:NetIncomeLoss": 12_700_000.0}

    balance = {
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": 31_000_000.0,
        "us-gaap:Assets": 250_000_000.0,
        "us-gaap:Liabilities": 140_000_000.0,
        "us-gaap:StockholdersEquity": 110_000_000.0,
    }
    cover: dict[str, float | str] = {
        "dei:DocumentPeriodEndDate": end.isoformat(),
        "dei:DocumentFiscalPeriodFocus": fiscal,
        "dei:EntityIncorporationStateCountryCode": "DE",
    }
    if annual:
        cover["dei:AuditorName"] = "Deloitte & Touche LLP"

    out: list[Fact] = []
    for concept, value in {**current, **cover}.items():
        out.append(_duration(concept, override.get(concept, value), start, end, accession, form))
    for concept, value in prior.items():
        out.append(_duration(concept, value, prior_start, prior_end, accession, form))
    for concept, value in balance.items():
        chosen = override.get(concept, value)
        assert isinstance(chosen, float)
        out.append(_instant(concept, chosen, end, accession, form))
        out.append(_instant(concept, value * 0.92, dt.date(2025, 12, 31), accession, form))
    return out
