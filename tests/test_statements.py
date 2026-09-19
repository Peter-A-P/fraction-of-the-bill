"""The locator, against documents built to mislead it.

Every test here is a way the wrong table could be taken for a statement, or a way the
right one could be read at the wrong scale. Either becomes a wrong label, and a wrong label
is invisible downstream: the grader scores the model against it and reports the result
with a confidence interval.
"""

from __future__ import annotations

import pytest
from filings import HIDDEN_SENTINEL, HIGHLIGHTS_CASH, document

from smallprint.data.statements import blocks, locate, locate_statement
from smallprint.schema import Section


def test_the_inline_xbrl_header_never_reaches_the_text() -> None:
    """It carries the filing's own tagged facts. Left in, it would put the answers in the question."""
    doc = document()
    assert HIDDEN_SENTINEL.encode() in doc
    assert all(HIDDEN_SENTINEL not in b.render() for b in blocks(doc))
    assert HIDDEN_SENTINEL not in locate(doc).render()


def test_display_none_content_is_not_read() -> None:
    doc = b'<html><body><p>shown</p><div style="display: none">secret</div></body></html>'
    rendered = " ".join(b.render() for b in blocks(doc))
    assert "shown" in rendered
    assert "secret" not in rendered


def test_the_statement_is_chosen_over_the_highlights_table_that_precedes_it() -> None:
    located = locate(document())
    assert located.income is not None and located.balance is not None
    assert "901,300" in located.income.render()  # the nine-month column only the statement has
    assert HIGHLIGHTS_CASH not in located.balance.render()


def test_an_untitled_statement_still_outscores_a_summary_with_fewer_lines() -> None:
    located = locate(document(income_title=""))
    assert located.income is not None
    assert "Provision for income taxes" in located.income.render()


def test_net_loss_income_with_parentheses_is_still_net_income() -> None:
    """Lifetime Brands labels the line "Net (loss) income"; found on the first live build."""
    rows = "".join(
        f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>"
        for label, a, b in (
            ("Net sales", "150,000", "140,000"),
            ("Income from operations", "5,000", "4,000"),
            ("Net (loss) income", "(1,000)", "900"),
            ("Diluted (loss) income per common share", "(0.05)", "0.04"),
        )
    )
    doc = f"<html><body><table>{rows}</table></body></html>".encode()
    assert locate_statement(Section.INCOME, blocks(doc)) is not None


def test_dollar_signs_and_parentheses_rejoin_their_numbers() -> None:
    located = locate(document())
    assert located.income is not None
    assert "Net income (loss) | $(4,520) | $12,700" in located.income.render()


@pytest.mark.parametrize("titled", [True, False])
def test_a_balance_sheet_broken_across_a_page_is_read_to_the_end(titled: bool) -> None:
    located = locate(document(continuation_titled=titled))
    assert located.balance is not None
    rendered = located.balance.render()
    assert "Total assets" in rendered
    assert "Total liabilities | 140,000" in rendered
    assert "Total stockholders" in rendered


def test_the_cash_flow_statement_is_never_appended_to_the_balance_sheet() -> None:
    located = locate(document(balance_split=False))
    assert located.balance is not None
    assert "Net cash provided" not in located.balance.render()


def test_a_table_titled_as_another_statement_is_never_taken() -> None:
    """The cash flow statement has net income and cash; it is still the cash flow statement."""
    rows = "".join(
        f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>"
        for label, a, b in (
            ("Net income", "1,000", "900"),
            ("Net income per share, diluted", "0.10", "0.09"),
            ("Revenue recognised from deferred balances", "5,000", "4,000"),
            ("Income taxes paid", "300", "200"),
            ("Weighted-average shares", "10,000", "10,000"),
        )
    )
    doc = (
        "<html><body><p>CONSOLIDATED STATEMENTS OF CASH FLOWS</p>"
        f"<table>{rows}</table></body></html>"
    ).encode()
    assert locate_statement(Section.INCOME, blocks(doc)) is None


def test_a_standalone_comprehensive_income_statement_is_not_the_income_statement() -> None:
    """It starts from net income and has no per-share lines."""
    rows = "".join(
        f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>"
        for label, a, b in (
            ("Net income", "1,000", "900"),
            ("Foreign currency translation, net of income tax", "(50)", "20"),
            ("Unrealised gains on securities", "10", "5"),
            ("Total other comprehensive income (loss)", "(40)", "25"),
            ("Comprehensive income", "960", "925"),
        )
    )
    doc = (
        "<html><body><p>CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME</p>"
        f"<table>{rows}</table></body></html>"
    ).encode()
    assert locate_statement(Section.INCOME, blocks(doc)) is None


def test_a_combined_statement_of_comprehensive_income_is_the_income_statement() -> None:
    """Deckers, on the first live build: the income statement is titled "comprehensive
    income", and excluding it handed the choice to the discussion section's results table."""
    statement_rows = (
        ("Net sales", "1,827,166", "1,560,264"),
        ("Income from operations", "566,971", "478,391"),
        ("Income tax expense", "(130,000)", "(110,000)"),
        ("Net income", "456,653", "396,154"),
        ("Diluted net income per share", "3.00", "2.52"),
        ("Weighted-average common shares outstanding, diluted", "152,386", "157,000"),
        ("Other comprehensive loss, net of tax", "(5,000)", "(3,000)"),
    )
    results_rows = (
        ("Net sales", "1,827,166", "1,560,264"),
        ("Income from operations", "566,971", "478,391"),
        ("Net income", "456,653", "396,154"),
        ("Diluted", "3.00", "2.52"),
    )

    def table(rows: tuple[tuple[str, str, str], ...]) -> str:
        body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
        return f"<table>{body}</table>"

    doc = (
        "<html><body><p>RESULTS OF OPERATIONS</p>"
        + table(results_rows)
        + "<p>CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME</p>"
        + "<p>(amounts in thousands, except per share data)</p>"
        + table(statement_rows)
        + "</body></html>"
    ).encode()
    statement = locate_statement(Section.INCOME, blocks(doc))
    assert statement is not None
    assert any("Weighted-average" in row[0] for row in statement.rows)
    assert statement.heading[0] == "CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"


def test_a_table_of_contents_naming_every_statement_is_not_a_statement() -> None:
    doc = (
        b"<html><body><table>"
        b"<tr><td>Consolidated Statements of Operations</td><td>45</td></tr>"
        b"<tr><td>Consolidated Balance Sheets</td><td>46</td></tr>"
        b"<tr><td>Net income per share</td><td>47</td></tr>"
        b"</table></body></html>"
    )
    items = blocks(doc)
    assert locate_statement(Section.INCOME, items) is None
    assert locate_statement(Section.BALANCE, items) is None


@pytest.mark.parametrize(
    ("note", "scale", "declared", "share_scale"),
    [
        ("(In thousands, except per share data)", 1e3, True, None),
        ("(Dollars in thousands, except per share amounts)", 1e3, True, None),
        ("(In millions, except share and per share data)", 1e6, True, 1.0),
        ("(in thousands, except shares and per-share amounts)", 1e3, True, 1.0),
        (
            "(In millions, except number of shares, which are reflected in thousands, "
            "and per-share amounts)",
            1e6,
            True,
            1e3,
        ),
        ("($000's omitted, except per share data)", 1e3, True, None),
        ("(In billions)", 1e9, True, None),
        ("", 1.0, False, None),
    ],
)
def test_the_scale_is_read_from_the_heading(
    note: str, scale: float, declared: bool, share_scale: float | None
) -> None:
    located = locate(document(scale_note=note))
    assert located.income is not None
    assert located.income.scale == scale
    assert located.income.scale_declared is declared
    assert located.income.share_scale == share_scale


def test_the_audit_report_is_not_mistaken_for_the_statement_heading() -> None:
    """In a 10-K the opinion runs straight into the statements; its date is not the heading."""
    located = locate(document(annual=True))
    assert located.income is not None
    assert located.income.heading[0] == "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS"
    assert "February 20, 2027" not in located.income.render()


def test_the_auditor_block_is_found_in_an_annual_report_and_absent_from_a_quarterly_one() -> None:
    assert "/s/ Deloitte & Touche LLP" in locate(document(annual=True)).auditor
    assert locate(document()).auditor == ""


def test_the_cover_page_carries_the_state_and_stops_at_the_contents() -> None:
    cover = locate(document()).cover
    assert "Delaware" in cover
    assert "September 30, 2026" in cover
    assert "TABLE OF CONTENTS" not in cover


def test_the_rendered_input_labels_sections_and_leaves_the_scale_where_it_was_printed() -> None:
    text = locate(document(annual=True)).render()
    positions = [
        text.index(f"[{s}]") for s in ("COVER PAGE", "INCOME STATEMENT", "BALANCE SHEET", "AUDITOR")
    ]
    assert positions == sorted(positions)
    assert "(In thousands, except per share data)" in text


def test_a_title_and_scale_printed_inside_the_table_are_read_there() -> None:
    """ONEOK's 10-K, on the first live build: company, title, years and then the scale, all
    rows of the statement table, with the audit report's last lines above it."""
    rows = [
        ("ONEOK, Inc. and Subsidiaries",),
        ("CONSOLIDATED STATEMENTS OF INCOME",),
        ("Years Ended Dec. 31,",),
        ("2024", "2023"),
        ("(Millions of dollars, except per share amounts )",),
        ("Total revenues", "21,698", "17,677"),
        ("Operating income", "5,000", "4,000"),
        ("Income taxes", "(900)", "(800)"),
        ("Net income", "3,112", "2,659"),
        ("Diluted earnings per share", "5.17", "5.48"),
    ]
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    doc = (
        "<html><body><p>We have served as the Company's auditor since 2007.</p><p>66</p>"
        f"<table>{body}</table></body></html>"
    ).encode()
    statement = locate_statement(Section.INCOME, blocks(doc))
    assert statement is not None
    assert statement.scale == 1e6
    assert statement.heading == ()


@pytest.mark.parametrize(
    "tenure",
    [
        "We have served as Aditxt's auditor since 2021.",
        "We have served as the auditor of the Company since 2019.",
        "We have served as the Company's independent registered public accounting firm since 2008.",
    ],
)
def test_the_audit_signature_is_found_under_the_wordings_filers_use(tenure: str) -> None:
    """On the 300-company build the one fixed phrase missed a third of the auditors."""
    doc = (
        "<html><body><p>/s/ dbbmckennon</p><p>Newport Beach, California</p>"
        f"<p>{tenure}</p></body></html>"
    ).encode()
    assert "/s/ dbbmckennon" in locate(doc).auditor


def test_a_signature_several_lines_above_the_tenure_sentence_is_kept_and_prose_is_not() -> None:
    long_paragraph = "Critical audit matters are matters arising from the audit. " * 5
    doc = (
        "<html><body><p>/s/ KPMG LLP</p><p>PCAOB ID 185</p><p>Toronto, Canada</p>"
        f"<p>{long_paragraph}</p><p>March 1, 2026</p><p>Page 70</p>"
        "<p>We have served as the Company's auditor since 2016.</p></body></html>"
    ).encode()
    auditor = locate(doc).auditor
    assert "/s/ KPMG LLP" in auditor
    assert "Critical audit matters" not in auditor


def test_both_reports_are_read_after_a_change_of_auditor() -> None:
    doc = (
        "<html><body>"
        "<p>/s/ Fruci &amp; Associates II, PLLC</p>"
        "<p>We have served as the Company's auditor since 2025.</p>"
        + "<p>"
        + "Filler paragraph between the two reports, long enough to be prose. " * 4
        + "</p>"
        + "<p>A</p><p>B</p><p>C</p><p>D</p><p>E</p><p>F</p><p>G</p><p>H</p><p>I</p>"
        "<p>/s/ Astra Audit &amp; Advisory LLC</p>"
        "<p>We served as the Company's auditor from 2022 to 2024. We have served as the "
        "auditor of the Company since 2022.</p>"
        "</body></html>"
    ).encode()
    auditor = locate(doc).auditor
    assert "Fruci" in auditor
    assert "Astra" in auditor


@pytest.mark.parametrize(
    ("label", "scale"),
    [
        ("Weighted-average number of common shares outstanding (in thousands):", 1e3),
        ("Weighted Average Shares Outstanding (000" + chr(0x2019) + "s):", 1e3),
        ("Diluted shares outstanding, in millions", 1e6),
    ],
)
def test_a_scale_in_the_share_rows_own_label_overrides_the_heading(
    label: str, scale: float
) -> None:
    """Insulet and Linde, on the 300-company build."""
    rows = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
        for r in (
            ("Total revenue", "2,071.6", "1,697.1"),
            ("Operating income", "400.1", "300.2"),
            ("Income tax expense", "(50.0)", "(40.0)"),
            ("Net income", "418.3", "206.3"),
            ("Diluted net income per share", "5.87", "2.95"),
            (label,),
            ("Diluted", "71,213", "70,166"),
        )
    )
    doc = (
        # Declared, as filings declare it: the label carries a curly apostrophe.
        '<html><head><meta charset="utf-8"></head><body><p>CONSOLIDATED STATEMENTS OF INCOME</p>'
        "<p>(in millions, except share and per share data)</p>"
        f"<table>{rows}</table></body></html>"
    ).encode()
    statement = locate_statement(Section.INCOME, blocks(doc))
    assert statement is not None
    assert statement.share_scale == scale


def test_a_sentence_introducing_a_table_is_not_its_title() -> None:
    """A discussion-section percentage table, introduced by a sentence naming the income
    statement, outscored the statement itself on the 300-company build."""
    summary = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>"
        for a, b, c in (
            ("Net sales", "100.0%", "100.0%"),
            ("Operating income", "12.1", "11.0"),
            ("Income tax expense", "2.0", "1.9"),
            ("Net income", "9.1", "8.2"),
            ("Diluted net income per share", "1.10", "0.98"),
        )
    )
    doc = (
        "<html><body><p>The following table sets forth the components of our Consolidated "
        "Statements of Income as a percentage of net sales:</p>"
        f"<table>{summary}</table></body></html>"
    ).encode()
    items = blocks(doc)
    assert locate_statement(Section.INCOME, items) is not None
    statement = locate_statement(Section.INCOME, items)
    assert statement is not None and statement.heading == ()


def test_html_comments_are_never_read_but_the_text_after_them_is() -> None:
    """One filing agent's comments reached the cover page on the 300-company build, and
    anything a filer leaves in a comment would have reached the model's input."""
    doc = (
        b"<html><body><!-- Field: Set; Name: xdx; ID: xdx_021 -->"
        b"<p>Table of Contents</p><p>FORM 10-Q</p>"
        b"<p>For the quarterly period ended <!-- a note -->January 31, 2025</p>"
        b"</body></html>"
    )
    rendered = " ".join(b.render() for b in blocks(doc))
    assert "Field: Set" not in rendered
    assert "a note" not in rendered
    assert "For the quarterly period ended January 31, 2025" in rendered
    assert "January 31, 2025" in locate(doc).cover
