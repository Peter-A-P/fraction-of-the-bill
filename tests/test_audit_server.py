"""The audit interface: that it finds each label where it is printed, points at what is
worth a second look without deciding anything, and never loses a verdict."""

from __future__ import annotations

import datetime as dt
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from items import TRUTH, make_item

from smallprint.data import audit
from smallprint.data.audit_server import (
    AuditApp,
    FieldEvidence,
    evidence,
    progress,
    save_verdict,
    server,
)
from smallprint.data.build import SplitItem
from smallprint.data.split import Split

PAGE = """[COVER PAGE]
FORM 10-K
For the fiscal year ended December 31 , 2025
Delaware | 12-3456789
(State of incorporation)

[INCOME STATEMENT]
(in thousands)
 | 2025 | 2024
Revenue | $1,200 | $1,100
Cost of revenue | (700) | (650)
Operating income | 300 | 280
Net income | 210 | 1,300
Basic earnings per share | $2.10 | $1.90
Diluted earnings per share | $2.05 | $1.85
Diluted weighted average shares | 100 | 98

[BALANCE SHEET]
(in thousands)
 | 2025 | 2024
Cash and cash equivalents | 800 | 700
Total assets | 4,000 | 3,900
Total liabilities | 1,500 | 1,450
Total stockholders equity | 2,500 | 2,450

[AUDITOR]
/s/ Deloitte & Touche LLP
We have served as the Company's auditor since 2010."""


def item(text: str = PAGE, **truth: object) -> SplitItem:
    base = make_item("audit-1", split=Split.TEST_POST_CUTOFF, text=text)
    return base.model_copy(
        update={"item": base.item.model_copy(update={"truth": TRUTH.model_copy(update=truth)})}
    )


def by_field(s: SplitItem) -> dict[str, FieldEvidence]:
    return {f.field: f for f in evidence(s)}


def test_every_label_is_shown_beside_the_row_it_was_read_from() -> None:
    found = by_field(item())
    revenue = found["revenue"]
    assert revenue.matches[0].line == "Revenue | $1,200 | $1,100"
    assert revenue.matches[0].column == 1
    assert revenue.printed_as == "1,200 at a scale of 1,000"
    for name, f in found.items():
        if name != "fiscal_period":
            assert f.matches, name
    assert not any(f.flags for f in found.values())


def test_a_date_printed_with_its_comma_set_apart_is_still_found() -> None:
    period = by_field(item())["period_end"]
    assert "December 31 , 2025" in period.matches[0].line


def test_a_value_found_only_in_the_prior_period_column_is_pointed_at() -> None:
    found = by_field(item(net_income=1_300_000.0))
    flags = found["net_income"].flags
    assert any("column 2" in f for f in flags)


def test_a_loss_printed_without_parentheses_is_pointed_at() -> None:
    found = by_field(item(operating_income=-300_000.0))
    assert any("parentheses" in f for f in found["operating_income"].flags)


def test_cost_of_revenue_in_parentheses_is_by_design_and_not_a_hint() -> None:
    assert by_field(item())["cost_of_revenue"].flags == ()


def test_a_label_not_printed_anywhere_in_its_section_is_pointed_at() -> None:
    found = by_field(item(total_assets=9_999_000.0))
    assert found["total_assets"].matches == ()
    assert any("Not found" in f for f in found["total_assets"].flags)


def test_a_zero_printed_as_a_dash_is_found() -> None:
    page = PAGE.replace("Operating income | 300 | 280", "Operating income | - | 280")
    found = by_field(item(page, operating_income=0.0))
    assert found["operating_income"].matches[0].column == 1


def test_a_null_says_why_it_is_null() -> None:
    found = by_field(item(total_liabilities=None))
    assert found["total_liabilities"].label is None
    assert any("reports no such fact" in f for f in found["total_liabilities"].flags)


def test_a_quarterly_report_labelled_as_a_year_is_pointed_at() -> None:
    s = item(fiscal_period="Q2")
    assert any("labelled Q2" in f for f in by_field(s)["fiscal_period"].flags)


# -- the sheet ----------------------------------------------------------------------------


@pytest.fixture
def sheet(tmp_path: Path) -> Path:
    chosen = [item()]
    chosen += [make_item(f"audit-{i}", split=Split.TEST_POST_CUTOFF) for i in range(2, 4)]
    return audit.write_audit(tmp_path, chosen)


def test_a_verdict_is_written_and_the_others_are_left_alone(sheet: Path) -> None:
    save_verdict(sheet, 2, "wrong", ["revenue", "net_income"], " digit dropped ")
    save_verdict(sheet, 1, "ok", [], "")
    rows = {int(r["n"]): r for r in audit.read_sheet(sheet)}
    assert rows[2]["wrong_fields"] == "revenue net_income"
    assert rows[2]["note"] == "digit dropped"
    assert rows[1]["verdict"] == "ok" and rows[3]["verdict"] == ""
    report = audit.report(sheet)  # the report reads what the server wrote
    assert (report.audited, report.pending) == (2, 1)


@pytest.mark.parametrize(
    ("verdict", "fields", "message"),
    [
        ("maybe", [], "ok or wrong"),
        ("wrong", [], "name the wrong fields"),
        ("wrong", ["revenu"], "not schema fields"),
        ("ok", ["revenue"], "cannot name wrong fields"),
    ],
)
def test_a_verdict_the_report_would_refuse_is_refused_when_given(
    sheet: Path, verdict: str, fields: list[str], message: str
) -> None:
    before = sheet.read_bytes()
    with pytest.raises(ValueError, match=message):
        save_verdict(sheet, 1, verdict, fields, "")
    assert sheet.read_bytes() == before


def test_progress_reports_the_error_rate_with_its_interval(sheet: Path) -> None:
    save_verdict(sheet, 1, "wrong", ["revenue"], "")
    state = progress(sheet)
    assert (state["done"], state["wrong"], state["first_pending"]) == (1, 1, 2)
    assert state["rate"]["low"] < 1.0 == state["rate"]["high"]


# -- the server ----------------------------------------------------------------------------


@pytest.fixture
def running(sheet: Path) -> Iterator[str]:
    items = [item()] + [make_item(f"audit-{i}", split=Split.TEST_POST_CUTOFF) for i in range(2, 4)]
    httpd = server(AuditApp(sheet, items), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def call(url: str, body: dict[str, object] | None = None) -> dict[str, object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())  # type: ignore[no-any-return]


def test_the_page_the_state_an_item_and_a_verdict_over_http(running: str, sheet: Path) -> None:
    with urllib.request.urlopen(running + "/", timeout=10) as response:
        assert b"Label audit" in response.read()
    assert call(running + "/api/state")["first_pending"] == 1
    detail = call(running + "/api/item/1")
    assert detail["item_id"] == "audit-1"
    assert len(detail["fields"]) == 15  # type: ignore[arg-type]
    after = call(running + "/api/item/1", {"verdict": "ok", "wrong_fields": [], "note": ""})
    assert after["done"] == 1 and after["first_pending"] == 2
    assert audit.read_sheet(sheet)[0]["verdict"] == "ok"


def test_a_bad_verdict_over_http_is_a_400_and_writes_nothing(running: str, sheet: Path) -> None:
    before = sheet.read_bytes()
    with pytest.raises(urllib.error.HTTPError) as refused:
        call(running + "/api/item/1", {"verdict": "wrong", "wrong_fields": []})
    assert refused.value.code == 400
    assert sheet.read_bytes() == before


def test_the_sample_must_come_from_the_build_it_is_checked_against(sheet: Path) -> None:
    with pytest.raises(ValueError, match="not in this build"):
        AuditApp(sheet, [item()])


def test_dates_render_in_the_item_detail(running: str) -> None:
    detail = call(running + "/api/item/1")
    assert detail["period_end"] == dt.date(2025, 12, 31).isoformat()
