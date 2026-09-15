"""The EDGAR client's fair-access behaviour, against a mock transport.

Deliberately never against sec.gov. A test suite that hammers a public regulator to prove
it is polite to that regulator has missed the point, and CI would run it on every push.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import pytest

from smallprint.data.edgar import (
    ContactNotDeclared,
    EdgarClient,
    FairAccessViolation,
    IndexEntry,
    declared_contact,
    parse_form_index,
)

CONTACT = "someone@example.com"

FORM_IDX = b"""Description:           Master Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 31, 2027

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------
10-K        ACME HOLDINGS, INC. /DE/                                      320193      2027-02-14  edgar/data/320193/0000320193-27-000006.txt
10-K        BETA CORP                                                     789019      2027-03-02  edgar/data/789019/0000789019-27-000011.txt
10-Q        ACME HOLDINGS, INC. /DE/                                      320193      2027-01-20  edgar/data/320193/0000320193-27-000002.txt
8-K         GAMMA INDUSTRIES LTD                                          1018724     2027-01-05  edgar/data/1018724/0001018724-27-000004.txt
"""


def responder(
    body: bytes = b"ok", status: int = 200, statuses: list[int] | None = None
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A transport that records its requests and optionally walks a list of statuses."""
    seen: list[httpx.Request] = []
    queue = list(statuses) if statuses is not None else None

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        code = queue.pop(0) if queue else status
        return httpx.Response(code, content=body if code == 200 else b"")

    return httpx.MockTransport(handle), seen


def test_a_missing_contact_is_refused_with_an_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    with pytest.raises(ContactNotDeclared, match="no default"):
        declared_contact()


def test_a_contact_that_is_not_an_address_is_refused() -> None:
    with pytest.raises(ContactNotDeclared, match="reaches a person"):
        declared_contact("smallprint research project")


def test_the_environment_supplies_the_contact_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMALLPRINT_EDGAR_CONTACT", CONTACT)
    assert declared_contact() == CONTACT


def test_the_client_refuses_to_be_built_without_a_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It fails before the first fetch, not part way through a corpus of thirty thousand."""
    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    with pytest.raises(ContactNotDeclared):
        EdgarClient(tmp_path)


def test_the_contact_reaches_the_user_agent(tmp_path: Path) -> None:
    transport, seen = responder()
    with EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar:
        edgar.get("https://www.sec.gov/one")
    assert CONTACT in seen[0].headers["user-agent"]
    assert seen[0].headers["user-agent"].startswith("smallprint/")


def test_a_rate_above_the_published_limit_is_refused(tmp_path: Path) -> None:
    transport, _ = responder()
    with pytest.raises(FairAccessViolation, match="ten per second"):
        EdgarClient(tmp_path, contact=CONTACT, rate_per_second=25, transport=transport)


def test_requests_are_paced_by_the_client_not_by_the_caller(tmp_path: Path) -> None:
    transport, seen = responder()
    with EdgarClient(tmp_path, contact=CONTACT, rate_per_second=10.0, transport=transport) as edgar:
        started = time.monotonic()
        for i in range(3):
            edgar.get(f"https://www.sec.gov/{i}")
        elapsed = time.monotonic() - started
    assert len(seen) == 3
    # Three requests at ten a second means at least two full intervals of waiting.
    assert elapsed >= 0.2


def test_a_rate_limited_response_is_backed_off_and_retried(tmp_path: Path) -> None:
    transport, seen = responder(statuses=[429, 200])
    with EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar:
        assert edgar.get("https://www.sec.gov/one") == b"ok"
    assert len(seen) == 2


def test_a_status_that_will_not_improve_is_not_retried(tmp_path: Path) -> None:
    """Retrying a 404 is not persistence, it is four requests for a document that is gone."""
    transport, seen = responder(status=404)
    with (
        EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar,
        pytest.raises(httpx.HTTPStatusError),
    ):
        edgar.get("https://www.sec.gov/missing")
    assert len(seen) == 1


def test_a_second_fetch_of_the_same_url_never_leaves_the_machine(tmp_path: Path) -> None:
    transport, seen = responder(body=b"body")
    with EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar:
        first = edgar.get("https://www.sec.gov/one")
        second = edgar.get("https://www.sec.gov/one")
    assert first == second == b"body"
    assert len(seen) == 1


def test_the_cache_records_what_was_fetched_and_when(tmp_path: Path) -> None:
    transport, _ = responder(body=b"body")
    with EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar:
        edgar.get("https://www.sec.gov/one")
        records = list(edgar.manifest())
    assert len(records) == 1
    assert records[0].url == "https://www.sec.gov/one"
    assert records[0].bytes == 4
    # sha256 of b"body"
    assert records[0].sha256.startswith("230d8358dc8e8890b4c58deeb62912ee")


def test_an_offline_client_reads_the_cache_and_needs_no_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset rebuild is reproducible from the cache without a network or an address."""
    transport, _ = responder(body=b"body")
    with EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar:
        edgar.get("https://www.sec.gov/one")

    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    with EdgarClient(tmp_path, offline=True) as offline:
        assert offline.get("https://www.sec.gov/one") == b"body"
        with pytest.raises(FairAccessViolation, match="no cached copy"):
            offline.get("https://www.sec.gov/two")


def test_the_quarter_is_validated_before_a_request_is_made(tmp_path: Path) -> None:
    transport, seen = responder()
    with (
        EdgarClient(tmp_path, contact=CONTACT, transport=transport) as edgar,
        pytest.raises(ValueError, match="quarter must be"),
    ):
        edgar.quarterly_index(2027, 5)
    assert seen == []


def test_the_form_index_survives_company_names_with_commas_and_slashes() -> None:
    """Splitting on whitespace loses these, and they are not rare."""
    entries = list(parse_form_index(FORM_IDX))
    assert len(entries) == 4
    assert entries[0].company == "ACME HOLDINGS, INC. /DE/"
    assert entries[0].cik == 320193
    assert entries[0].form == "10-K"
    assert entries[0].filed.isoformat() == "2027-02-14"
    assert entries[0].accession == "0000320193-27-000006"


def test_the_form_index_can_be_filtered_to_the_forms_this_project_uses() -> None:
    entries = list(parse_form_index(FORM_IDX, forms=frozenset({"10-K", "10-Q"})))
    assert [e.form for e in entries] == ["10-K", "10-K", "10-Q"]


def test_an_index_whose_format_has_changed_fails_loudly() -> None:
    with pytest.raises(ValueError, match="format has changed"):
        list(parse_form_index(b"Form Type  CIK\n10-K  320193\n"))


def test_an_index_entry_exposes_its_accession_number() -> None:
    entry = IndexEntry(
        form="10-K",
        cik=1,
        company="X",
        filed=dt.date(2027, 1, 1),
        path="edgar/data/1/0000000001-27-000001.txt",
    )
    assert entry.accession == "0000000001-27-000001"
