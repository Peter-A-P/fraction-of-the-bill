"""The corpus build end to end, against a mock EDGAR.

Two companies across two quarters. One files a 10-Q, a 10-K, an amended 10-K and an 8-K,
with its older filings on a second page of its filing history. The other has no XBRL facts
at all. The build has to select the right forms, page back for the old filing, pair what it
can, name what it cannot, split the rest, and do all of it again offline from the cache.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import filings
import httpx
import pytest
from typer.testing import CliRunner

from smallprint.cli import app
from smallprint.data.build import (
    BuildReport,
    SplitItem,
    build,
    quarter_range,
    size_band,
    write_build,
)
from smallprint.data.edgar import EdgarClient
from smallprint.data.pair import Dropped
from smallprint.data.split import Split
from smallprint.data.xbrl import Fact

NO_FACTS_CIK = 7654321
NO_FACTS_ACCESSION = "0007654321-26-000003"
BANK_CIK = 5550001
BANK_ACCESSION = "0005550001-26-000009"

HEADER = (
    "Description:           Master Index of EDGAR Dissemination Feed by Form Type\n"
    "\n"
    "Form Type   Company Name                                                  CIK         "
    "Date Filed  File Name\n" + "-" * 120 + "\n"
)


def index_line(form: str, company: str, cik: int, filed: str, accession: str) -> str:
    return f"{form:<12}{company:<62}{cik:<12}{filed:<12}edgar/data/{cik}/{accession}.txt\n"


FORM_INDEX = {
    (2026, 4): HEADER
    + index_line(
        "10-Q", "EXAMPLE WIDGETS, INC.", filings.CIK, "2026-11-04", filings.QUARTERLY_ACCESSION
    )
    + index_line("8-K", "EXAMPLE WIDGETS, INC.", filings.CIK, "2026-11-05", "0001234567-26-000050")
    + index_line("10-Q", "NO FACTS CORP", NO_FACTS_CIK, "2026-11-10", NO_FACTS_ACCESSION)
    + index_line("10-Q", "FIRST EXAMPLE BANCORP", BANK_CIK, "2026-11-12", BANK_ACCESSION),
    (2027, 1): HEADER
    + index_line(
        "10-K", "EXAMPLE WIDGETS, INC.", filings.CIK, "2027-02-25", filings.ANNUAL_ACCESSION
    )
    + index_line(
        "10-K/A", "EXAMPLE WIDGETS, INC.", filings.CIK, "2027-03-20", "0001234567-27-000011"
    ),
}


def company_facts_json(facts: list[Fact]) -> dict[str, object]:
    """The shape of the companyfacts API, which carries the numbers but not the cover page."""
    tree: dict[str, dict[str, dict[str, dict[str, list[dict[str, object]]]]]] = {}
    for fact in facts:
        if fact.concept.startswith("dei:"):
            continue
        taxonomy, name = fact.concept.split(":")
        entry: dict[str, object] = {
            "end": fact.end.isoformat(),
            "val": fact.value,
            "accn": fact.accession,
            "form": fact.form,
            "filed": fact.filed.isoformat(),
        }
        if fact.start is not None:
            entry["start"] = fact.start.isoformat()
        units = tree.setdefault(taxonomy, {}).setdefault(name, {"units": {}})["units"]
        units.setdefault(fact.unit, []).append(entry)
    return {"cik": filings.CIK, "facts": tree}


def routes() -> dict[str, bytes]:
    archive = "https://www.sec.gov/Archives/edgar"
    data = "https://data.sec.gov"
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": [filings.ANNUAL_ACCESSION, "0001234567-27-000011"],
                "primaryDocument": ["exw-20261231.htm", "exw-10ka.htm"],
            },
            "files": [
                {
                    "name": "CIK0001234567-submissions-001.json",
                    "filingFrom": "2019-01-01",
                    "filingTo": "2026-12-31",
                },
                {
                    "name": "CIK0001234567-submissions-002.json",
                    "filingFrom": "2001-01-01",
                    "filingTo": "2018-12-31",
                },
            ],
        }
    }
    older = {
        "accessionNumber": [filings.QUARTERLY_ACCESSION],
        "primaryDocument": ["exw-20260930.htm"],
    }
    facts = filings.facts() + filings.facts(annual=True)
    out = {
        f"{archive}/full-index/{y}/QTR{q}/form.idx": body.encode("latin-1")
        for (y, q), body in FORM_INDEX.items()
    }
    out |= {
        f"{data}/submissions/CIK0001234567.json": json.dumps(submissions).encode(),
        f"{data}/submissions/CIK0001234567-submissions-001.json": json.dumps(older).encode(),
        f"{data}/submissions/CIK{BANK_CIK:010d}.json": json.dumps(
            {"sic": "6022", "sicDescription": "State commercial banks", "filings": {}}
        ).encode(),
        f"{data}/api/xbrl/companyfacts/CIK0001234567.json": json.dumps(
            company_facts_json(facts)
        ).encode(),
        f"{archive}/data/{filings.CIK}/{filings.QUARTERLY_ACCESSION.replace('-', '')}"
        "/exw-20260930.htm": filings.document(),
        f"{archive}/data/{filings.CIK}/{filings.ANNUAL_ACCESSION.replace('-', '')}"
        "/exw-20261231.htm": filings.document(annual=True),
    }
    return out


class MockEdgar:
    def __init__(self) -> None:
        self.routes = routes()
        self.requested: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        body = self.routes.get(url)
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)


CUTOFF = dt.date(2026, 12, 31)

Built = tuple[list[SplitItem], list[Dropped], BuildReport, MockEdgar, Path]


@pytest.fixture
def built(tmp_path: Path) -> Built:
    mock = MockEdgar()
    cache = tmp_path / "raw"
    with EdgarClient(
        cache, contact="someone@example.com", transport=httpx.MockTransport(mock)
    ) as edgar:
        items, dropped, report = build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF)
    return items, dropped, report, mock, cache


def test_only_ten_ks_and_ten_qs_are_selected(
    built: Built,
) -> None:
    _, _, report, _, _ = built
    assert report.filings_selected == 4  # not the 8-K, not the 10-K/A
    assert report.companies_selected == 3


def test_every_selected_filing_is_kept_or_dropped_by_name(
    built: Built,
) -> None:
    _, _, report, _, _ = built
    assert report.pairing == {"kept": 2, "bank": 1, "not_fetched": 1}


def test_a_bank_is_dropped_by_name_for_the_price_of_one_request(built: Built) -> None:
    _, _, _, mock, _ = built
    assert [u for u in mock.requested if str(BANK_CIK) in u] == [
        f"https://data.sec.gov/submissions/CIK{BANK_CIK:010d}.json"
    ]


def test_the_history_is_paged_back_only_as_far_as_the_range(
    built: Built,
) -> None:
    _, _, _, mock, _ = built
    assert any(u.endswith("submissions-001.json") for u in mock.requested)
    assert not any(u.endswith("submissions-002.json") for u in mock.requested)


def test_a_company_without_facts_costs_no_document_fetches(
    built: Built,
) -> None:
    _, _, _, mock, _ = built
    assert not any(f"/data/{NO_FACTS_CIK}/" in u for u in mock.requested)


def test_the_split_is_applied_and_its_drops_are_counted(
    built: Built,
) -> None:
    """One company in one stratum goes to training; its filing after the cutoff is dropped."""
    items, _, report, _, _ = built
    assert [(s.split, s.item.item_id) for s in items] == [
        (Split.TRAIN, filings.QUARTERLY_ACCESSION)
    ]
    assert report.split.dropped_after_cutoff == 1


def test_the_build_is_written_and_reproduces_offline_from_the_cache(
    built: Built, tmp_path: Path
) -> None:
    items, dropped, report, _, cache = built
    out = tmp_path / "build"
    write_build(out, items, dropped, report)
    written = (out / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert [SplitItem.model_validate_json(line) for line in written] == items
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["pairing"]["kept"] == 2

    with EdgarClient(cache, offline=True) as edgar:
        again, _, again_report = build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF)
    assert again == items
    assert again_report.pairing == report.pairing


def test_a_company_limit_selects_the_same_companies_every_time(tmp_path: Path) -> None:
    chosen = []
    for run in range(2):
        mock = MockEdgar()
        with EdgarClient(
            tmp_path / f"raw{run}",
            contact="someone@example.com",
            transport=httpx.MockTransport(mock),
        ) as edgar:
            _, _, report = build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF, companies=1)
        assert report.companies_selected == 1
        chosen.append(report.pairing)
    assert chosen[0] == chosen[1]


def test_quarter_ranges_are_inclusive_and_validated() -> None:
    assert quarter_range("2026Q3", "2027Q2") == [(2026, 3), (2026, 4), (2027, 1), (2027, 2)]
    with pytest.raises(ValueError, match="2022Q1"):
        quarter_range("2022-Q1", "2022Q4")
    with pytest.raises(ValueError, match="after"):
        quarter_range("2027Q1", "2026Q4")


@pytest.mark.parametrize(
    ("assets", "band"),
    [(5e7, "under_100m"), (1e8, "100m_to_1b"), (2.5e9, "1b_to_10b"), (3e11, "over_10b")],
)
def test_size_bands(assets: float, band: str) -> None:
    assert size_band(assets) == band


def test_the_build_command_needs_a_cutoff_and_runs_offline_from_a_warm_cache(
    built: Built,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, cache = built
    runner = CliRunner()
    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    base = ["data", "build", "--first", "2026Q4", "--last", "2027Q1", "--cache-dir", str(cache)]

    missing = runner.invoke(app, [*base, "--offline"])
    assert missing.exit_code != 0

    out = tmp_path / "cli-build"
    result = runner.invoke(
        app, [*base, "--offline", "--cutoff", CUTOFF.isoformat(), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "kept: 2" in result.output
    assert (out / "items.jsonl").exists()


def test_an_unanticipated_error_in_one_filing_is_a_named_drop_not_a_stopped_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smallprint.data.build as build_module

    def explode(*args: object, **kwargs: object) -> object:
        raise KeyError("something no rule anticipated")

    monkeypatch.setattr(build_module, "pair_filing", explode)
    mock = MockEdgar()
    with EdgarClient(
        tmp_path, contact="someone@example.com", transport=httpx.MockTransport(mock)
    ) as edgar:
        _, _, report = build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF)
    assert report.pairing["pairing_error:KeyError"] == 2


def test_a_parallel_offline_rebuild_writes_exactly_what_a_sequential_one_does(
    built: Built,
) -> None:
    items, dropped, report, _, cache = built
    with EdgarClient(cache, offline=True) as edgar:
        again, again_dropped, again_report = build(
            edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF, workers=2
        )
    assert again == items
    assert again_dropped == dropped
    assert again_report.pairing == report.pairing


def test_parallel_workers_are_refused_for_a_live_fetch(tmp_path: Path) -> None:
    """Several clients each pacing themselves would together exceed the SEC's limit."""
    from smallprint.data.edgar import FairAccessViolation

    with (
        EdgarClient(
            tmp_path, contact="someone@example.com", transport=httpx.MockTransport(MockEdgar())
        ) as edgar,
        pytest.raises(FairAccessViolation, match="offline rebuilds only"),
    ):
        build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF, workers=4)
