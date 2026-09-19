"""The datasheet, the checksums and the hand audit, on a build made through the mock EDGAR."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import httpx
import pytest
from test_build import CUTOFF, MockEdgar

from smallprint.data import audit
from smallprint.data.build import SplitItem, build, write_build
from smallprint.data.datasheet import read_items, verify, write_datasheet
from smallprint.data.edgar import EdgarClient
from smallprint.data.pair import Item
from smallprint.data.split import Split
from smallprint.grade import GradeContext
from smallprint.schema import Extraction


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    with EdgarClient(
        tmp_path / "raw", contact="someone@example.com", transport=httpx.MockTransport(MockEdgar())
    ) as edgar:
        items, dropped, report = build(edgar, first="2026Q4", last="2027Q1", cutoff=CUTOFF)
    out = tmp_path / "build"
    write_build(out, items, dropped, report)
    return out


def test_the_datasheet_is_written_from_the_build_and_names_what_is_missing(
    build_dir: Path,
) -> None:
    sheet, _ = write_datasheet(build_dir)
    text = sheet.read_text(encoding="utf-8")
    assert "**1 items**" in text
    assert "`bank`" in text and "`not_fetched`" in text
    assert "2026-12-31" in text
    assert "decided before publication" in text  # the licence is a gate, not a guess
    assert "--companies" not in text  # no limit was set, and the rebuild command says so


def test_checksums_catch_a_file_changed_after_the_build(build_dir: Path) -> None:
    write_datasheet(build_dir)
    assert verify(build_dir) == []
    with (build_dir / "items.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert verify(build_dir) == ["items.jsonl"]


def item(item_id: str, split: Split, form: str) -> SplitItem:
    return SplitItem(
        split=split,
        item=Item(
            item_id=item_id,
            cik=1,
            form=form,
            filed=dt.date(2026, 1, 1),
            period_end=dt.date(2025, 12, 31),
            fiscal_period="FY" if form == "10-K" else "Q3",
            text="[INCOME STATEMENT]\nRevenue | 1",
            truth=Extraction(revenue=1000.0),
            context=GradeContext(scale=1000.0),
        ),
    )


def population() -> list[SplitItem]:
    out = [item(f"train-q-{i}", Split.TRAIN, "10-Q") for i in range(300)]
    out += [item(f"train-k-{i}", Split.TRAIN, "10-K") for i in range(100)]
    out += [item(f"test-k-{i}", Split.TEST_POST_CUTOFF, "10-K") for i in range(3)]
    return out


def test_the_sample_is_the_same_every_time_and_reaches_every_stratum() -> None:
    first = audit.sample(population(), 20, seed=1)
    second = audit.sample(population(), 20, seed=1)
    assert [s.item.item_id for s in first] == [s.item.item_id for s in second]
    assert len(first) == 20
    assert sum(1 for s in first if s.split is Split.TEST_POST_CUTOFF) == 3


def test_a_sample_larger_than_the_corpus_is_the_whole_corpus() -> None:
    assert len(audit.sample(population()[:5], 200)) == 5


@pytest.mark.parametrize(
    ("count", "n", "low", "high"),
    [
        (0, 200, 0.0, 0.0188),  # no errors seen is still not "no errors"
        (10, 100, 0.0552, 0.1744),
        (50, 100, 0.4038, 0.5962),
    ],
)
def test_the_wilson_interval_matches_the_textbook_values(
    count: int, n: int, low: float, high: float
) -> None:
    result = audit.wilson(count, n)
    assert result.low == pytest.approx(low, abs=5e-4)
    assert result.high == pytest.approx(high, abs=5e-4)


def fill(sheet: Path, verdicts: dict[int, tuple[str, str]]) -> None:
    rows = audit.read_sheet(sheet)
    for row in rows:
        verdict, fields = verdicts.get(int(row["n"]), ("", ""))
        row["verdict"], row["wrong_fields"] = verdict, fields
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_the_audit_writes_pages_and_reports_the_error_rate(tmp_path: Path) -> None:
    sheet = audit.write_audit(tmp_path, audit.sample(population(), 4, seed=3))
    pages = sorted((tmp_path / "items").iterdir())
    assert len(pages) == 4
    assert "| revenue | 1000.0 | income |" in pages[0].read_text(encoding="utf-8")

    fill(sheet, {1: ("ok", ""), 2: ("wrong", "revenue net_income"), 3: ("ok", "")})
    result = audit.report(sheet)
    assert (result.audited, result.pending) == (3, 1)
    assert result.items_wrong.count == 1
    assert result.wrong_by_field == {"revenue": 1, "net_income": 1}


def test_a_wrong_verdict_must_name_real_fields(tmp_path: Path) -> None:
    sheet = audit.write_audit(tmp_path, audit.sample(population(), 2, seed=3))
    fill(sheet, {1: ("wrong", "")})
    with pytest.raises(ValueError, match="name the wrong fields"):
        audit.report(sheet)
    fill(sheet, {1: ("wrong", "revenu")})
    with pytest.raises(ValueError, match="not schema fields"):
        audit.report(sheet)


def test_verdicts_already_given_are_never_overwritten(tmp_path: Path) -> None:
    sheet = audit.write_audit(tmp_path, audit.sample(population(), 2, seed=3))
    fill(sheet, {1: ("ok", "")})
    with pytest.raises(FileExistsError):
        audit.write_audit(tmp_path, audit.sample(population(), 2, seed=3))


def test_items_read_back_from_a_build_are_what_was_written(build_dir: Path) -> None:
    assert [s.item.item_id for s in read_items(build_dir)] == ["0001234567-26-000042"]
