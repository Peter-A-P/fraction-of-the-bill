"""The command line says what is true, including when the answer is that nothing exists yet."""

from __future__ import annotations

from pathlib import Path

import pytest
from filings import document
from typer.testing import CliRunner

from smallprint import __version__
from smallprint.cli import app
from smallprint.data.edgar import EdgarClient

runner = CliRunner()


def test_version_is_the_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_the_schema_command_lists_every_field_and_its_truth_concept() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert "15 fields" in result.output
    assert "us-gaap:Assets" in result.output
    assert "dei:DocumentPeriodEndDate" in result.output


def test_the_schema_command_can_print_the_prompt_form() -> None:
    result = runner.invoke(app, ["schema", "--prompt"])
    assert result.exit_code == 0
    assert "period_end (string, YYYY-MM-DD, or null)" in result.output


def test_check_access_refuses_and_explains_when_no_contact_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    result = runner.invoke(app, ["data", "check-access"])
    assert result.exit_code == 1
    assert "Not configured" in result.output


def test_check_access_confirms_without_fetching_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALLPRINT_EDGAR_CONTACT", "someone@example.com")
    result = runner.invoke(app, ["data", "check-access"])
    assert result.exit_code == 0
    assert "someone@example.com" in result.output
    assert "8 per second" in result.output


def test_the_cache_command_says_so_when_nothing_has_been_fetched(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "cache", "--cache-dir", str(tmp_path / "absent")])
    assert result.exit_code == 0
    assert "Nothing has been fetched" in result.output


def test_the_cache_command_lists_what_was_fetched_with_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"body"))
    with EdgarClient(tmp_path, contact="someone@example.com", transport=transport) as edgar:
        edgar.get("https://www.sec.gov/one")

    monkeypatch.delenv("SMALLPRINT_EDGAR_CONTACT", raising=False)
    result = runner.invoke(app, ["data", "cache", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "https://www.sec.gov/one" in result.output
    assert "1 documents" in result.output


def test_locate_shows_the_input_and_the_scales_it_read(tmp_path: Path) -> None:
    path = tmp_path / "filing.htm"
    path.write_bytes(document())
    result = runner.invoke(app, ["data", "locate", str(path)])
    assert result.exit_code == 0
    assert "[INCOME STATEMENT]" in result.output
    assert "income statement: 10 rows, scale 1,000" in result.output


def test_locate_fails_when_a_statement_is_not_found(tmp_path: Path) -> None:
    path = tmp_path / "empty.htm"
    path.write_bytes(b"<html><body><p>Nothing here.</p></body></html>")
    result = runner.invoke(app, ["data", "locate", str(path)])
    assert result.exit_code == 1
    assert "income statement: not found" in result.output


def test_no_command_promises_work_that_does_not_exist() -> None:
    """train, quantise, serve and bench are absent rather than stubbed. Help must not offer them.

    breakeven is present: it computes from inputs the reader supplies. So is baseline,
    which runs a model over held-out filings through the gateway, and report, which writes
    the results table from the runs that exist.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for absent in ("train", "quantise", "serve", "bench "):
        assert absent not in result.output
    for present in ("data", "baseline", "breakeven", "schema", "report"):
        assert present in result.output
