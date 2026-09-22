"""The results table. It is generated, never typed, so what it leaves out matters as much
as what it contains."""

from __future__ import annotations

from pathlib import Path

from test_baseline import RIGHT, WRONG, StubGateway, items, reply, zero_shot

from smallprint import baseline, report


def make_run(out: Path, answers: dict[int, str], *, model: str, cost: float, n: int = 4) -> None:
    gateway = StubGateway(lambda i: reply(answers.get(i, RIGHT), cost=cost))
    baseline.run(
        gateway,
        model=model,
        items=items(n),
        prompt=zero_shot(),
        out_dir=out,
        build_dir=Path("data/build/full"),
        run_id=f"run-{model}",
    )


def test_the_table_is_cheapest_first_and_every_delta_is_paired(tmp_path: Path) -> None:
    make_run(tmp_path / "cheap", {1: WRONG}, model="openai/cheap", cost=0.001)
    make_run(tmp_path / "dear", {}, model="openai/dear", cost=0.010)

    rows, skipped = report.load(report.run_dirs(tmp_path), items())
    assert [r.summary.model for r in rows] == ["openai/cheap", "openai/dear"]
    assert skipped == []

    table = report.frontier_table(rows)
    assert "US$1.00" in table and "US$10.00" in table
    cheap, dear = table.splitlines()[2], table.splitlines()[3]
    assert "baseline" in cheap  # the anchor is the cheapest run, not the best one
    assert "+3.3% (+0.0% to +10.0%)" in dear  # 2 of 60 fields, paired over the same filings


def test_a_run_over_part_of_the_split_is_kept_out_of_the_published_table(tmp_path: Path) -> None:
    """A smoke run lives beside the real ones and is the same shape on disk. A twenty-item
    row in a published table next to seven-hundred-item rows is a mistake waiting to be
    quoted, and it cannot be paired with them either."""
    make_run(tmp_path / "whole", {}, model="openai/whole", cost=0.001)
    make_run(tmp_path / "smoke", {}, model="openai/smoke", cost=0.001, n=2)

    rows, skipped = report.load(report.run_dirs(tmp_path), items())
    assert [r.summary.model for r in rows] == ["openai/whole"]
    assert [d.name for d in skipped] == ["smoke"]

    rows_all, _ = report.load(report.run_dirs(tmp_path), items(), complete_only=False)
    assert len(rows_all) == 2


def test_a_run_that_answered_items_a_rebuild_dropped_still_covers_the_split(
    tmp_path: Path,
) -> None:
    make_run(tmp_path / "before", {}, model="openai/before", cost=0.001)
    rows, skipped = report.load(report.run_dirs(tmp_path), items(3))
    assert skipped == []
    assert (rows[0].summary.graded, rows[0].summary.set_aside) == (3, 1)


def test_the_field_table_names_every_field_worst_first(tmp_path: Path) -> None:
    make_run(tmp_path / "one", {0: WRONG, 1: WRONG}, model="openai/one", cost=0.001)
    rows, _ = report.load(report.run_dirs(tmp_path), items())
    table = report.field_table(rows[0])
    assert table.count("| `") == 15
    assert table.splitlines()[2].startswith("| `revenue`")  # wrong twice, so the worst
    assert "hallucinated" not in table  # WRONG misses by value and by absence, not invention
    assert "| `period_end` | 100.0%" in table
