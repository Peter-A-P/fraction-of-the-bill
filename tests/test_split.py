"""The split test: no filer is used for two purposes, and the cutoff means what it says.

This is the test the whole dataset rests on. If it passes for the wrong reason the headline
number is memorisation, and nothing downstream, not the intervals, not the non-inferiority
run, would show it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from smallprint.data.split import (
    POOL_OF,
    FilingRef,
    Pool,
    Split,
    SplitLeakage,
    assign_pools,
    assign_splits,
    check_no_leakage,
)

CUTOFF = dt.date(2026, 10, 1)


def corpus(n_companies: int = 200, per_company: int = 8, strata: int = 1) -> list[FilingRef]:
    """A corpus shaped like the real one: every filer files repeatedly, either side of the cutoff."""
    filings = []
    for c in range(n_companies):
        cik = 1_000_000 + c
        for q in range(per_company):
            filed = dt.date(2025, 1, 15) + dt.timedelta(days=91 * q)
            filings.append(
                FilingRef(
                    item_id=f"{cik}-{q}",
                    cik=cik,
                    form="10-Q" if q % 4 else "10-K",
                    filed=filed,
                    stratum=f"decile-{c % strata}",
                )
            )
    return filings


def test_no_filer_is_used_for_both_training_and_testing() -> None:
    filings = corpus()
    assignment, _ = assign_splits(filings, cutoff=CUTOFF)
    pools_seen: dict[int, set[Pool]] = {}
    for filing in filings:
        if filing.item_id in assignment:
            pools_seen.setdefault(filing.cik, set()).add(POOL_OF[assignment[filing.item_id]])
    assert all(len(pools) == 1 for pools in pools_seen.values())


def test_the_two_test_sets_share_their_filers_on_purpose() -> None:
    """The contamination gap is a within-filer comparison, or it is confounded by company mix."""
    filings = corpus()
    assignment, _ = assign_splits(filings, cutoff=CUTOFF)
    by_id = {f.item_id: f for f in filings}
    pre = {by_id[i].cik for i, s in assignment.items() if s is Split.TEST_PRE_CUTOFF}
    post = {by_id[i].cik for i, s in assignment.items() if s is Split.TEST_POST_CUTOFF}
    assert pre and post
    assert pre == post


def test_nothing_in_the_training_set_is_dated_after_the_cutoff() -> None:
    filings = corpus()
    assignment, report = assign_splits(filings, cutoff=CUTOFF)
    by_id = {f.item_id: f for f in filings}
    trained_on = [by_id[i] for i, s in assignment.items() if s in (Split.TRAIN, Split.VALIDATION)]
    assert trained_on
    assert all(f.filed <= CUTOFF for f in trained_on)
    assert report.dropped_after_cutoff > 0


def test_the_post_cutoff_set_holds_only_filings_after_the_cutoff() -> None:
    filings = corpus()
    assignment, _ = assign_splits(filings, cutoff=CUTOFF)
    by_id = {f.item_id: f for f in filings}
    for item_id, split in assignment.items():
        if split is Split.TEST_POST_CUTOFF:
            assert by_id[item_id].filed > CUTOFF
        if split is Split.TEST_PRE_CUTOFF:
            assert by_id[item_id].filed <= CUTOFF


def test_every_filing_is_either_assigned_or_counted_as_dropped() -> None:
    """Nothing disappears silently between the corpus and the splits."""
    filings = corpus()
    assignment, report = assign_splits(filings, cutoff=CUTOFF)
    assert len(assignment) + report.dropped_after_cutoff == len(filings)
    assert sum(report.counts.values()) == len(assignment)


def test_the_assignment_is_the_same_on_every_machine_and_every_rebuild() -> None:
    """A held-out set that moves between runs was never held out from the earlier one."""
    first, _ = assign_splits(corpus(), cutoff=CUTOFF)
    second, _ = assign_splits(list(reversed(corpus())), cutoff=CUTOFF)
    assert first == second


def test_a_different_seed_gives_a_different_partition() -> None:
    a = assign_pools(corpus(), seed=1)
    b = assign_pools(corpus(), seed=2)
    assert a != b


def test_pools_come_out_close_to_the_fractions_asked_for() -> None:
    pools = assign_pools(corpus(n_companies=500), test_fraction=0.2, validation_fraction=0.1)
    counts = {p: sum(1 for v in pools.values() if v is p) for p in Pool}
    assert counts[Pool.TEST] == 100
    assert counts[Pool.VALIDATION] == 50
    assert counts[Pool.TRAIN] == 350


def test_every_stratum_contributes_to_the_test_pool() -> None:
    """A stratum that only ever trains is one the held-out number cannot speak for."""
    filings = corpus(n_companies=60, strata=10)
    pools = assign_pools(filings, test_fraction=0.2)
    by_stratum: dict[str, set[Pool]] = {}
    for filing in filings:
        by_stratum.setdefault(filing.stratum, set()).add(pools[filing.cik])
    assert len(by_stratum) == 10
    assert all(Pool.TEST in pools_seen for pools_seen in by_stratum.values())


def test_a_tiny_stratum_still_yields_a_test_company() -> None:
    filings = [
        FilingRef(
            item_id=f"{cik}-0", cik=cik, form="10-K", filed=dt.date(2025, 3, 1), stratum="rare"
        )
        for cik in (1, 2)
    ]
    pools = assign_pools(filings, test_fraction=0.2)
    assert Pool.TEST in pools.values()


def test_fractions_that_leave_no_training_data_are_refused() -> None:
    with pytest.raises(ValueError, match="no training data"):
        assign_pools(corpus(), test_fraction=0.7, validation_fraction=0.4)


def test_a_hand_edited_split_that_reuses_a_filer_is_caught() -> None:
    """The failure this exists for: a few held-out filings put back into training by hand."""
    filings = corpus(n_companies=10, per_company=2)
    assignment, _ = assign_splits(filings, cutoff=CUTOFF)
    victim = next(f for f in filings if assignment.get(f.item_id) is Split.TRAIN)
    tampered = dict(assignment)
    tampered[victim.item_id] = Split.TEST_POST_CUTOFF

    with pytest.raises(SplitLeakage, match=f"filer {victim.cik}"):
        check_no_leakage(filings, tampered)


def test_the_report_reads_as_a_line_for_the_datasheet() -> None:
    _, report = assign_splits(corpus(), cutoff=CUTOFF)
    line = str(report)
    assert "cutoff 2026-10-01" in line
    assert "test_post_cutoff" in line
    assert "dropped" in line
