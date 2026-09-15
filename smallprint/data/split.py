"""Splitting filings into train, validation and the two test sets.

Two constraints have to hold at once and they pull against each other.

*No filer is used for both training and testing.* Companies file the same statements
quarter after quarter with the same layout, the same auditor, the same state of
incorporation and numbers that barely move. A model that has seen one of a filer's 10-Qs
has effectively seen the next one, so a split that divides a company's filings between
train and test measures memorisation and calls it accuracy.

*The headline is measured after the base models' training cutoff.* SEC filings are public
and are in every pre-training corpus, so a score on filings a base model may have read is
not evidence about extraction. The post-cutoff set carries the headline and the pre-cutoff
set is reported beside it as the contamination gap.

The two are satisfied in that order. Companies are partitioned into three pools first, and
the cutoff then divides the test pool's own filings in time. So a test filer contributes to
both the pre-cutoff and the post-cutoff set, deliberately: the contamination gap is then a
within-filer comparison, and a difference between the two scores cannot be an artefact of
one set happening to hold easier companies than the other. `PLAN.md` section 3 says only
that no filer appears in both train and test; this module is where that is made exact, and
`check_no_leakage` enforces the pool rule on every build rather than the split rule, which
the two test sets are designed to break.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Pool(StrEnum):
    """What a company may be used for. A company is in exactly one, for ever."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Split(StrEnum):
    """Where a filing ends up. The two test splits share their filers with each other."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_PRE_CUTOFF = "test_pre_cutoff"
    TEST_POST_CUTOFF = "test_post_cutoff"


#: Which pool each split draws from. The leakage check is stated in terms of pools.
POOL_OF: Mapping[Split, Pool] = {
    Split.TRAIN: Pool.TRAIN,
    Split.VALIDATION: Pool.VALIDATION,
    Split.TEST_PRE_CUTOFF: Pool.TEST,
    Split.TEST_POST_CUTOFF: Pool.TEST,
}


class FilingRef(BaseModel):
    """The little a split needs to know about a filing."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    cik: int
    form: str
    filed: dt.date
    #: Any label the pools should be balanced across: a size decile, an industry code, the
    #: fiscal period. The default puts every company in one stratum, which is unstratified
    #: assignment.
    stratum: str = "all"


class SplitLeakage(AssertionError):
    """Raised when a filer is used for more than one purpose."""


class SplitReport(BaseModel):
    """What a split did, in enough detail to put in the datasheet."""

    model_config = ConfigDict(frozen=True)

    cutoff: dt.date
    counts: Mapping[str, int]
    companies: Mapping[str, int]
    #: Filings from training-pool companies dated after the cutoff. They are dropped rather
    #: than used, so the training set holds nothing a reader could mistake for the held-out
    #: period, and the number is published rather than quietly swallowed.
    dropped_after_cutoff: int

    def __str__(self) -> str:
        rows = " | ".join(
            f"{s.value}: {self.counts.get(s.value, 0):,} filings, "
            f"{self.companies.get(s.value, 0):,} filers"
            for s in Split
        )
        return f"cutoff {self.cutoff.isoformat()} | {rows} | dropped {self.dropped_after_cutoff:,}"


def _fraction(seed: int, cik: int) -> float:
    """A stable number in [0, 1) for a company.

    blake2b rather than the built-in hash, which is salted per process, and rather than a
    shuffle, which depends on the order the filings happened to arrive in. A company must
    land in the same pool on every machine and after every rebuild, or the held-out set is
    not held out from the run that came before it.
    """
    digest = hashlib.blake2b(f"{seed}:{cik}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_pools(
    filings: Iterable[FilingRef],
    *,
    seed: int = 20270405,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> dict[int, Pool]:
    """Put each company in exactly one pool, balanced within each stratum.

    Assignment is by rank within a stratum rather than by comparing each company's hash to
    a threshold. With a few hundred companies in a stratum, thresholding gives pools whose
    sizes wander by several percent from the fractions asked for, and the proportions of a
    held-out set are not something to leave to sampling noise.
    """
    if not 0 < test_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("fractions must lie in (0, 1)")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("no training data would be left")

    strata: dict[str, set[int]] = {}
    for filing in filings:
        strata.setdefault(filing.stratum, set()).add(filing.cik)

    assignment: dict[int, Pool] = {}
    for ciks in (strata[key] for key in sorted(strata)):
        ordered = sorted(ciks, key=lambda cik: (_fraction(seed, cik), cik))
        n = len(ordered)
        n_test = round(n * test_fraction)
        n_validation = round(n * validation_fraction)
        # A stratum too small to give a company to every pool still gives one to the test
        # pool, because a stratum present only in training is one the held-out set cannot
        # speak for.
        if n >= 2:
            n_test = max(n_test, 1)
        for i, cik in enumerate(ordered):
            if i < n_test:
                assignment[cik] = Pool.TEST
            elif i < n_test + n_validation:
                assignment[cik] = Pool.VALIDATION
            else:
                assignment[cik] = Pool.TRAIN
    return assignment


def assign_splits(
    filings: Sequence[FilingRef],
    *,
    cutoff: dt.date,
    seed: int = 20270405,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> tuple[dict[str, Split], SplitReport]:
    """Assign every filing to a split, and report what happened.

    `cutoff` is the documented training cutoff of the base models, from `docs/models.md`.
    Filings from test companies dated after it are the headline set.
    """
    pools = assign_pools(
        filings,
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )

    assignment: dict[str, Split] = {}
    dropped = 0
    for filing in filings:
        pool = pools[filing.cik]
        if pool is Pool.TEST:
            assignment[filing.item_id] = (
                Split.TEST_POST_CUTOFF if filing.filed > cutoff else Split.TEST_PRE_CUTOFF
            )
        elif filing.filed > cutoff:
            # A training filing from after the cutoff sits in the same period as the
            # headline test set. Nothing about it is unusable, but a reader should not have
            # to take on trust that it was not the reason the score held up.
            dropped += 1
        else:
            assignment[filing.item_id] = Split.TRAIN if pool is Pool.TRAIN else Split.VALIDATION

    by_id = {f.item_id: f for f in filings}
    counts = Counter(assignment.values())
    companies = {
        split: len({by_id[i].cik for i, s in assignment.items() if s is split}) for split in Split
    }

    report = SplitReport(
        cutoff=cutoff,
        counts={s.value: counts.get(s, 0) for s in Split},
        companies={s.value: companies[s] for s in Split},
        dropped_after_cutoff=dropped,
    )
    check_no_leakage(filings, assignment)
    return assignment, report


def check_no_leakage(filings: Iterable[FilingRef], assignment: Mapping[str, Split]) -> None:
    """Raise if any filer is used for more than one purpose.

    Called by `assign_splits` on every build rather than only by the test suite. A split
    file is the kind of artefact that gets hand-edited once, late at night, to put back a
    few filings that were dropped, and that edit is exactly what this refuses to let pass.
    """
    seen: dict[int, tuple[Pool, str]] = {}
    for filing in filings:
        split = assignment.get(filing.item_id)
        if split is None:
            continue
        pool = POOL_OF[split]
        previous = seen.get(filing.cik)
        if previous is None:
            seen[filing.cik] = (pool, filing.item_id)
        elif previous[0] is not pool:
            raise SplitLeakage(
                f"filer {filing.cik} is used for both {previous[0].value} and {pool.value}: "
                f"filings {previous[1]} and {filing.item_id}"
            )
