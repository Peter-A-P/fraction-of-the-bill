"""Small hand-made items, for the tests that are about what is done with an item rather
than about how one is made. The build's own items come from the mock EDGAR in
`test_build.py`, which is slower and says nothing extra about a prompt or a runner."""

from __future__ import annotations

import datetime as dt

from smallprint.data.build import SplitItem
from smallprint.data.pair import Item
from smallprint.data.split import Split
from smallprint.grade import GradeContext
from smallprint.schema import Extraction

TRUTH = Extraction(
    period_end=dt.date(2025, 12, 31),
    fiscal_period="FY",
    revenue=1_200_000.0,
    cost_of_revenue=700_000.0,
    operating_income=300_000.0,
    net_income=210_000.0,
    eps_basic=2.1,
    eps_diluted=2.05,
    shares_diluted=100_000.0,
    total_assets=4_000_000.0,
    total_liabilities=1_500_000.0,
    cash_and_equivalents=800_000.0,
    stockholders_equity=2_500_000.0,
    auditor_name="Deloitte & Touche LLP",
    state_of_incorporation="DE",
)


def make_item(
    item_id: str,
    *,
    split: Split = Split.TRAIN,
    form: str = "10-K",
    text: str | None = None,
    truth: Extraction = TRUTH,
) -> SplitItem:
    body = text if text is not None else f"[COVER]\n{item_id}\n[INCOME STATEMENT]\nRevenue | 1,200"
    return SplitItem(
        split=split,
        item=Item(
            item_id=item_id,
            cik=1234,
            form=form,
            filed=dt.date(2026, 2, 20),
            period_end=dt.date(2025, 12, 31),
            fiscal_period="FY" if form == "10-K" else "Q3",
            text=body,
            truth=truth if form == "10-K" else truth.model_copy(update={"auditor_name": None}),
            context=GradeContext(scale=1000.0),
        ),
    )


def training_pool(n: int = 12) -> list[SplitItem]:
    """Half annual, half quarterly, with lengths that differ so the shorter half is a choice."""
    pool = []
    for i in range(n):
        form = "10-K" if i % 2 == 0 else "10-Q"
        pool.append(make_item(f"train-{i:02d}", form=form, text="x" * (100 + i * 10)))
    return pool
