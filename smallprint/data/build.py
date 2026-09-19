"""The corpus build: select filings, fetch them, pair them, split them, and write it all down.

Everything upstream of this module decides what one item is. This one decides which filings
are tried, and it is where the numbers the datasheet publishes come from: how many filings
were selected, how many became items, and why each of the others did not.

Selection is by company, not by filing. Companies are ranked by a keyed hash and the first
`companies` of them are taken with every 10-K and 10-Q they filed in the range, because the
split is by company and a company's filing history comes in one request. The hash is keyed
differently from the one `split.py` assigns pools with, so that being selected and landing
in the test pool are independent draws rather than two readings of the same number.

Every request goes through the fair-access client, so the build is paced, cached and
refuses to run without a declared contact. A rebuild from a warm cache runs offline and
reproduces the corpus without asking the SEC for it again.

A filing that cannot be fetched is not an error that stops the build. It is a drop with a
named reason, counted with the others, because a build over thirty thousand filings will
meet a withdrawn document or a company with no XBRL facts, and the right response is to
say how many rather than to stop or to skip them silently.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

from smallprint.data.edgar import EdgarClient, FairAccessViolation, IndexEntry, parse_form_index
from smallprint.data.ixbrl import cover_facts
from smallprint.data.pair import Dropped, DropReason, Item, pair_filing, tally
from smallprint.data.split import FilingRef, Split, SplitReport, assign_splits
from smallprint.data.xbrl import Fact, UnusableFacts, parse_company_facts

#: The forms the task reads. Amendments are left out: a 10-K/A often restates only the part
#: that changed, and its statements, if any, are labelled against the amended facts.
FORMS: Final[frozenset[str]] = frozenset({"10-K", "10-Q"})

#: Salt for company selection. Distinct from the pool assignment's seed by construction.
_SELECTION_SALT: Final = "select"

_QUARTER = re.compile(r"^(\d{4})Q([1-4])$")

#: Filer size bands by total assets, the stratum the pools are balanced across.
_SIZE_BANDS: Final[tuple[tuple[float, str], ...]] = (
    (1e8, "under_100m"),
    (1e9, "100m_to_1b"),
    (1e10, "1b_to_10b"),
)

_FETCH_ERRORS: Final = (httpx.HTTPError, FairAccessViolation, UnusableFacts, ValueError)


def quarter_range(first: str, last: str) -> list[tuple[int, int]]:
    """Every quarter from `first` to `last` inclusive, written like 2022Q1."""
    bounds = []
    for label in (first, last):
        match = _QUARTER.match(label)
        if match is None:
            raise ValueError(f"a quarter is written like 2022Q1; got {label!r}")
        bounds.append((int(match.group(1)), int(match.group(2))))
    (year, quarter), end = bounds
    if (year, quarter) > end:
        raise ValueError(f"{first} is after {last}")
    out = []
    while (year, quarter) <= end:
        out.append((year, quarter))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    return out


def _selection_rank(seed: int, cik: int) -> bytes:
    return hashlib.blake2b(f"{_SELECTION_SALT}:{seed}:{cik}".encode(), digest_size=8).digest()


def select_filings(
    edgar: EdgarClient,
    quarters: Sequence[tuple[int, int]],
    *,
    companies: int | None,
    seed: int,
) -> list[IndexEntry]:
    """The filings to try: every 10-K and 10-Q in range from the selected companies."""
    entries: dict[str, IndexEntry] = {}
    for year, quarter in quarters:
        for entry in parse_form_index(edgar.quarterly_index(year, quarter), FORMS):
            entries.setdefault(entry.accession, entry)
    ciks = sorted({e.cik for e in entries.values()}, key=lambda c: (_selection_rank(seed, c), c))
    chosen = set(ciks if companies is None else ciks[:companies])
    return sorted(
        (e for e in entries.values() if e.cik in chosen),
        key=lambda e: (e.cik, e.filed, e.accession),
    )


def _column(table: object, key: str) -> list[str]:
    if not isinstance(table, Mapping):
        return []
    values = table.get(key)
    return [str(v) for v in values] if isinstance(values, list) else []


def _documents_in(table: object) -> dict[str, str]:
    accessions = _column(table, "accessionNumber")
    documents = _column(table, "primaryDocument")
    return {a: d for a, d in zip(accessions, documents, strict=False) if d}


def primary_documents(edgar: EdgarClient, cik: int, *, since: dt.date) -> dict[str, str]:
    """Accession number to primary document name, back as far as `since`.

    Older pages are fetched only when they reach back into the range, because a large
    filer's history runs to many pages of filings this build never reads.
    """
    payload = edgar.submissions(cik)
    if not isinstance(payload, Mapping):
        raise UnusableFacts(f"submissions for {cik} is not an object")
    filings = payload.get("filings")
    if not isinstance(filings, Mapping):
        raise UnusableFacts(f"submissions for {cik} has no filings section")
    out = _documents_in(filings.get("recent"))
    pages = filings.get("files")
    for page in pages if isinstance(pages, list) else []:
        if not isinstance(page, Mapping):
            continue
        name, until = page.get("name"), page.get("filingTo")
        if not isinstance(name, str) or not isinstance(until, str):
            continue
        if dt.date.fromisoformat(until) >= since:
            out.update(_documents_in(edgar.submissions_page(name)))
    return out


def build_company(
    edgar: EdgarClient, cik: int, entries: Sequence[IndexEntry]
) -> list[Item | Dropped]:
    """Every selected filing of one company, paired or dropped by name."""
    try:
        facts: list[Fact] = parse_company_facts(edgar.company_facts(cik))
        documents = primary_documents(edgar, cik, since=min(e.filed for e in entries))
    except _FETCH_ERRORS:
        return [Dropped(item_id=e.accession, reason=DropReason.NOT_FETCHED) for e in entries]

    results: list[Item | Dropped] = []
    for entry in entries:
        name = documents.get(entry.accession)
        if name is None:
            results.append(Dropped(item_id=entry.accession, reason=DropReason.NO_PRIMARY_DOCUMENT))
            continue
        try:
            document = edgar.document(cik, entry.accession, name)
        except _FETCH_ERRORS:
            results.append(Dropped(item_id=entry.accession, reason=DropReason.NOT_FETCHED))
            continue
        # The document's own cover tags come first, so that if the facts API ever does carry
        # the same concepts, the filing's own reading of its cover page is the one used.
        cover = cover_facts(document, accession=entry.accession, form=entry.form, filed=entry.filed)
        results.append(
            pair_filing(
                document,
                [*cover, *facts],
                cik=cik,
                accession=entry.accession,
                form=entry.form,
                filed=entry.filed,
            )
        )
    return results


def size_band(total_assets: float) -> str:
    for ceiling, label in _SIZE_BANDS:
        if total_assets < ceiling:
            return label
    return "over_10b"


class SplitItem(BaseModel):
    """One line of the published corpus: an item and the split it belongs to."""

    model_config = ConfigDict(frozen=True)

    split: Split
    item: Item


class BuildReport(BaseModel):
    """What a build did, in the form the datasheet quotes it."""

    model_config = ConfigDict(frozen=True)

    built_at: dt.datetime
    first_quarter: str
    last_quarter: str
    forms: tuple[str, ...]
    seed: int
    companies_requested: int | None
    companies_selected: int
    filings_selected: int
    #: Kept and dropped counts by reason, before the split.
    pairing: Mapping[str, int]
    split: SplitReport


def build(
    edgar: EdgarClient,
    *,
    first: str,
    last: str,
    cutoff: dt.date,
    companies: int | None = None,
    seed: int = 20270405,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[SplitItem], list[Dropped], BuildReport]:
    """Select, fetch, pair and split. `cutoff` is the latest base-model training cutoff."""
    entries = select_filings(edgar, quarter_range(first, last), companies=companies, seed=seed)
    by_company: dict[int, list[IndexEntry]] = {}
    for entry in entries:
        by_company.setdefault(entry.cik, []).append(entry)

    results: list[Item | Dropped] = []
    for done, (cik, own) in enumerate(sorted(by_company.items()), start=1):
        results.extend(build_company(edgar, cik, own))
        if progress is not None:
            progress(done, len(by_company))

    items = [r for r in results if isinstance(r, Item)]
    dropped = sorted((r for r in results if isinstance(r, Dropped)), key=lambda d: d.item_id)

    # The stratum has to be constant for a company, because pools are assigned to companies.
    # A company's size band is the median of its own balance sheets.
    assets: dict[int, list[float]] = {}
    for item in items:
        if item.truth.total_assets is not None:
            assets.setdefault(item.cik, []).append(item.truth.total_assets)
    band = {cik: size_band(statistics.median(values)) for cik, values in assets.items()}

    refs = [
        FilingRef(
            item_id=item.item_id,
            cik=item.cik,
            form=item.form,
            filed=item.filed,
            stratum=band.get(item.cik, "unknown"),
        )
        for item in items
    ]
    assignment, split_report = assign_splits(refs, cutoff=cutoff, seed=seed)
    split_items = sorted(
        (
            SplitItem(split=assignment[item.item_id], item=item)
            for item in items
            if item.item_id in assignment
        ),
        key=lambda s: s.item.item_id,
    )

    report = BuildReport(
        built_at=dt.datetime.now(dt.UTC),
        first_quarter=first,
        last_quarter=last,
        forms=tuple(sorted(FORMS)),
        seed=seed,
        companies_requested=companies,
        companies_selected=len(by_company),
        filings_selected=len(entries),
        pairing=tally(results),
        split=split_report,
    )
    return split_items, dropped, report


def _write_lines(path: Path, records: Iterable[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def write_build(
    out_dir: Path,
    items: Sequence[SplitItem],
    dropped: Sequence[Dropped],
    report: BuildReport,
) -> None:
    """items.jsonl, dropped.jsonl and report.json, sorted so two builds diff cleanly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(out_dir / "items.jsonl", items)
    _write_lines(out_dir / "dropped.jsonl", dropped)
    (out_dir / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
