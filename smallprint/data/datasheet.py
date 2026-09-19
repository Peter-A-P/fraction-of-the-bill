"""The datasheet: what the published corpus is, how it was made, and what it leaves out.

Written from a build's own outputs, never by hand, so that every number in it is the number
the build produced. It goes out with the dataset on Hugging Face next to a checksum file,
and the repository keeps the code that writes it rather than a copy of it.

The sections follow the questions a reader of a dataset has to be able to answer before
trusting a score measured on it: where the text and the labels come from, which filings are
missing and why, how the splits keep a company out of both training and test, and which
populations the corpus underrepresents on purpose.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from smallprint.data.build import BuildReport, SplitItem
from smallprint.data.split import Split
from smallprint.schema import SCHEMA

#: The files a build writes, in the order the checksum file lists them.
BUILD_FILES: Final[tuple[str, ...]] = ("items.jsonl", "dropped.jsonl", "report.json")

REPOSITORY: Final = "https://github.com/Peter-A-P/fraction-of-the-bill"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksums(build_dir: Path) -> str:
    """`sha256sum` format, so a reader can check the download with the standard tool."""
    return "".join(f"{sha256_of(build_dir / name)}  {name}\n" for name in BUILD_FILES)


def read_items(build_dir: Path) -> list[SplitItem]:
    with (build_dir / "items.jsonl").open(encoding="utf-8") as handle:
        return [SplitItem.model_validate_json(line) for line in handle if line.strip()]


def read_report(build_dir: Path) -> BuildReport:
    return BuildReport.model_validate_json((build_dir / "report.json").read_text(encoding="utf-8"))


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def field_coverage(items: Sequence[SplitItem]) -> list[tuple[str, str, int]]:
    """How often each field has a label, per field. Optional fields are often null."""
    rows = []
    for spec in SCHEMA:
        labelled = sum(1 for s in items if getattr(s.item.truth, spec.name) is not None)
        rows.append((spec.name, _share(labelled, len(items)), labelled))
    return rows


def render(report: BuildReport, items: Sequence[SplitItem], sums: str) -> str:
    """The datasheet as Markdown."""
    forms = Counter((s.split, s.item.form) for s in items)
    periods = Counter(s.item.fiscal_period for s in items)
    kept = report.pairing.get("kept", 0)
    dropped_total = sum(v for k, v in report.pairing.items() if not k.startswith("kept"))
    chars = sorted(len(s.item.text) for s in items)
    median_chars = chars[len(chars) // 2] if chars else 0
    # No limit means every company in range; the rebuild command says so by omitting it.
    companies = (
        "" if report.companies_requested is None else f" --companies {report.companies_requested}"
    )

    split_rows = [
        (
            split.value,
            f"{report.split.counts.get(split.value, 0):,}",
            f"{report.split.companies.get(split.value, 0):,}",
            f"{forms.get((split, '10-K'), 0):,}",
            f"{forms.get((split, '10-Q'), 0):,}",
        )
        for split in Split
    ]
    drop_rows = [
        (f"`{reason}`", f"{count:,}", _share(count, report.filings_selected))
        for reason, count in report.pairing.items()
        if not reason.startswith("kept")
    ]

    return f"""# Datasheet: smallprint SEC filings extraction corpus

Structured extraction from the primary financial statements of SEC 10-K and 10-Q filings
into fifteen fields, each labelled with the XBRL fact the company itself filed. Built by
[smallprint]({REPOSITORY}) on {report.built_at:%Y-%m-%d}. Every number below was written by
the build; none was edited by hand.

## Composition

**{len(items):,} items** from {report.companies_selected:,} selected companies, filings
dated {report.first_quarter} to {report.last_quarter}, forms {", ".join(report.forms)}.
Of {report.filings_selected:,} filings tried, {kept:,} became items before the split and
{dropped_total:,} were dropped with a named reason (below). The split then drops a further
{report.split.dropped_after_cutoff:,} training-pool filings dated after the cutoff.

Each item is one filing: `text`, the model's input (the cover page, the income statement,
the balance sheet and, in an annual report, the audit signature, located and rendered from
the filing's HTML with hidden inline XBRL removed); `truth`, the fifteen labels; and
`context`, the reporting scale and the other periods' values of each line, which the grader
uses to name a wrong-period reading. The median input is {median_chars:,} characters.

## Splits

Companies, not filings, are assigned to pools, so no company appears in both training and
test. The cutoff, **{report.split.cutoff.isoformat()}**, is the latest documented training
cutoff of the base models; test-pool filings after it are the headline set, and before it
the contamination comparison. Test filers appear in both test sets by design.

{_table(("Split", "Filings", "Companies", "10-K", "10-Q"), split_rows)}

Fiscal periods: {", ".join(f"{p} {n:,}" for p, n in sorted(periods.items()))}.

## Where the labels come from

Numeric labels are XBRL facts from the SEC `companyfacts` API for the filing's own
accession, for the period the report covers; cover-page labels are the filing's own inline
XBRL tags. No model produced or checked any label. A filing is kept only when every label is
printed in the section of the input its field is read from, within the grader's tolerance
after the printed scale, so the task is extraction and not inference. The one exception is
diluted weighted shares: when the income statement prints no share-count line at all, that
label is null and the filing is kept
({report.pairing.get("kept_with_null:shares_diluted", 0):,} filings). The rules and their evidence are in
[docs/data.md]({REPOSITORY}/blob/main/docs/data.md).

## Label coverage

{_table(("Field", "Labelled", "Items"), field_coverage(items))}

## What is missing, and why

{_table(("Reason", "Filings", "Share of those tried"), drop_rows)}

**Underrepresented on purpose.** Banks and savings institutions (industry codes 6021, 6022,
6029, 6035, 6036) are excluded: they report no single revenue line to label. Funds, trusts,
business development companies and SPACs mostly drop out because they print no income
statement of the kind the task reads, or no revenue. Filers whose statements print no
reporting scale are dropped rather than labelled by guessing it. The corpus therefore
underweights financial companies and very small or pre-revenue ones, and a score on it
says nothing about extraction from their statements.

## Collection

Fetched from SEC EDGAR under its fair-access policy: a declared contact address in every
request, paced below ten requests a second, every response cached with its date and digest
so the corpus rebuilds offline without refetching. Company selection is a keyed hash,
seed {report.seed}. Rebuild with:

    smallprint data build --first {report.first_quarter} --last {report.last_quarter} \\
      --cutoff {report.split.cutoff.isoformat()}{companies} --seed {report.seed}

## Licence

SEC filings and XBRL data are public records of the US government. **The licence under which
this corpus is published is decided before publication and recorded here.**

## Checksums

    {sums.strip().replace(chr(10), chr(10) + "    ")}
"""


def write_datasheet(build_dir: Path) -> tuple[Path, Path]:
    """Write DATASHEET.md and SHA256SUMS into the build directory."""
    sums = checksums(build_dir)
    report = read_report(build_dir)
    items = read_items(build_dir)
    sheet = build_dir / "DATASHEET.md"
    sums_path = build_dir / "SHA256SUMS"
    sheet.write_text(render(report, items, sums), encoding="utf-8", newline="\n")
    sums_path.write_text(sums, encoding="utf-8", newline="\n")
    return sheet, sums_path


def verify(build_dir: Path) -> list[str]:
    """Files whose digest no longer matches SHA256SUMS. Empty means the build is intact."""
    expected = {}
    for line in (build_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    return [name for name, digest in expected.items() if sha256_of(build_dir / name) != digest]
